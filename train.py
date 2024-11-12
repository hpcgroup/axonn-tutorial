# Initializing mpi4py is necessary to use
# pytorch lightning within interactive slurm
# sesions
try:
    from mpi4py import MPI
except ImportError:
    pass
import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    DataCollatorForSeq2Seq,
)
import torch
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lightning.fabric import Fabric, seed_everything
from axonn.lightning import AxonnStrategy
from lightning.fabric.strategies import FSDPStrategy
from transformers.utils import logging
import sys


from utils import (
    all_reduce_avg,
    color,
    get_fsdp_block_name_for_hf,
    print_axonn_timer_data,
)
from args import parse_json_args
#from manual_tuning import llama
from model import get_model
sys.path.append("../external/")
from litgpt_utils.model import Block

logging.set_verbosity_error()


def init_everything(
    dtype, num_nodes, strategy="axonn", dims=None, enable_axonn_timers=True
):
    torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    if strategy == "axonn":
        if not dims:
            dims = [1, 1, world_size]

        pl_strategy = AxonnStrategy(
            G_intra_r=dims[0],
            G_intra_c=dims[1],
            G_intra_d=dims[2],
            overlap_communication=True,
            enable_timers=enable_axonn_timers,
        )
    elif strategy == "fsdp":
        pl_strategy = FSDPStrategy(
            auto_wrap_policy={get_fsdp_block_name_for_hf(args.model_id), Block},
            state_dict_type="sharded",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    elif strategy == "deepspeed":
        pl_strategy = "deepspeed_stage_3"

    fabric = Fabric(
        strategy=pl_strategy,
        devices=torch.cuda.device_count(),
        num_nodes=num_nodes,
        precision=dtype,
    )
    fabric.launch()

    if torch.distributed.get_rank() == 0:
        print(f"Going to distribute the model over {world_size} GPUs")

    return fabric

def create_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default="sample_args_file.json",
        help="Name of JSON file with args",
    )
    return parser


def get_tokenized_dataset(dataset, tokenizer, sequence_length):
    assert dataset in ["alpaca", "ultrachat", "ultrachat_200k"]

    data_dir = os.path.join("data", dataset, args.model_id)
    try:
        return load_from_disk(data_dir)
    except Exception as e:
        raise Exception(
            f"Dataset not tokenized. cd into 'data/{dataset}' and run 'python prepare.py --model_id {args.model_id}' to tokenize dataset"
        )


def pretty_log(
    iteration,
    total_train_iters,
    train_loss,
    elapsed_time_per_iteration,
    grad_norm,
    learning_rate,
    wandb_log=False,
):
    log_string = "> global batch {:8d}/{:8d} |".format(iteration, total_train_iters)
    log_string += " elapsed time per global batch (ms): {:.1f} |".format(
        elapsed_time_per_iteration
    )
    log_string += " learning rate: {:.3E} |".format(learning_rate)
    log_string += " loss: {:.5f} |".format(train_loss)
    curr_mem = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    log_string += " memory used by tensors {:.3f} GB (peak {:.3f} GB) |".format(
        curr_mem, peak_mem
    )
    log_string += " grad_norm: {:.2f}".format(grad_norm)
    if wandb_log:
        import wandb

        wandb.log(
            {
                "iter": iteration,
                "train/loss": train_loss,
                "lr": learning_rate,
                "peak_memory": peak_mem,
                "state_memory": curr_mem,
                "iteration_time": elapsed_time_per_iteration,
            }
        )
    return log_string


if __name__ == "__main__":
    parser = create_parser()
    parser_args = parser.parse_args()
    args = parse_json_args(parser_args.file)
    fabric = init_everything(
        args.dtype, args.num_nodes, args.strategy, args.axonn_dimensions
    )
    seed_everything(args.seed)
    if args.wandb_log and torch.distributed.get_rank() == 0:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    model =  get_model(model_id=args.model_id, 
                model_impl=args.model_impl, 
                fabric=fabric, 
                litgpt_checkpoint_directory=args.litgpt_checkpoint_directory, 
                random_init=args.random_init, 
                use_flash_attention=args.use_flash_attention)
    #model = fabric.setup_module(model)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.0
    )

    optimizer = fabric.setup_optimizers(optimizer)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    tokenized_dataset = get_tokenized_dataset(
        args.dataset_id, tokenizer, args.sequence_length
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.global_batch_size
        // args.gradient_acc_steps
        // torch.distributed.get_world_size(),
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )

    dataloader = fabric.setup_dataloaders(dataloader)

    iters_per_epoch = len(dataloader) // args.gradient_acc_steps
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=iters_per_epoch * args.num_epochs
    )
    warmup_iters = 100
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_iters
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, main_lr_scheduler],
        milestones=[warmup_iters],
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iter_no = 0

    if torch.distributed.get_rank() == 0:
        print(f"\n\n{color.PURPLE} Configuration = {args} {color.END}\n\n")

    state_dict = model.state_dict()
    if torch.distributed.get_rank() == 0:
        for key, val in state_dict.items():
            if torch.is_tensor(val):
                print(key, val.shape)

    exit()
    all_iter_times = []
    iters = 0
    early_finished = False

    for epoch_no in range(args.num_epochs):
        if early_finished:
            break
        microbatch_no = 0
        start_event.record()

        batch_loss = torch.tensor([0.0], dtype=torch.float32, device="cuda")

        for batch in dataloader:
            input_ids, labels, attention_mask = (
                batch["input_ids"].cuda(),
                batch["labels"].cuda(),
                batch["attention_mask"].cuda(),
            )
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            labels = labels[:, 1:]
            output = model(input_ids=input_ids[:,:2048], attention_mask=attention_mask)
            logits = output["logits"]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]).float(), labels[:,:2048].reshape(-1))

            fabric.backward(loss / args.gradient_acc_steps, model=model)
            batch_loss += loss / args.gradient_acc_steps

            microbatch_no += 1
            if microbatch_no == args.gradient_acc_steps:
                grad_norm = fabric.clip_gradients(model, optimizer, max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                iter_no += 1
                end_event.record()

                batch_loss = all_reduce_avg(batch_loss)

                if args.strategy == "axonn":
                    times, _ = fabric._strategy.get_timers().get_times()

                if torch.distributed.get_rank() == 0 and (
                    iter_no % args.log_interval == 0
                ):
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event)

                    log_string = pretty_log(
                        iter_no,
                        len(dataloader) * args.num_epochs // args.gradient_acc_steps,
                        batch_loss.item(),
                        elapsed_time,
                        grad_norm=grad_norm,
                        learning_rate=optimizer.param_groups[0]["lr"],
                        wandb_log=args.wandb_log,
                    )
                    print(log_string)
                    if args.strategy == "axonn":
                        print_axonn_timer_data(times)

                    all_iter_times.append(elapsed_time)
                    if args.max_iters > 0 and len(all_iter_times) == args.max_iters:
                        # skip first iteration as it has expensive one-time cuda mem allocs
                        mean_iter_time = np.mean(all_iter_times[1:])
                        std_iter_time = np.std(all_iter_times[1:])
                        if torch.distributed.get_rank() == 0:
                            print(
                                f"\n\n {color.PURPLE} Mean Iter Time = {mean_iter_time:.3f} +- {std_iter_time:.3f} ms {color.END}"
                            )
                        early_finished = True
                        break

                microbatch_no = 0
                batch_loss = 0
                start_event.record()
    
        torch.distributed.destroy_process_group()
        # state = {"optimizer": optimizer.state_dict(), "model": model.state_dict()}

        # fabric.save(f"./ckpt/ckpt_model_{epoch_no}.pt", state["model"])
        # fabric.save(f"./ckpt/ckpt_optimizer_{epoch_no}.pt", state["optimizer"])
