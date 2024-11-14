import os
from datasets import load_from_disk
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
import torch
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from lightning.fabric import Fabric, seed_everything
from axonn.lightning import AxonnStrategy
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from transformers.utils import logging
from utils import all_reduce_avg, pretty_log
from args import parse_json_args
from model import get_model
from external.model import Block
import pprint

logging.set_verbosity_error()


def init_everything(dtype, strategy, axonn_dims):
    # initialize torch distributed
    torch.distributed.init_process_group(backend="nccl")
    world_size = torch.distributed.get_world_size()
    
    # create pytorch lightning strategy
    if strategy == "single_device":
        pl_strategy = "auto"
    elif strategy == "ddp":
        pl_strategy = DDPStrategy()
    elif strategy == "fsdp":
        pl_strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            state_dict_type="sharded",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    elif strategy == "axonn":
        pl_strategy = AxonnStrategy(
            G_intra_x=axonn_dims[0],
            G_intra_y=axonn_dims[1],
            G_intra_z=axonn_dims[2],
            overlap_communication=True,
        )

    # create lightning fabric object
    fabric = Fabric(
        strategy=pl_strategy,
        devices=1 if strategy == "single_device" else torch.cuda.device_count(),
        num_nodes=1,
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


def get_dataloader(args):
    data_dir = os.path.join(os.getenv("SCRATCH", "data"), "alpaca", args.model_id)
    try:
        tokenized_dataset = load_from_disk(data_dir)
    except Exception as e:
        raise Exception(
            f"Dataset not tokenized. cd into 'data/alpaca' and run 'python prepare.py --model_id {args.model_id}' to tokenize dataset"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.global_batch_size
        // args.gradient_acc_steps
        // torch.distributed.get_world_size(),
        collate_fn=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    return dataloader

def get_lr_scheduler(total_train_iters, warmup_iters=100):
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_train_iters
    )
    warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_iters
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_lr_scheduler, main_lr_scheduler],
        milestones=[warmup_iters],
    )
    return lr_scheduler

if __name__ == "__main__":
    # Parse arguments
    parser = create_parser()
    parser_args = parser.parse_args()
    args = parse_json_args(parser_args.file)
    
    # Create lightning fabric object
    fabric = init_everything(args.dtype, args.strategy, args.axonn_dimensions)
    seed_everything(args.seed)

    # Create model
    model = get_model(
        fabric=fabric,
        litgpt_checkpoint_directory=os.path.join(
            os.getenv("SCRATCH", "./external/"), f"checkpoints/{args.model_id}"
        ),
        random_init=args.random_init,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-5, betas=(0.9, 0.95), eps=1e-5, weight_decay=0.0
    )
    optimizer = fabric.setup_optimizers(optimizer)

    # Create dataloader
    dataloader = get_dataloader(args)
    dataloader = fabric.setup_dataloaders(dataloader)

    # Create learning rate scheduler
    iters_per_epoch = len(dataloader) // args.gradient_acc_steps
    lr_scheduler = get_lr_scheduler(iters_per_epoch * args.num_epochs)

    # Create loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Cuda events for timing each batch/iteration
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    iter_no = 0

    # Printing arguments from rank 0
    if torch.distributed.get_rank() == 0:
        print("\n\n\n")
        pprint.pprint(args)
        print("\n\n\n")

    for epoch_no in range(args.num_epochs):
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
            labels = labels[:, 1:]
            
            # forward pass
            output = model(input_ids=input_ids[:, :model.max_seq_length])
            logits = output["logits"]
            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]).float(), #loss function should be computed in float for stability
                labels[:, :model.max_seq_length].reshape(-1),
            )

            # backward pass
            fabric.backward(loss / args.gradient_acc_steps, model=model)
            batch_loss += loss / args.gradient_acc_steps

            microbatch_no += 1
            if microbatch_no == args.gradient_acc_steps:
                # gradient clipping
                grad_norm = fabric.clip_gradients(model, optimizer, max_norm=1.0)
                # optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                lr_scheduler.step()
                
                iter_no += 1
                end_event.record()

                batch_loss = all_reduce_avg(batch_loss)

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
                    )
                    print(log_string)

                microbatch_no = 0
                batch_loss = 0
                start_event.record()

    torch.distributed.destroy_process_group()
