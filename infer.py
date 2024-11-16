from lightning.fabric import Fabric, seed_everything
from axonn.lightning import AxonnStrategy
from transformers import AutoTokenizer
import torch
import torch.distributed as dist
import time
from pathlib import Path
from litgpt.utils import load_checkpoint
from litgpt.model import Config
import os

from train import init_everything, create_parser
from args import parse_json_args
from external.inference_optimized_model import GPT, Block

def print_rank0(msg):
    if dist.get_rank() == 0:
        print(f"{msg}")

def init_everything(precision, strategy, axonn_dims):
    # initialize torch distributed
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", 1))
    gpus_per_node = int(os.getenv("SLURM_NTASKS_PER_NODE", 1))
    torch.distributed.init_process_group(rank=rank, 
            world_size=world_size, 
            backend="nccl")
 
    assert strategy == "axonn", "Inference has been setup just for axonn"
    assert axonn_dims[2] == 1, "Inference doesn't support z tensor parallelism"
    assert precision == "bf16-mixed"
    pl_strategy = AxonnStrategy(
        G_intra_x=axonn_dims[0],
        G_intra_y=axonn_dims[1],
        G_intra_z=1,
        overlap_communication=True,
    )

    # create lightning fabric object
    fabric = Fabric(
        strategy=pl_strategy,
        devices=gpus_per_node,
        num_nodes=1,
        precision=precision,
    )
    fabric.launch()

    if torch.distributed.get_rank() == 0:
        print(f"Going to distribute the model over {world_size} GPUs")

    return fabric

def get_model(fabric, litgpt_checkpoint_directory, random_init: bool = False):
    with fabric.init_module(empty_init=True): 
        # empty_init=True initializes meta tensors on the CPU i.e.
        # tensors with no data
        checkpoint_dir = Path(litgpt_checkpoint_directory)
        config = Config.from_file(checkpoint_dir / "model_config.yaml")
        model = GPT(config)
    # setup_module moves the model to the GPU. Actual model tensors 
    # are created in this step, directly on the GPU. 
    model = fabric.setup_module(model).to(torch.bfloat16)
    if not random_init:
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        load_checkpoint(fabric, model, checkpoint_path)
    model.train()
    return model

@torch.no_grad()
def prefill(model, tokens):
    # Forward pass through the model
    input_pos = torch.arange(0, tokens.size(0), device="cuda", dtype=torch.int64)
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    #token_id = torch.distributions.Categorical(logits=logits[0, -1]).sample()
    token_id = torch.argmax(logits[0, -1])
    return token_id
    
@torch.no_grad()
def generate(model, tokens, input_pos):
    # Forward pass through the model
    logits = model(tokens.view(1, -1), input_pos)["logits"]
    #token_id = torch.distributions.Categorical(logits=logits[0, -1]).sample()
    token_id = torch.argmax(logits[0, -1])
    return token_id

if __name__ == "__main__":
    # Parse arguments
    parser = create_parser()
    parser_args = parser.parse_args()
    args = parse_json_args(parser_args.config_file)
    
    # Create lightning fabric object
    fabric = init_everything(args.precision, args.strategy, args.axonn_dimensions)
    seed_everything(args.seed)

    # Create model
    model = get_model(
        fabric=fabric,
        litgpt_checkpoint_directory=os.path.join(
            os.getenv("SCRATCH", "./external/"), f"checkpoints/{args.model_id}"
        ),
        random_init=args.random_init,
    )

    # Initialize prompt and tokenizer
    prompt = "You are a helpful chatbot. Answer the following question.\nHow to bake a cake?"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    prompt_tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].squeeze().cuda()
    tokens_to_gen = 256

    # Print the initial prompt details
    print_rank0(f"Initial prompt: '{prompt}'")
    #print(f"Tokenized prompt (IDs): {tokens.tolist()}")

    # Setup input position and model's KV cache for fast generation
    model.set_kv_cache(batch_size=1, device='cuda', dtype=torch.bfloat16)

    # Generation loop
    #print("\nStarting token generation:")
    for TRIAL in range(10):
        start = time.time()
        tokens = prompt_tokens
        output_tokens = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=True):
            for step in range(tokens_to_gen):
                if step == 0: # prefill
                    next_token = prefill(model, tokens)
                    input_pos = torch.tensor([tokens.size(0)], device="cuda", dtype=torch.int64)
                else:
                    next_token = generate(model, tokens, input_pos)
                    input_pos.add_(1)
                # Append token to output and log details
                output_tokens.append(next_token.item())
                tokens = next_token.clone()
        end = time.time()
        time_taken = end - start
        # Decode and display the final generated output
        generated_text = tokenizer.decode(output_tokens)
        if TRIAL == 0:
            print_rank0("\nGenerated text:\n" + "-" * 40)
            print_rank0(generated_text)
            print_rank0("-" * 40)

        tokens_per_second = len(output_tokens) / time_taken
        print_rank0(f"Output {tokens_per_second} tok/s") 
