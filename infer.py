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
from external.generate import generate_text

# for pretty printing
BLUE = '\033[94m'
GREEN = '\033[92m'
ENDC = '\033[0m'

def print_rank0(msg):
    if dist.get_rank() == 0:
        print(f"{msg}")

def init_everything(precision, strategy, tp_dimensions):
    # initialize torch distributed
    rank = int(os.getenv("SLURM_PROCID", 0))
    world_size = int(os.getenv("SLURM_NTASKS", 1))
    gpus_per_node = int(os.getenv("SLURM_NTASKS_PER_NODE", 1))
    torch.distributed.init_process_group(rank=rank, 
            world_size=world_size, 
            backend="nccl")
 
    assert strategy == "axonn", "Inference has been setup just for axonn"
    assert tp_dimensions[2] == 1, "Inference doesn't support z tensor parallelism"
    assert precision == "bf16-mixed"
    pl_strategy = AxonnStrategy(
        G_intra_x=tp_dimensions[0],
        G_intra_y=tp_dimensions[1],
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
    checkpoint_dir = Path(litgpt_checkpoint_directory)
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    model = GPT(config)
    model = model.to(torch.bfloat16).to("cuda")
    if not random_init:
        checkpoint_path = checkpoint_dir / "lit_model.pth"
        load_checkpoint(fabric, model, checkpoint_path)
    model.train()
    return model

if __name__ == "__main__":
    # Parse arguments
    parser = create_parser()
    parser_args = parser.parse_args()
    args = parse_json_args(parser_args.config_file)
    # Create lightning fabric object
    fabric = init_everything(args.precision, args.strategy, args.tp_dimensions)
    seed_everything(args.seed)

    # Create model
    model = get_model(
        fabric=fabric,
        litgpt_checkpoint_directory=os.path.join(
            os.getenv("SCRATCH", "./external/"), f"checkpoints/{args.model_id}"
        ),
        random_init=args.random_init,
    )
    # Setup input position and model's KV cache for fast generation
    model.set_kv_cache(batch_size=1, device='cuda', dtype=torch.bfloat16)

    with open("data/inference/prompts.txt", 'r') as file:
        prompts = [line.strip() for line in file if line.strip()]

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    for user_prompt in prompts:
        system_prompt = "You are a helpful chatbot. Answer the following question.\n"
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        formatted_prompt = tokenizer.apply_chat_template(conversation, 
                add_generation_prompt=True, 
                tokenize=False)
        start = time.time()
        generated_text, tokens_per_second = generate_text(model=model,
                prompt=formatted_prompt, 
                compile=args.compile, 
                tokenizer=tokenizer, 
                terminators=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                max_tokens_to_gen=256)

        print_rank0(f"-"*40 + "\n")
        print_rank0(f"{BLUE}User: {user_prompt}{ENDC}")
        print_rank0(f"{GREEN}AI Assistant: {generated_text}{ENDC}")
        print_rank0(f"\nRate of Token Generation = {tokens_per_second:.2f} tokens/second")
