import json
from types import SimpleNamespace


def parse_json_args(filename):
    args = {
        "model_id": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "model_impl": "hf",
        "litgpt_checkpoint_directory": "",
        "dataset_id": "ultrachat",
        "seed": 123456,
        "dtype": "bf16-mixed",
        "strategy": "axonn",
        "use_flash_attention": True,
        "axonn_dimensions": [],
        "global_batch_size": 4,
        "gradient_acc_steps": 1,
        "sequence_length": 2048,
        "num_nodes": 1,
        "log_interval": 10,
        "num_epochs": 1,
        "wandb_log": False,
        "wandb_project": "",
        "wandb_run_name": "",
        "max_iters": -1,
        "random_init": False,
    }

    user_args = {}
    with open(filename) as f:
        try:
            user_args = json.load(f)
        except:
            print("Invalid JSON given. ")
        args.update((k, user_args[k]) for k in args.keys() & user_args.keys())

    # convert dict to object
    args = SimpleNamespace(**args)

    return args
