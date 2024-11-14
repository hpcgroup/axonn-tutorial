import json
from types import SimpleNamespace


def parse_json_args(filename):
    args = {
        "model_id": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "dataset_id": "alpaca",
        "seed": 123456,
        "dtype": "32",
        "strategy": "axonn",
        "axonn_dimensions": [],
        "global_batch_size": 4,
        "gradient_acc_steps": 1,
        "sequence_length": 2048,
        "log_interval": 1,
        "num_epochs": 1,
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
