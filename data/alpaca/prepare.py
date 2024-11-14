import os
from datasets import load_dataset
from transformers import AutoTokenizer
from alpaca_data_utils import get_tokenizer_mapping_fn
from argparse import ArgumentParser


def create_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_id",
        default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        type=str,
        help="name of huggingface transformers model whose tokenizer you want to use",
    )
    parser.add_argument(
        "--sequence-length", type=int, default=256, help="Sequence Length"
    )

    return parser


def get_tokenized_dataset(tokenizer, sequence_length):
    data = load_dataset("yahma/alpaca-cleaned")
    mapping_fn = get_tokenizer_mapping_fn(
        tokenizer, cutoff_len=sequence_length, train_on_inputs=False
    )
    train_data = (
        data["train"]
        .shuffle()
        .map(
            mapping_fn,
            remove_columns=data["train"].column_names,
            num_proc=os.cpu_count() // 2 if os.cpu_count() // 2 else 8,
        )
    )
    return train_data


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    model_id = args.model_id
    max_seq_len = args.sequence_length

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    train_data = get_tokenized_dataset(tokenizer, max_seq_len)
    train_data.save_to_disk(f"./{model_id}")
