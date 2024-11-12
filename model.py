import torch
from litgpt.model import Config
from transformers import AutoModelForCausalLM, AutoConfig
from pathlib import Path
import sys
from litgpt.utils import load_checkpoint
sys.path.append("../external")

from litgpt_utils.model import GPT


def get_hf_kwargs(use_flash_attention):
    return {
            "attn_implementation": (
                    "eager" if not use_flash_attention else "flash_attention_2"
                ),
            "trust_remote_code" : True,
    }


def get_model(model_id, model_impl, fabric, litgpt_checkpoint_directory=None, 
        random_init:bool=False, use_flash_attention:bool=False):
    assert model_impl in ["hf", "litgpt"]
    with fabric.init_module(empty_init=False):
        if model_impl == "hf":
            kwargs = get_hf_kwargs(use_flash_attention)
            if not random_init:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, **kwargs
                )
            else:
                config = AutoConfig.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_config(
                    config, **kwargs
                )
            model = fabric.setup_module(model)
            model.train()
            model.gradient_checkpointing_enable()
        elif model_impl == "litgpt":
            checkpoint_dir = Path(litgpt_checkpoint_directory)
            config = Config.from_file(checkpoint_dir / "model_config.yaml")
            model = GPT(config, False)
            model = fabric.setup_module(model)
            if not random_init:
                checkpoint_path = checkpoint_dir / "lit_model.pth"
                load_checkpoint(fabric, model, checkpoint_path)
            model.train()
    return model
