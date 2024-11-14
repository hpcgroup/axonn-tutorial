from litgpt.model import Config
from pathlib import Path
import torch


from model import GPT

if __name__ == "__main__":
    checkpoint_dir = Path(
        "./checkpoints/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
    )
    config = Config.from_file(checkpoint_dir / "model_config.yaml")
    model = GPT(config)

    checkpoint_path = checkpoint_dir / "lit_model.pth"
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)
    print("successfully loaded model")
    print(model.state_dict().keys())
