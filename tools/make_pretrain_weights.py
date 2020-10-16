import sys
from pathlib import Path
root = Path("~/Desktop/MaxMin_Pytorch").expanduser()
sys.path.append(str(root / "src"))
import torch
import argparse
import json5
import os
import model_wrapper

parser = argparse.ArgumentParser(description="Wave-U-Net for Speech Enhancement")
parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
args = parser.parse_args()

with open(root / "configs" / args.configuration) as file:
    config = json5.load(file)
    config['root'] = root
    config["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    print(config["experiment_name"])

pkg = torch.load(root / "experiments" / config["experiment_name"] / "checkpoints" / "best_model.pth")
model = model_wrapper.Model(config)
model.load_state_dict(pkg["model"])

os.makedirs(root / "pretrained" / config["experiment_name"], exist_ok=True)
torch.save(model.encoder.state_dict(), root / "pretrained" / config["experiment_name"] / "encoder.pth")
torch.save(model.backbone.state_dict(), root / "pretrained" / config["experiment_name"] / "backbone.pth")
torch.save(model.classifier.state_dict(), root / "pretrained" / config["experiment_name"] / "classifier.pth")
