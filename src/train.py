import argparse
import os

import json5
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import initialize_config

import data, loss, metric, model_wrapper, trainer

def main(config, resume, debug):
    torch.manual_seed(0)  # for both CPU and GPU
    np.random.seed(0)

    dataset_config = config["dataset"]
    trainset = getattr(data, dataset_config["type"])(dataset_config["train"], **dataset_config["args"], debug=debug)
    valset = getattr(data, dataset_config["type"])(dataset_config["val"], **dataset_config["args"], debug=debug)

    train_dataloader = DataLoader(dataset=trainset, **config["dataloader"]["args"])
    valid_dataloader = DataLoader(dataset=valset, **config["dataloader"]["args"])

    model = model_wrapper.Model(config)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["optimizer"]["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config["scheduler"]["step_size"], config["scheduler"]["gamma"])

    loss_function = getattr(loss, config["loss"]["type"])(**config["loss"]["args"])
    metrics = {item["type"]: getattr(metric, item["type"])(**item["args"]) for item in config['metrics']}

    solver = trainer.Trainer(
        config=config,
        resume=resume,
        model=model,
        loss_function=loss_function,
        metrics = metrics,
        optimizer=optimizer,
        scheduler = scheduler,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    solver.train()


if __name__ == '__main__':
    root = "~/Desktop/MaxMin_Pytorch"

    root = os.path.expanduser(root)
    parser = argparse.ArgumentParser(description="Wave-U-Net for Speech Enhancement")
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.json).")
    parser.add_argument("-R", "--resume", action='store_true', help="Resume experiment from latest checkpoint.")
    parser.add_argument("-D", "--debug", action='store_true', help="Only use first 100 examples")
    args = parser.parse_args()

    with open(os.path.join(root, "configs", args.configuration)) as file:
        configuration = json5.load(file)

    configuration["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    if args.debug:
        configuration["experiment_name"] += '_debug'

    configuration["root"] = root

    main(configuration, resume=args.resume, debug=args.debug)