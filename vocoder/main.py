import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import random
import yaml
from vocoder import Vocoder
from dataset import MNGU0Dataset, train_collate_fn, test_and_validation_collate_fn
from train import trainer_vocoder
from losses import MultiScaleSpectralLoss
import argparse


def calc_nparam(model):
    nparam = 0
    for p in model.parameters():
        if p.requires_grad:
            nparam += p.numel()
    return nparam


if __name__ == "__main__":
    # set random seeds
    torch.manual_seed(324)
    torch.cuda.manual_seed(324)
    np.random.seed(324)
    random.seed(324)

    # load yaml config
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config yaml")
    args = parser.parse_args()

    yaml_name = args.config
    with open("yamls/" + yaml_name, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # define dataloader
    print("Loading dataset...")
    train_dataset = MNGU0Dataset("train", config)
    val_dataset = MNGU0Dataset("val", config)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=train_collate_fn
    )
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=test_and_validation_collate_fn)
    print(f"Total training data: {len(train_dataset)}")
    print(f"train loudness dir: {train_dataset.loudness_dir}, train pitch dir: {train_dataset.pitch_dir}")
    print(f"Total val data: {len(val_dataset)}")
    print(f"val loudness dir: {val_dataset.loudness_dir}, val pitch dir: {val_dataset.pitch_dir}")

    # define model
    print("Loading model...")
    model = Vocoder(
        hidden_dim=config["hidden_dim"],
        nharmonics=config["nharmonics"],
        nbands=config["nbands"],
        attenuate=config["attenuate"],
        fs=config["fs"],
        framesize=config["framesize"],
    )
    model = model.to(config["device"])
    print(f"number of parameters: {calc_nparam(model)}")

    # optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config["milestones"], gamma=config["gamma"]
    )

    # define criterion
    criterion = MultiScaleSpectralLoss().to(config["device"])

    # train the model
    trainer_vocoder(model, optimizer, lr_scheduler, criterion, train_dataloader, val_dataloader, config)
