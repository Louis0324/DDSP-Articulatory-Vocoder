import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import argparse
import yaml
from vocoder import Vocoder
from discriminator import MultiScaleDiscriminator
from dataset import MNGU0Dataset, train_collate_fn, test_and_validation_collate_fn
from train import trainer
from losses import MultiScaleSpectralLoss
from utils import calc_nparam

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
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=train_collate_fn, num_workers=2)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=test_and_validation_collate_fn, num_workers=2)
    print(f"Total training data: {len(train_dataset)}")
    print(f"Total val data: {len(val_dataset)}")

    # define model
    print("Loading model...")
    model = Vocoder(
        hidden_dim=config["hidden_dim"],
        nharmonics=config["nharmonics"],
        nbands=config["nbands"],
        attenuate=config["attenuate"],
        fs=config["fs"],
        framesize=config["framesize"],
        dilations=config['dilations'],
        nstacks=config['nstacks'],
        reverb_len=config['reverb_len']
    )
    model = model.to(config["device"])
    print(f"number of parameters: {calc_nparam(model)}")
    
    # define discriminator
    discriminator = MultiScaleDiscriminator(nscales=len(config['scales']), features=config['features'])

    # optimizer and scheduler
    optimizer_gen = torch.optim.Adam(model.parameters(), lr=config["lr_gen"], betas=(0.9, 0.999))
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=config["lr_disc"], betas=(0.9, 0.999))
    lr_scheduler_gen = torch.optim.lr_scheduler.MultiStepLR(optimizer_gen, milestones=config["milestones"], gamma=config["gamma"])
    lr_scheduler_disc = torch.optim.lr_scheduler.MultiStepLR(optimizer_disc, milestones=config["milestones"], gamma=config["gamma"])

    # define criterion
    MSSLoss = MultiScaleSpectralLoss(nffts=[2048, 1024, 512, 256, 128, 64]).to(config["device"])
    MSELoss = torch.nn.MSELoss()
    
    # train the model
    trainer(
        model, discriminator, 
        optimizer_gen, optimizer_disc, 
        lr_scheduler_gen, lr_scheduler_disc, 
        MSSLoss, MSELoss, config['alpha'], config['beta'], 
        config['scales'], config['fft_factor'], config['overlap'], 
        train_dataloader, val_dataloader, 
        config['total_epochs'], config['comment'], config['device']
    )
