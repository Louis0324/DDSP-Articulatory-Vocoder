import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from torch.utils.tensorboard.writer import SummaryWriter


def val_vocoder(model, writer, val_dataloader, step, criterion, previous_loss, config):
    device = config["device"]
    model.eval()
    with torch.no_grad():
        print(f"Starting Validation...")
        running_loss = 0.0
        for i, batch in enumerate(val_dataloader):
            # loading the data
            x, ema, f0, loudness = batch
            x = x.to(device).float()
            ema = ema.to(device).float()
            f0 = f0.to(device).float()
            loudness = loudness.to(device).float()
            # run through the model
            x_hat = model(f0, loudness, ema)
            # calculate loss
            loss = criterion(x_hat, x)
            # stats and writer
            running_loss += loss
        avg_loss = running_loss / (i + 1)
        writer.add_scalar("MSSLoss_val", avg_loss, step)
        print(f"MSSLoss_val: {avg_loss:.{6}f}")
        # update checkpoint and previous loss
        if previous_loss > avg_loss:
            torch.save(
                model.state_dict(),
                os.path.join(config["saved_models_dir"], "Vocoder_" + config["comment"] + "_best.pth"),
            )
            print(f"saving better model...")
        return min(avg_loss, previous_loss)


def trainer_vocoder(model, optimizer, lr_scheduler, criterion, train_dataloader, val_dataloader, config):
    device = config["device"]
    print(f"Starting training on {device}...")
    writer = SummaryWriter("logs/" + config["comment"])
    val_step = 1
    previous_loss = 1000
    for epoch in range(config["total_epochs"]):
        model.train()
        interval = 10  # print training loss every `interval` batches
        running_loss = 0.0
        for i, batch in enumerate(train_dataloader):
            if i % interval == 0:
                start = time.time()
            # loading the data
            x, ema, f0, loudness = batch
            x = x.to(device).float()
            ema = ema.to(device).float()
            f0 = f0.to(device).float()
            loudness = loudness.to(device).float()
            # run through the model
            x_hat = model(f0, loudness, ema)
            # calculate loss
            loss = criterion(x_hat, x)
            # backprop and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # stats and writer
            running_loss += loss
            if (i + 1) % interval == 0:
                end = time.time()
                elapsed_time = end - start
                writer.add_scalar("MSSLoss_training", running_loss / interval, epoch * len(train_dataloader) + i)

                print(f"#################RUNTIME: {elapsed_time:.{4}f}#################")
                print(
                    f"Epoch [{epoch+1}/{config['total_epochs']}], Batch [{i+1}/{len(train_dataloader)}], MSSLoss_training: {(running_loss / interval):.{6}f}"
                )
                running_loss = 0.0
        # one epoch ends, start val
        previous_loss = val_vocoder(model, writer, val_dataloader, val_step, criterion, previous_loss, config)
        val_step += 1
        model.train()
        # scheduler step and save model
        lr_scheduler.step()
        print(f"current lr: {lr_scheduler.get_last_lr()[0]}")
        torch.save(
            model.state_dict(),
            os.path.join(config["saved_models_dir"], "Vocoder_" + config["comment"] + "_running.pth"),
        )
