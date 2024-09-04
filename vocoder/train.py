import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
from torchaudio.transforms import Spectrogram
from torch.utils.tensorboard.writer import SummaryWriter

def val(model, writer, val_dataloader, step, criterion, previous_loss, comment, device):
    model.eval()
    with torch.no_grad():
        print(f'Starting Validation...')
        running_loss = 0.
        for i, batch in enumerate(val_dataloader):
            
            # loading the data
            x, ema, f0, loudness = batch
            x = x.to(device).float()
            ema = ema.to(device).float()
            f0 = f0.to(device).float()
            loudness = loudness.to(device).float()
            
            # run through the model
            x_hat = model(f0, loudness, ema)
            
            # save val example
            if i == 0 and step % 20 == 0:
                writer.add_audio('ground truth', x, step, sample_rate=16000)
                writer.add_audio('pred', x_hat, step, sample_rate=16000)
                
            # calculate loss
            loss = criterion(x_hat, x)
            
            # stats and writer
            running_loss += loss.item()
        avg_loss = running_loss / (i+1)
        writer.add_scalar('MSSLoss_val', avg_loss, step)
        print(f'MSSLoss_val: {avg_loss:.{6}f}')   
             
        # update checkpoint and previous loss
        if previous_loss > avg_loss:
            torch.save(model.state_dict(), os.path.join(YOUR_BEST_MODEL_SAVE_PATH_HERE))
            print(f'saving better model...')
        return min(avg_loss, previous_loss)

def trainer(model, discriminator, optimizer_gen, optimizer_disc, lr_scheduler_gen, lr_scheduler_disc, MSSLoss, MSELoss, alpha, beta, scales, fft_factor, overlap, train_dataloader, val_dataloader, total_epochs, comment, device):
    print(f'Starting training on {device}...')
    writer = SummaryWriter('logs/'+comment)
    val_step = 1
    previous_loss = 1000
    interval = 10 # print training loss every `interval` batches
    
    # start training
    for epoch in range(total_epochs):
        model.train()
        running_loss_recon = 0.
        running_loss_disc = 0.
        running_loss_gen = 0.
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
            
            # generate spectrogram for x and x_hat
            spec_reals = []
            spec_fakes = []
            for scale in scales:
                win_length = scale
                nfft = fft_factor * win_length
                hopsize = int((1-overlap) * win_length)
                spec = Spectrogram(n_fft=nfft, win_length=win_length, hop_length=hopsize, power=1).to(device)
                spec_reals.append(spec(x).unsqueeze(1))
                spec_fakes.append(spec(x_hat).unsqueeze(1))
            
            # train discriminator
            disc_reals = discriminator(spec_reals)
            loss_disc_real = 0.
            for disc_real in disc_reals:
                loss_disc_real += MSELoss(disc_real, torch.ones_like(disc_real, device=device)) / len(disc_reals)
                 
            disc_fakes = discriminator(spec_fakes, detach=True)
            loss_disc_fake = 0.
            for disc_fake in disc_fakes:
                loss_disc_fake += MSELoss(disc_fake, torch.zeros_like(disc_fake, device=device)) / len(disc_fakes)
            
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            # backprop and update
            optimizer_disc.zero_grad()
            loss_disc.backward()
            optimizer_disc.step() 
            
            # train generator
            loss_recon = MSSLoss(x_hat, x)
            disc_fakes = discriminator(spec_fakes)
            loss_gen = 0.
            for disc_fake in disc_fakes:
                loss_gen += MSELoss(disc_fake, torch.ones_like(disc_fake, device=device)) / len(disc_fakes)        

            loss = alpha * loss_recon + beta * loss_gen 
            
            # backprop and update
            optimizer_gen.zero_grad()
            loss.backward()
            optimizer_gen.step()
            
            # stats and writer
            running_loss_recon += loss_recon.item()
            running_loss_disc += loss_disc.item()
            running_loss_gen += loss_gen.item()
            
            if (i+1) % interval == 0:
                end = time.time()
                elapsed_time = end - start
                writer.add_scalar('MSSLoss_training', running_loss_recon / interval, epoch*len(train_dataloader)+i)
                writer.add_scalar('Disc_loss_training', running_loss_disc / interval, epoch*len(train_dataloader)+i)
                writer.add_scalar('Gen_loss_training', running_loss_gen / interval, epoch*len(train_dataloader)+i)
                
                print(f'#################RUNTIME: {elapsed_time:.{4}f}#################')
                print(f'Epoch [{epoch+1}/{total_epochs}], Batch [{i+1}/{len(train_dataloader)}], MSSLoss_training: {(running_loss_recon / interval):.{6}f}, Gen_loss_training: {(running_loss_gen / interval):.{6}f}, Disc_loss_training: {(running_loss_disc / interval):.{6}f}')
                running_loss_recon = 0.
                running_loss_disc = 0.
                running_loss_gen = 0.

        # one epoch ends, start val
        previous_loss = val(model, writer, val_dataloader, val_step, MSSLoss, previous_loss, comment, device)
        val_step += 1
        model.train()
        
        # scheduler step and save model
        lr_scheduler_gen.step()
        lr_scheduler_disc.step()
        print(f'current lr: {lr_scheduler_gen.get_last_lr()[0]}')
        torch.save(model.state_dict(), os.path.join(YOUR_RUNNING_MODEL_SAVE_PATH_HERE))








