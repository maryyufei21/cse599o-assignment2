# training script to train a transformer language model on user-provided input
# requirements: Ability to configure and control the various model and optimizer hyperparameters
# Memory-eﬀicient loading of large training and validation datasets with np.memmap
# Serializing checkpoints to a user-provided path
# Periodically logging training and validation performance (e.g., to console and/or an external
# service like Weights & Biases)
# use json file to configure hyperparameters
import json

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

import torch
import numpy as np
import wandb
from datetime import datetime
import os
from einops import rearrange

from cse599o_basics.transformer_lm import TransformerLM
from cse599o_basics.optimizer import AdamW
from cse599o_basics.utils import cross_entropy_loss, learning_rate_schedule_cosine
from cse599o_basics.utils import data_loading, save_checkpoint, load_checkpoint

import torch._dynamo
torch._dynamo.config.suppress_errors = True


def main():
    # load config
    config = load_config('./cse599o_basics/configs/basicbf16_bs256.json')

    # print config
    print("Configuration:")
    print(json.dumps(config, indent=4))

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # load dataset
    print("Loading datasets...")
    train_data = config['dataset']['train_path']
    val_data = config['dataset']['val_path']
    train_dataset = np.memmap(train_data, dtype=np.uint16, mode='r')
    val_dataset = np.memmap(val_data, dtype=np.uint16, mode='r')
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Validation dataset length: {len(val_dataset)}")

    # model and optimizer setup
    print("Setting up model and optimizer...")
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        num_layers=config['model']['num_layers'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        context_length=config['model']['max_seq_len'],
        use_rope=config['model']['use_rope'],
        theta=config['model'].get('rope_theta', None),
        token_positions=None,
        device=device,
        dtype = getattr(torch, config['model']['dtype'].split('.')[-1])
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=config['optimizer']['max_lr'],
        betas=(config['optimizer']['beta1'], config['optimizer']['beta2']),
        eps=config['optimizer']['eps'],
        weight_decay=config['optimizer']['weight_decay']
    )
    # lr_max = config['optimizer']['max_lr']
    # lr_min = config['optimizer']['min_lr']
    # T_warmup = config['optimizer']['T_warmup']
    # T_cosine = config['optimizer']['T_cosine']

    # checkpoint loading if specified
    print("Checking for existing checkpoint...")
    start_iteration = 0
    if config['training'].get('load_checkpoint_path') and os.path.exists(config['training']['load_checkpoint_path']):
        start_iteration = load_checkpoint(
            config['training']['load_checkpoint_path'], model, optimizer)
        print(f"Resumed from checkpoint at iteration {start_iteration}")
    else:
        print("No checkpoint found, starting fresh training.")

    # setup save checkpoint directory
    BASE_CKPT_DIR = config['training'].get('save_checkpoint_path', './checkpoints/')
    run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN_CKPT_DIR = os.path.join(BASE_CKPT_DIR, run_ts)
    os.makedirs(RUN_CKPT_DIR, exist_ok=True)


    # initialize wandb
    print("Setting up Weights & Biases logging...")
    if config['wandb'].get('use_wandb', False):
        wandb.init(project=config['wandb']['project'],
                   name=config['wandb']['run_name'],
                   config=config)
        wandb.watch(model, log="all")
        print("Initialized Weights & Biases logging.")
    else:
        print("Weights & Biases logging not enabled.")

    # training loop
    max_iterations = config['training']['max_iterations']
    batch_size = config['training']['batch_size']
    context_length = config['model']['max_seq_len']
    max_grad_norm = config['training']['max_grad_norm']
    eval_interval = config['training']['eval_interval']
    log_interval = config['training']['log_interval']
    checkpoint_interval = config['training']['checkpoint_interval']
    lr_max = config['optimizer']['max_lr']
    lr_min = config['optimizer']['min_lr']
    T_warmup = config['optimizer']['T_warmup']
    T_cosine = config['optimizer']['T_cosine']
    best_val_loss = float('inf')

    model.train()

    print("Starting training loop...")
    for iteration in range(start_iteration, max_iterations):
        print(f"Starting iteration {iteration + 1}/{max_iterations}")
        if config['optimizer'].get('enable_cosine_lr', False):
            lr = learning_rate_schedule_cosine(
                iteration, lr_max, lr_min, T_warmup, T_cosine)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # load batch
        inputs, targets = data_loading(
            train_dataset, batch_size, context_length, device)
        # forward pass
        logits = model(inputs)
        # batch batch_size and context_length dimensions together for loss computation
        # print("[before rearrange] logits.shape:", logits.shape)
        # print("[before rearrange] targets.shape:", targets.shape)
        # logits = rearrange(logits, 'b s c -> (b s) c')
        # targets = rearrange(targets, 'b s -> (b s)')
        # print("[after rearrange] logits.shape:", logits.shape)
        # print("[after rearrange] targets.shape:", targets.shape)
        loss = cross_entropy_loss(logits, targets)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        # logging
        if (iteration + 1) % log_interval == 0:
            print(f"Iteration {iteration + 1}, Training Loss: {loss.item():.4f}")
            if config['wandb'].get('use_wandb', False):
                wandb.log({'train/loss': loss.item(), 'train/lr': optimizer.state[optimizer.param_groups[0]['params'][0]]['lr'], 'iteration': iteration + 1})
        # evaluation
        if (iteration + 1) % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                val_inputs, val_targets = data_loading(
                    val_dataset, batch_size, context_length, device)
                val_logits = model(val_inputs)
                val_loss = cross_entropy_loss(val_logits, val_targets)

                # input_ids = [7454, 2402, 257, 640, 11, 612, 373, 257, 2495, 2576, 3706, 20037, 13]
                # true_value = [2402, 257, 640, 11, 612, 373, 257, 2495, 2576, 3706, 20037, 13, 1375]
                # logits = model(torch.tensor([input_ids], dtype=torch.int32, device=device))
                # print("logits.shape:", logits.shape)
                # print("loss", cross_entropy_loss(logits, torch.Tensor(true_value).to(torch.int32).cuda()))

                # for i in range(30):
                #     new_token = torch.argmax(logits[0, -1, :]).item()
                #     input_ids.append(new_token)
                #     logits = model(torch.tensor([input_ids], dtype=torch.int32, device=device))
                # print("Generated token IDs:", input_ids)
                # from cse599o_basics.tokenizer import BPETokenizer
                # tokenizer = BPETokenizer(vocab={}, merges=[])
                # print("Generated text:", tokenizer.decode(input_ids))

                print(f"Iteration {iteration + 1}, Validation Loss: {val_loss.item():.4f}")
                if config['wandb'].get('use_wandb', False):
                    wandb.log({'val/loss': val_loss.item(), 'iteration': iteration + 1})
                # save best model
                # if val_loss.item() < best_val_loss:
                #     best_val_loss = val_loss.item()
                #     if checkpoint_path:
                #         if not os.path.exists(os.path.dirname(checkpoint_path)):
                #             os.makedirs(os.path.dirname(checkpoint_path))
                #         save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
                #         print(f"New best model saved at iteration {iteration + 1} with Validation Loss: {best_val_loss:.4f}")
            model.train()
        # periodic checkpointing
        if RUN_CKPT_DIR and (iteration + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(RUN_CKPT_DIR, f"checkpoint_iter_{iteration + 1}.pt")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            save_checkpoint(model, optimizer, iteration + 1, checkpoint_path)
            print(f"Checkpoint saved at iteration {iteration + 1}")
    print("Training complete.")

if __name__ == "__main__":
    main()