import torch
import torch.nn as nn
from pathlib import Path
from utils.model import build_bert
from utils.config import get_weights_path, get_config
from utils.dataloader import create_ds
from tqdm import tqdm
import wandb

def get_model(config, inp_vocab_size):
    model = build_bert(
        inp_vocab_size, 
        config['num_labels'], 
        config['seq_len'], 
        config['d_model'], 
        config['num_layers'], 
        config['num_heads'], 
        config['dropout'], 
        config['intermediate_size']
        )
    return model

def train_model(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    train_dataloader, val_dataloader, tokenizer = create_ds(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'], eps= 1e-9)
    
    init_epoch =0
    global_step = 0
    if config['preload']:
        model_filename = get_weights_path(config, config['preload'])
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        init_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
        
    loss_fn = nn.CrossEntropyLoss().to(device)
    
    wandb.define_metric("global_step")
    # define which metrics will be plotted against it
    wandb.define_metric("train/*", step_metric="global_step")
    
    for epoch in range(init_epoch, config['epochs']):
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {(epoch+1):02d}")
        for batch in batch_iterator:
            model.train()
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device) # (B, 1)
            
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, Seq_Len, d_model)
            classification_output = model.project(encoder_output) # (B, 1, label_size)

            loss = loss_fn(classification_output, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            
            # Log the loss
            wandb.log({'train/loss': loss.item(), 'global_step': global_step})
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            global_step += 1
        
        model_filename = get_weights_path(config, f'{epoch:02d}')
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'global_step' : global_step
        }, model_filename)
        
            
if __name__ == '__main__':
    config = get_config()
    wandb.login(key='key')
    wandb.init(
        project = "Bert-Base Sentiment Analysis", 
        config = get_config(),
        name= 'Version 1'
        )
    train_model(config)