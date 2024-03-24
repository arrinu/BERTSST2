import torch
from tokenizers import Tokenizer
from pathlib import Path
from model import build_bert
from config import get_weights_path, get_config
from train import create_ds

def val_acc(model, validation_dl, device):
    model.eval()
    correct_predictions = 0

    with torch.no_grad():
        for batchidx, batch in enumerate(validation_dl):
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device)  # (B, 1)

            encoder_output = model.encode(encoder_input, encoder_mask)  # (B, Seq_Len, d_model)
            classification_output = model.project(encoder_output)
            model_output = torch.argmax(classification_output, dim=-1)

            correct_predictions += torch.sum(model_output.int() == label.int()).item()
            
            if(batchidx%80==0):
                print()
                print(model_output, label)

    accuracy = correct_predictions / len(validation_dl)
    
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

config = get_config()

tokenizer_path = Path(config['tokenizer_file'])
tokenizer = Tokenizer.from_file(str(tokenizer_path))

model = build_bert(
    tokenizer.get_vocab_size(), 
    config['num_labels'], 
    config['seq_len'], 
    config['d_model'], 
    config['num_layers'], 
    config['num_heads'], 
    config['dropout'], 
    config['intermediate_size']
)

epoch = '09'
model_filename = get_weights_path(config, epoch)
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

train_dataloader, val_dataloader, tokenizer = create_ds(config)
    
val_acc(model, val_dataloader, device)