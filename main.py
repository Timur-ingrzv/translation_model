import comet_ml

import argparse
import os

from tqdm import tqdm

import numpy as np
import random

import torch
from torch.utils.data import DataLoader

from models import EncoderRNN, DecoderRNN

from config import DataConfig
from config import ModelConfig
from config import TrainConfig

from loops import train_epoch
from loops import eval_epoch
from loops import evaluate_bleu

# for experiments representation
def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
set_random_seed(0xDEADF00D)

# parser
parser = argparse.ArgumentParser()
parser.add_argument("-exp", "--experiment_name", type=str, default="test")
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--continue_training", action="store_true")
parser.add_argument("--log_experiment", action='store_true')
parser.add_argument("--no_display", action='store_false')
parser.add_argument("--experiment_key", default='no')
parser.add_argument("--api_key", default='')

# parse args
args = parser.parse_args()
experiment_name = args.experiment_name
TrainConfig.num_epochs = args.n_epochs
TrainConfig.verbose = args.no_display

# configs
data_config = DataConfig()
model_config = ModelConfig()
training_config = TrainConfig()
os.makedirs('checkpoints', exist_ok=True)

# init objects
device = model_config.device

train_dataset = data_config.get_train_dataset()
de_vocab, en_vocab = train_dataset.de_vocab, train_dataset.en_vocab
val_dataset = data_config.get_val_dataset(de_vocab, en_vocab)
train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=training_config.n_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size, shuffle=False, num_workers=training_config.n_workers, pin_memory=True)

encoder = EncoderRNN(
    data_config.vocab_size,
    model_config.embed_dim,
    model_config.hidden_dim,
    model_config.dropout_rate
).to(device)

decoder = DecoderRNN(
    data_config.vocab_size,
    model_config.embed_dim,
    model_config.hidden_dim,
    en_vocab,
    data_config.max_length
).to(device)

loss_fn = training_config.loss_fn

optimizer_encoder = training_config.get_optimizer(encoder)
optimizer_decoder = training_config.get_optimizer(decoder)

scheduler_encoder = training_config.get_scheduler(optimizer_encoder)
scheduler_decoder = training_config.get_scheduler(optimizer_decoder)

# init comet
if args.log_experiment:
    comet_ml.login(api_key=args.api_key)
    experiment_config = comet_ml.ExperimentConfig(
        name=experiment_name
    ) 
    if args.continue_training:
        experiment = comet_ml.start(
            project_name="bhw-2",
            experiment_config=experiment_config,
            experiment_key=args.experiment_key
        )
    else:
        experiment = comet_ml.start(
            project_name="bhw-2",
            experiment_config=experiment_config   
        )
    experiment.log_parameter('optimizer_encoder', optimizer_encoder.__class__.__name__)
    experiment.log_parameter('optimizer_decoder', optimizer_decoder.__class__.__name__)
    experiment.log_parameter('scheduler', scheduler_encoder.__class__.__name__)
    experiment.log_parameter('scheduler', scheduler_decoder.__class__.__name__)
    experiment.log_parameter('batch_size', TrainConfig.batch_size)

# training process
num_epochs = TrainConfig.num_epochs
best_val_bleu = 0.0
val_bleus = []
val_bleu = -1
cur_epoch = 1
patience_counter = 0

if args.continue_training:
    checkpoint = torch.load(f"checkpoints/{experiment_name}.pth")
    best_val_acc = checkpoint['best_val_acc']
    cur_epoch = checkpoint['epoch'] + 1
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer_encoder.load_state_dict(checkpoint['optimizer_encoder_state_dict'])
    optimizer_decoder.load_state_dict(checkpoint['optimizer_decoder_state_dict'])
    scheduler_encoder.load_state_dict(checkpoint['scheduler_encoder_state_dict'])
    scheduler_decoder.load_state_dict(checkpoint['scheduler_decoder_state_dict'])

print(f'Starting experiment {experiment_name}')
print(f'device: ', model_config.device)
print(f'encoder_params: ', sum(param.numel() for param in encoder.parameters()))
print(f'decoder_params: ', sum(param.numel() for param in decoder.parameters()))
print(f'log experiment: ', 'Yes' if args.log_experiment else 'No')

for epoch in range(cur_epoch, cur_epoch + num_epochs):
    print(f"\nEpoch {epoch}/{cur_epoch + num_epochs - 1}")
    print("-" * 60)
    lr = optimizer_encoder.param_groups[0]['lr']

    # train
    train_loss = train_epoch(
        encoder, decoder,
        optimizer_encoder, optimizer_decoder,
        train_loader, loss_fn, device
    )

    # validation
    val_loss = eval_epoch(encoder, decoder, loss_fn, val_loader, device)

    scheduler_encoder.step()
    scheduler_decoder.step()

    if epoch % 1 == 0:
        val_bleu = evaluate_bleu(encoder, decoder, val_loader, device)
        if args.log_experiment:
            experiment.log_metrics({'val_bleu': val_bleu}, epoch=epoch)
    
    # log experiment
    if args.log_experiment:
        experiment.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'learning_rate': lr
            }, 
            epoch=epoch
        )
    # save best model
    if val_bleu > best_val_bleu:
        best_val_bleu = val_bleu
        checkpoint = {
            'epoch': epoch,
            'best_val_acc': best_val_bleu,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_encoder_state_dict': optimizer_encoder.state_dict(),
            'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
            'optimizer_class': optimizer_encoder.__class__.__name__,
            'scheduler_state_dict': scheduler_encoder.state_dict(),
            'scheduler_class': scheduler_encoder.__class__.__name__
        }
        if training_config.verbose:
            print(f"--> New best model saved (Val BLEU: {best_val_bleu:.2f}%)")

        torch.save(checkpoint, f'checkpoints/{experiment_name}.pth')
        patience_counter = 0
    else:
        patience_counter += 1

    # Early stopping
    if patience_counter >= TrainConfig.early_stopping_rounds:
        print(f'\nEarly stopping done after {epoch} epochs')

# load best model
checkpoint = torch.load(f"checkpoints/{experiment_name}.pth")
decoder.load_state_dict(checkpoint['decoder_state_dict'])


# predict on test
print('Starting prediction on test set')
test_dataset = data_config.get_test_dataset(de_vocab, en_vocab)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=training_config.batch_size,num_workers=training_config.n_workers)
predictions = []
for de_indices, en_indices, de_length, en_length in tqdm(val_loader, desc='Prediction on test'):
    de_indices = de_indices[:, :de_length.max()].to(device)

    encoder_outputs, encoder_hidden = encoder(de_indices, de_length)
    tokens_ids = decoder.inference(encoder_hidden).cpu().tolist()
    predictions.extend(decoder.en_vocab.decode(tokens_ids))

predictions = [s + '\n' for s in predictions]

# Записываем в файл
with open('output.txt', 'w', encoding='utf-8') as file:
    file.writelines(predictions)

if args.log_experiment: 
    experiment.end()

    
