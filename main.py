import comet_ml

import argparse
import os

from tqdm import tqdm

import numpy as np
import random

import torch
from torch.utils.data import DataLoader

from models import TranslationModel

from config import DataConfig
from config import ModelConfig
from config import TrainConfig

from loops import train_epoch
from loops import eval_epoch
from loops import evaluate_bleu

from utils import convert_to_text

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
parser.add_argument("--n_epochs", type=int, default=15)
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

model = TranslationModel(
    de_vocab.vocab.vocab_size(),
    en_vocab.vocab.vocab_size(),
    model_config.d_model,
    model_config.dropout_rate,
    model_config.num_encoder_layers,
    model_config.num_decoder_layers,
    data_config.max_length,
    model_config.dim_feedforward
).to(device)
loss_fn = training_config.loss_fn

optimizer = training_config.get_optimizer(model)

scheduler = training_config.get_scheduler(optimizer)

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
    experiment.log_parameter('optimizer', optimizer.__class__.__name__)
    experiment.log_parameter('scheduler', scheduler.__class__.__name__)
    experiment.log_parameter('batch_size', TrainConfig.batch_size)

# training process
num_epochs = TrainConfig.num_epochs
best_val_bleu = -1
val_bleus = []
val_bleu = -1
cur_epoch = 1
patience_counter = 0

if args.continue_training:
    checkpoint = torch.load(f"checkpoints/{experiment_name}.pth")
    best_val_acc = checkpoint['best_val_acc']
    cur_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

print(f'Starting experiment: {experiment_name}')
print(f'device: ', model_config.device)
print(f'model params: ', sum(param.numel() for param in model.parameters()))
print(f'log experiment: ', 'Yes' if args.log_experiment else 'No')

for epoch in range(cur_epoch, cur_epoch + num_epochs):
    print(f"\nEpoch {epoch}/{cur_epoch + num_epochs - 1}")
    print("-" * 60)
    lr = optimizer.param_groups[0]['lr']

    # train
    train_loss = train_epoch(
        model,
        optimizer,
        train_loader, loss_fn, device
    )

    # validation
    val_loss = eval_epoch(model, loss_fn, val_loader, device)

    scheduler.step()

    if epoch % 4 == 1:
        val_bleu = evaluate_bleu(model, val_loader, device)
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
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'optimizer_class': optimizer.__class__.__name__,
            'scheduler_state_dict': scheduler.state_dict()
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
model.load_state_dict(checkpoint['model_state_dict'])


# predict on test
print('Starting prediction on test set')
test_dataset = data_config.get_test_dataset(de_vocab, en_vocab)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=training_config.batch_size,num_workers=training_config.n_workers)
predictions = []
for de_indices, de_lengths in tqdm(test_loader, desc='Predicting on test'):
    de_indices = de_indices[:, :de_lengths.max()].to(device)
    predicted_tokens = model.inference(de_indices).cpu().tolist()
    predictions.extend([convert_to_text(seq, en_vocab) for seq in predicted_tokens])

predictions = [s + '\n' for s in predictions]

# write predictions to file
with open('test1.de-en.en', 'w', encoding='utf-8') as file:
    file.writelines(predictions)

if args.log_experiment: 
    experiment.end()

    
