from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class TrainConfig:
    num_epochs: int = 15
    batch_size: int = 256

    n_workers: int = 4
    early_stopping_rounds: int = 50

    optimizer_name: str = 'AdamW'
    learning_rate: int = 1e-2
    weight_decay = 0.0001
    scheduler_name: str = 'CosineScheduler'
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=0) # PAD_ID

    verbose: bool = True

    def get_optimizer(self, model):
        if self.optimizer_name == 'Adam':
            return torch.optim.Adam(model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'AdamW':
             return torch.optim.AdamW(model.parameters(), self.learning_rate, weight_decay=self.weight_decay)
    
    def get_scheduler(self, optimizer):
        if self.scheduler_name == 'CosineScheduler':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.num_epochs)
        elif self.scheduler_name == 'Cosine scheduler with warmup':
            scheduler1 = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=5
            )
            scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=295, eta_min=1e-5
            )
            return torch.optim.lr_scheduler.SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[5])
    



