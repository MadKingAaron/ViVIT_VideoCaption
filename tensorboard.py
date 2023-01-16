import torch
from torch.utils.tensorboard import SummaryWriter


class TensorBoard():
    def __init__(self) -> None:
        self.writer = SummaryWriter()

    def write_train_loss(self, train_loss, iter):
        self.writer.add_scalar('Loss/train', train_loss, iter)
    
    def close_tb(self):
        self.writer.flush()
        self.writer.close()