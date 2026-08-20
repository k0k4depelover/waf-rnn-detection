import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split

# Load Tokens pre-generated

df = torch.load("csic_rnn_inputs.pt")

X = df['X']
Y = df['y']
vocab_size = df['vocab_size']

#  Pairs X and Y by index

dataset = TensorDataset(X,Y)

# Distribute dataset into train and testing sizes.

train_size = int(0.8 * len(dataset))

val_size = len(dataset) - train_size


# Split data across testing and training

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# First we create a class WAF_RRN that hereda inherits nn.Module
class WAF_RNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)