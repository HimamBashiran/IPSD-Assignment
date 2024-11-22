#################################### TRAIN.PY ###########################################
#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from Utils.getData import Data
from Models.CNN import SimpleCNN

def main():
    # PARAMETER
    BATCH_SIZE = 8
    EPOCH = 25

    # HYPERPARAMETER
    LEARNING_RATE = 0.001

    train_loader = DataLoader(Data("/Users/User/IPSD Assignment/MODUL-9-PYTORCH/Dataset/"), batch_size=BATCH_SIZE, shuffle=True)

    model = SimpleCNN(input_dim=200, input_c=3, output=5, hidden_dim=128, dropout=0.5, device='cpu')
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    loss_all = []
    for epoch in range(EPOCH):
        loss_train = 0
        for batch, (src, trg) in enumerate(train_loader):
            src = torch.permute(src, (0, 3, 1, 2))
            pred = model(src)

            loss = loss_fn(trg, pred)
            loss_train += loss.detach().numpy()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        print("epoch = ", epoch + 1, " loss = ", loss_train / len(train_loader))
        loss_all.append(loss_train / len(train_loader))
    
    plt.plot(range(EPOCH), loss_all, color="#931a00", label='Training')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./training.png")
    
if __name__=="__main__":
    main()