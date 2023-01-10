import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
from model import MyAwesomeModel
from torch import nn, optim

from src.data.make_dataset import Mnist

#import click



#@click.group()
#def cli():
#    pass

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = MyAwesomeModel()
    model.load_state_dict(checkpoint['state_dict'])
    return model


#@click.command()
#@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr, testset, trainset, experiment):
    print(os.listdir())
    print("Training day and night")
    print(lr)
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)
    epochs = 4
    eval_delta = 50
    train_loss = 0
    train_losses = []
    test_losses = []
    step = 0
    cp_it = 0
    last_test = 0
    for e in range(epochs):
        model.train()
        for images, labels in trainset:
            step += 1
            optimizer.zero_grad()
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if step % eval_delta == 0:
                checkpoint = {'state_dict': model.state_dict()}
                cp_path = f"experiments/{experiment}/checkpoint_{cp_it}.pth"
                cp_it += 1
                torch.save(checkpoint, cp_path)
                accuracy, test_loss = evaluate(cp_path, testset, criterion)
                test_losses.append(test_loss)
                train_losses.append(train_loss/(step-last_test))
                print(f'Epoch: {step/len(trainset)}/{epochs}\t test accuracy: {accuracy * 100}%\t Test loss: {test_loss}\t Train loss: {train_loss/(step-last_test)}')
                train_loss = 0
                last_test = step
                model.train()
    return train_losses, test_losses

#@click.command()
#@click.argument("model_checkpoint")
def evaluate(model_checkpoint, testset, criterion):
    print("Evaluating until hitting the ceiling")
    #print(model_checkpoint, testset)
    model = load_checkpoint(model_checkpoint)
    accuracy = 0
    test_loss = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testset:
            log_ps = model.forward(images)
            test_loss += criterion(log_ps, labels).item()
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)

            accuracy += torch.mean(equals.type(torch.FloatTensor))
        divisor = len(testset)
    return accuracy/divisor, test_loss/divisor


#cli.add_command(train)
#cli.add_command(evaluate)


if __name__ == "__main__":
    testset = torch.utils.data.DataLoader(Mnist("raw", "processed", "test"),
                                           batch_size=64, shuffle=True)
    trainset = torch.utils.data.DataLoader(Mnist("raw", "processed", "train")
                                           , batch_size=64, shuffle=True)
    experiment = "1"
    train_l, test_l = train(1e-3, testset, trainset, experiment)
    index = range(1,len(train_l)+1)
    fig = plt.figure()
    plt.plot(index, train_l, label="Training Loss")
    plt.plot(index, test_l, label="Test Loss")
    plt.legend()
    plt.show()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fig.savefig(os.path.join(root, "reports", "figures", f"experiment_{experiment}.png"))
    #cli()