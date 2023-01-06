import argparse
import sys

import torch
import click

from data import mnist
from model import MyAwesomeModel
from torchvision import transforms
from torch import nn, optim


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr, trainloader, testloader):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    epochs = 10
    trainlosses, testlosses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()
            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backwards()
            optimizer.step()
            running_loss += loss.item()


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint, testloader):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)

if __name__ == "__main__":
    cli()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    trainset = mnist(filepaths, transform=transform)
    testset = mnist(filepath, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)








