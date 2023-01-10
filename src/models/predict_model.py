import argparse
import sys

import click
import numpy as np
import torch
from model import MyAwesomeModel
from torch import nn, optim
from torchvision import transforms

from src.data.make_dataset import Mnist


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint, filepath):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    images = torch.tensor(np.load(filepath), dtype=torch.float)

    output = model(images)
    prediction = output.argmax(dim=-1)
    for i in range(images.shape[0]):
        print(
            f"Image {i+1} predicted to be class {prediction[i].item()} with probability {output[i, prediction[i]].item()}"
        )


cli.add_command(evaluate)

if __name__ == "__main__":
    cli()
