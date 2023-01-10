# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    Mnist()
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


class Mnist(Dataset):
    def __init__(self, ingest, processed, subset):
        self.ing = ingest
        self.proc = processed
        self.sub = subset
        self.root = os.path.dirname(os.path.abspath(__file__))
        try:
            self.data, self.labels = torch.load(
                os.path.join(self.root, processed, f"{subset}.pt")
            )
        except:
            self.data, self.labels = self._concat_content()
            torch.save(
                [self.data, self.labels],
                os.path.join(self.root, processed, f"{subset}.pt"),
            )

    def __len__(self):
        return self.labels.numel()

    def __getitem__(self, idx):
        img, target = self.data[idx], self.labels[idx]
        return img.float(), target

    def _load_file(self):
        content = []
        files = os.listdir(os.path.join(self.root, self.ing))
        itter = 0
        if f"{self.sub}_0.npz" in files:
            while f"{self.sub}_{itter}.npz" in files:
                content.append(
                    np.load(
                        os.path.join(self.root, self.ing, f"{self.sub}_{itter}.npz")
                    )
                )
                itter += 1
        else:
            content.append(
                np.load(os.path.join(self.root, self.ing, f"{self.sub}.npz"))
            )
        return content

    def _concat_content(self):
        content = self._load_file()
        data = torch.tensor(np.concatenate([c.f.images for c in content])).reshape(
            -1, 1, 28, 28
        )
        labels = torch.tensor(np.concatenate([c.f.labels for c in content]))
        return data, labels


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
