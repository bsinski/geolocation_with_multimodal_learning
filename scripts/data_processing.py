import requests
import os
import sys
import pandas as pd
import re
from PIL import Image
from shutil import move
from argparse import ArgumentParser
os.chdir(r"../")

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--split",
        type=str,
        default = "train"
    )
    return args.parse_args()

def delete_empty_images(split):
    '''
    Delete images that are empty from the dataset and update the csv file
    :param split: str, split of the dataset
    '''
    source_directory = rf"dataset\{split}\images"
    destination_directory = rf"dataset\{split}\images_empty"
    source_csv = rf"dataset\{split}\{split}.csv"
    empty_image = Image.open(r"dataset\empty_image.png")
    images = pd.read_csv(source_csv)
    for image in os.listdir(source_directory):
        img = Image.open(os.path.join(source_directory, image))
        if img == empty_image:
            move(os.path.join(source_directory, image), destination_directory)
            images = images[images.IMG_ID != image]
    images.to_csv(source_csv, index=False)

if __name__ == "__main__":
    args = parse_args()
    delete_empty_images(args.split,args.obs)