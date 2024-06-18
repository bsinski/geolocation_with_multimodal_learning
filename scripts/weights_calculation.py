import numpy as np
import pandas as pd
import tensorflow as tf
import json
from argparse import ArgumentParser
import pandas as pd
from pathlib import Path

def get_class_weights(s2_countries_path,images_countries_path,output_path):
    '''
    Compute the class weights for the training data that are inversely proportional to the class frequencies
    :param s2_countries_path: str, path to the csv file containing all country labels
    :param images_countries_path: str, path the file contating mapping of avialable images to countries
    :param output_path: str, path to the output file
    '''
    countries_df = pd.read_csv(s2_countries_path)
    labels = countries_df['class_label'].values 
    with open(images_countries_path,'rb') as f:
            data = json.load(f)
    train_data = np.array([obs[0] for obs in data.values()])
    train_data_subset = train_data[np.isin(train_data, labels)]
    bincount = np.array([np.sum(train_data_subset == value) for value in labels])
    recip_freq = len(train_data) / (len(labels) * bincount)
    recip_freq[recip_freq == np.inf] = 0
    weights_train = recip_freq[labels]
    with open(output_path,'w') as f:
        json.dump([weights_train.tolist()], f)


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--output",
        type=Path,
    )
    args.add_argument(
        "--s2_cells_csv",
        type=Path,
    )
    args.add_argument(
        "--images_countries",
        type=Path,
    )
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    get_class_weights(args.s2_cells_csv, args.images_countries, args.output)