from transformers import RobertaTokenizer, RobertaModel
import pandas as pd
import pickle
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--geo_sentences_csv",
        type=Path
    )
    args.add_argument(
        "--output",
        type=Path
    )
    return args.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # Load pre-trained model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = RobertaModel.from_pretrained('roberta-base')
    # Load sentences
    sentences = pd.read_csv(args.geo_sentences_csv)
    sentences_list = sentences['0'].tolist()
    features_dict = OrderedDict()
    # Encode sentences
    for i in range(len(sentences_list)):
        encoded_input = tokenizer(sentences_list[i], return_tensors='pt')
        output = model(**encoded_input)
        features_dict[i] = output.pooler_output.detach().numpy()[0]
    # Save features
    with open(args.output, 'wb') as file:
        pickle.dump(features_dict, file)