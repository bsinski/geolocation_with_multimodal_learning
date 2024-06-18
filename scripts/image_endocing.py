import pickle
from transformers import CLIPProcessor, CLIPModel, CLIPImageProcessor, CLIPVisionModel
import torch
from PIL import Image
import os
import clip
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser
from pathlib import Path
import time

def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--data_path",
        type=Path,
        default=Path("/")
    )
    args.add_argument(
        "--output_path",
        type=Path,
        default=Path("/")
    )
    return args.parse_args()


def get_image_embedding(image_path):
    '''
    Get the image embeddins using  StreetCLIP model
    :param image_path: str, path to the image
    '''
    image = Image.open(image_path)
    input = image_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():  
        image_features = model.get_image_features(**input)
    output_features = image_features.cpu().squeeze().detach().numpy()
    del image_features, input
    torch.cuda.empty_cache()
    return output_features


if __name__ == "__main__":
    args = parse_args()
    # Load pre-trained model and processor
    model = CLIPModel.from_pretrained("geolocal/StreetCLIP").cuda()
    image_processor = CLIPImageProcessor.from_pretrained("geolocal/StreetCLIP")
    image_embeddings = OrderedDict()
    splits = ["train","val", "test"]
    for split in splits:
        i = 0
        split_path = os.path.join(args.data_path, rf"{split}\images")
        for file in os.listdir(split_path):
            file_path = os.path.join(split_path, file)
            image_embeddings[file] = get_image_embedding(file_path)
            i += 1
            if i % 100 == 0:
                print(f"Processed {i}/{len(os.listdir(split_path))} images from {split} split.")
    with open(args.output, "wb") as file:
        pickle.dump(image_embeddings, file)