from omegaconf import OmegaConf
from lime import lime_image
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import sys
import torch.nn.functional as F
sys.path.append(r"../")
from classification.train.train_classification import MultiPartitioningClassifier, load_yaml
os.chdir(r"../")
from transformers import CLIPModel, CLIPImageProcessor
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import json
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
import time
device = "cuda"

def get_image_embedding(image,image_processor,model):
    '''
    Get the image embedding using  StreetCLIP model
    :param image: np.array, image
    :param image_processor: CLIPImageProcessor, image processor
    '''
    input = image_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():  
        image_features = model.get_image_features(**input)
    output_features = image_features.cpu().squeeze().detach().numpy()
    del image_features, input
    torch.cuda.empty_cache()
    return output_features

def process_images(images,image_processor,model):
    '''
    Get image embeddings and ids
    :param images: list, list of images
    :param image_processor: CLIPImageProcessor, image processor
    :param model: CLIPModel, model
    '''
    image_embeddings = [get_image_embedding(Image.fromarray(image),image_processor,model) for image in images]
    ids = list(range(len(images)))
    return image_embeddings,ids

def get_preds(images):
    '''
    Get predictions for the images, this method is used for LIME explanation
    :param images: list, list of images
    '''
    ckpt = r"weights\streetclip_clues\1\ckpts\last.ckpt"
    config = load_yaml(r"config\resnet50_image_clip_clues.yml")
    config.model_params.weights = ckpt
    config.model_params.text_labels_file = ""
    image_model = CLIPModel.from_pretrained("geolocal/StreetCLIP").cuda()
    image_processor = CLIPImageProcessor.from_pretrained("geolocal/StreetCLIP")
    batch = process_images(images,image_processor,image_model)
    model = MultiPartitioningClassifier(config["model_params"], None)
    model = model.to(device)
    model = model.eval()
    with torch.no_grad():
        images, ids = batch
        images = torch.tensor(images).to(device)  # Convert images to torch tensor
        output = model((images, ids))
        probs = F.softmax(output['output'][0], dim=1).detach().cpu().numpy()
    return probs

def get_relevant_clues(image_input,clues):
    '''
    Get the most 5 relevant clues for the image
    :param image_input: np.array, image
    :param clues: list, list of clues
    '''
    ckpt = r"weights\streetclip_clues_no_pretrained\128\ckpts\last.ckpt"
    config = load_yaml(r"config\resnet50_image_clip_clues.yml")
    config.model_params.weights = ckpt
    config.model_params.text_labels_file = ""
    image_model = CLIPModel.from_pretrained("geolocal/StreetCLIP").cuda()
    image_processor = CLIPImageProcessor.from_pretrained("geolocal/StreetCLIP")
    batch = process_images([image_input],image_processor,image_model)
    model = MultiPartitioningClassifier(config["model_params"], None)
    model = model.to(device)
    model = model.eval()
    with torch.no_grad():
        image, ids = batch
        image = torch.tensor(image).to(device)  
        output = model((image, ids))     
        # Get the attention scores for the clues for given image
        scores = output['attn']['attn_scores'][0].detach().cpu().numpy()
        most_revelant = [clues[i] for i in scores.argsort()[-5:][::-1]]
    return most_revelant     

def get_top_class(image,labels):
    '''
    Get the top class for the image
    :param image: np.array, image
    :param labels: list, list of labels
    '''
    preds = get_preds([image])
    max_class = np.argmax(preds)
    return labels[max_class]

def join_explanations(image,clues,true_label,label_list):
    '''
    Join the LIME explanation with the most relevant clues
    :param image: np.array, image
    :param clues: list, list of clues
    :param true_label: str, true label
    :param label_list: list, list of labels
    '''
    explainer = lime_image.LimeImageExplainer() 
    explanation = explainer.explain_instance(image, get_preds, top_labels=2, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
    img_boundry1 = mark_boundaries(temp/255.0, mask)
    plt.imshow(img_boundry1)
    plt.show()
    most_relevant = get_relevant_clues(image,clues)
    print("True label :",true_label)
    print("Predicted label: ",get_top_class(image,label_list))
    print("Most revelant clues:")
    for c in most_relevant:
        print(c)
        print("")

def get_explanation(image_name):
    '''
    Wrapper method to get the explanation for the image
    :param image_name: str, image name
    '''
    data_path = r"dataset\test\images"
    clues = pd.read_csv(r"dataset\clues\merged.csv")['0'].tolist()
    with open(r"dataset\test\label_mapping\countries_names.json","r") as f:
        mapping = json.load(f)
    labels = pd.read_csv(r"dataset\s2_cells\countries.csv")['country'].to_list()
    image = np.array(Image.open(os.path.join(data_path,image_name)))
    true_label = mapping[image_name][0]
    print("Image ID: ",image_name)
    join_explanations(image,clues,true_label,labels)