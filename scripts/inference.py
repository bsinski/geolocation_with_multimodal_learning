import glob
import json
from omegaconf import OmegaConf
import os
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix
import torch
from tqdm import tqdm
from pathlib import Path
import warnings
import sys
warnings.filterwarnings("ignore")
sys.path.append("../../geolocation_via_guidebook_grounding/g3")
from classification.train.train_classification import MultiPartitioningClassifier as MPC,load_yaml

os.chdir(Path("../"))
print("Working directory changed to:", os.getcwd())
device = "cuda"
def run_val(model):
    '''
    Run validation on the model and return predictions, labels and attentions
    :param model: MultiPartitioningClassifier, model to be validated
    '''
    i = 0
    predictions = []
    attentions = []
    labels = []
    batch_ids = []
    outputs = []
    dataloader = model.val_dataloader()
    for j, batch in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            images, target, ids = batch
            output = model((images, ids))
            predictions.append(output["output"][i])
            if "attn" in output:
                attentions.append(output["attn"]["attn_scores"])
            if type(target) is list:
                labels.append(target[i])
            else:
                labels.append(target)
            batch_ids.extend(ids)
    predictions = torch.cat(predictions)
    labels = torch.cat(labels)
    if attentions:
        attentions = torch.cat(attentions)
    return predictions, labels, attentions, batch_ids

def get_class_accuracies(y_true, y_pred, labels):
    '''
    Get the class accuracies
    :param y_true: np.array, true labels
    :param y_pred: np.array, predicted labels
    :param labels: list, list of labels
    '''
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return np.diagonal(cm)

def validate(predictions, labels):
    '''
    Validate the model
    :param predictions: torch.tensor, model predictions
    :param labels: torch.tensor, true labels
    '''
    acc_dict = {}
    for k in [1, 5, 10]:
        chosen_predictions = predictions.topk(k=k, dim=-1).indices
        correct = torch.any(chosen_predictions == labels.unsqueeze(dim=-1), dim=-1).sum()
        correct = correct.item() / len(labels)
        print(f"top-{k} acc:", correct)
        acc_dict[f"top-{k}"] = correct
    labels = labels.detach().cpu().numpy()
    final_predictions = predictions.argmax(dim=-1).detach().cpu().numpy()
    class_accs = get_class_accuracies(labels, final_predictions, range(249))
    print("avg class acc:", np.nanmean(class_accs))
    return final_predictions,acc_dict
        
def save(name, config, predictions, labels, attentions, batch_ids):
    '''
    Save the predictions
    :param name: str, name of the file
    :param config: dict, configuration
    :param predictions: torch.tensor, model predictions
    :param labels: torch.tensor, true labels
    :param attentions: torch.tensor, attentions
    :param batch_ids: list, batch ids
    '''
    anns = []
    for i in range(labels.shape[0]):
        ann = {}
        ann["label"] = labels[i].item()
        ann["predictions"] = predictions[i].cpu().numpy()
        if attentions != []:
            ann["attn"] = attentions[i].cpu().numpy()
        ann["id"] = batch_ids[i]
        anns.append(ann)

    folder = os.path.dirname(config.model_params.weights).replace("\ckpts", "")
    print("Saving predictions to:", folder)
    pickle.dump(anns, open(f"{folder}/{name}", "wb"))

def evalute(ckpt,eval_test = True,meta_path = "test_final.csv", label_mapping_path = "label_mapping/countries.json", save_predictions = False,return_predictions = False):
    '''
    Evaluate the model, calcualter the acuracy metrics and save the predictions
    :param ckpt: str, path to the model checkpoint
    :param eval_test: bool, whether to evaluate on test set or validation set
    :param meta_path: str, path to the meta file with image ids for test set
    :param label_mapping_path: str, path to the label mapping file for test set
    :param save_predictions: bool, whether to save the predictions to a file
    :param return_predictions: bool, whether to return the predictions
    '''
    config_name = "resnet50_clip.yml"
    config = load_yaml(f"./config/{config_name}")
    config.model_params.weights = ckpt

    if eval_test:
        config.model_params.val_meta_path = f"./dataset/test/{meta_path}"
        config.model_params.val_label_mapping = f"./dataset/test/{label_mapping_path}"
        config.model_params.msgpack_val_dir = "./dataset/test/msgpack"
        config.model_params.text_labels_file = ""
        name = "predictions_test.json"
    else:
        name = "predictions_val.json"
    model = MPC(config["model_params"], None)
    model = model.to(device)
    model = model.eval()
    predictions, labels, attentions, batch_ids = run_val(model)
    labels = labels.to(device)

    final_predictions,acc_dict = validate(predictions, labels)
    if save_predictions:
            save(name, config, predictions, labels, attentions, batch_ids)
    if return_predictions:
        return final_predictions,acc_dict
    
