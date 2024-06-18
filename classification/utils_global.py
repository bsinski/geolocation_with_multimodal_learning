import logging
from collections import OrderedDict
from pathlib import Path
from typing import Union, List

import torch
import torchvision

def vstack(tensors):
    return torch.cat([t.unsqueeze(0) for t in tensors], dim=0)

def check_is_valid_torchvision_architecture(architecture: str):
    """Raises an ValueError if architecture is not part of available torchvision models
    """
    available = sorted(
        name
        for name in torchvision.models.__dict__
        if name.islower()
        and not name.startswith("__")
        and callable(torchvision.models.__dict__[name])
    )
    if architecture not in available:
        raise ValueError(f"{architecture} not in {available}")

def load_weights_if_available(
    model: torch.nn.Module, classifier: torch.nn.Module, attn_layer: torch.nn.Module, weights_path: Union[str, Path]
):
    # print(f"Loading weights from {weights_path}")
    # Ignore classifier and only load weights of featurizer
    checkpoint = torch.load(weights_path, map_location=lambda storage, loc: storage)

    state_dict_features = OrderedDict()
    state_dict_classifier = OrderedDict()
    state_dict_attn = OrderedDict()
    for k, w in checkpoint["state_dict"].items():
        if k.startswith("model"):
            state_dict_features[k.replace("model.", "")] = w
        elif k.startswith("classifier"):
            state_dict_classifier[k.replace("classifier.", "")] = w
        elif k.startswith("attn_layer"):
            state_dict_attn[k.replace("attn_layer.", "")] = w
        else:
            logging.warning(f"Unexpected prefix in state_dict: {k}")
    if model:
        model.load_state_dict(state_dict_features, strict=False)
    if classifier:
        classifier.load_state_dict(state_dict_classifier, strict=False)
    if attn_layer:
        attn_layer.load_state_dict(state_dict_attn, strict=False)
    return model, classifier, attn_layer

def vectorized_gc_distance(latitudes, longitudes, latitudes_gt, longitudes_gt):
    R = 6371
    factor_rad = 0.01745329252
    longitudes = factor_rad * longitudes
    longitudes_gt = factor_rad * longitudes_gt
    latitudes = factor_rad * latitudes
    latitudes_gt = factor_rad * latitudes_gt
    delta_long = longitudes_gt - longitudes
    delta_lat = latitudes_gt - latitudes
    subterm0 = torch.sin(delta_lat / 2) ** 2
    subterm1 = torch.cos(latitudes) * torch.cos(latitudes_gt)
    subterm2 = torch.sin(delta_long / 2) ** 2
    subterm1 = subterm1 * subterm2
    a = subterm0 + subterm1
    c = 2 * torch.asin(torch.sqrt(a))
    gcd = R * c
    return gcd


def gcd_threshold_eval(gc_dists, thresholds=[1, 25, 200, 750, 2500]):
    # calculate accuracy for given gcd thresolds
    results = {}
    for thres in thresholds:
        results[thres] = torch.true_divide(
            torch.sum(gc_dists <= thres), len(gc_dists)
        ).item()
    return results


def accuracy(output, target, partitioning_shortnames: list, topk=(1, 5, 10)):
    def _accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = {}
            for k in topk:
                # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
                res[k] = correct_k / batch_size
            return res

    with torch.no_grad():
        out_dict = {}
        for i, pname in enumerate(partitioning_shortnames):
            res_dict = _accuracy(output[i], target[i], topk=topk)
            for k, v in res_dict.items():
                out_dict[f"acc{k}_val/{pname}"] = v

        return out_dict

def summarize_loss_acc_stats(pnames: List[str], outputs, topk=[1, 5, 10]):

    loss_acc_dict = {}
    metric_names = []
    for k in topk:
        accuracy_names = [f"acc{k}_val/{p}" for p in pnames]
        metric_names.extend(accuracy_names)
    metric_names.extend([f"loss_val/{p}" for p in pnames])
    # Add classification and attn loss
    metric_names.extend([f"loss_val/{p}/cls_loss" for p in pnames])
    metric_names.extend([f"loss_val/{p}/attn_loss" for p in pnames])
    for metric_name in ["loss_val/total", *metric_names]:
        metric_total = 0
        if metric_name not in outputs[0]:
            print(f'Metric name: {metric_name} not found in outputs')
            continue
        print(f'Metric name: {metric_name}')
        for output in outputs:
            metric_value = output[metric_name]
            metric_total += metric_value
        loss_acc_dict[metric_name] = metric_total / len(outputs)
    return loss_acc_dict


