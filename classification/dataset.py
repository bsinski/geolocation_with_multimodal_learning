import os
import pickle
import random
import re
import sys
from io import BytesIO
from math import ceil
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast
import pandas as pd
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
import msgpack

class MsgPackIterableDatasetMultiTargetWithDynLabels(torch.utils.data.IterableDataset):
    """
    Data source: bunch of msgpack files
    Target values are generated on the fly given a mapping (id->[target1, target, ...])
    """

    def __init__(
        self,
        path: str,
        target_mapping: Dict[str, int],
        key_img_id: str = "id",
        key_img_encoded: str = "image",
        transformation=None,
        shuffle=True,
        meta_path=None,
        cache_size=6 * 4096,
        lat_key="LAT",
        lon_key="LON",
    ):

        super(MsgPackIterableDatasetMultiTargetWithDynLabels, self).__init__()
        self.path = path
        self.cache_size = cache_size
        self.transformation = transformation
        self.shuffle = shuffle
        self.seed = random.randint(1, 100)
        self.key_img_id = key_img_id.encode("utf-8")
        self.key_img_encoded = key_img_encoded.encode("utf-8")
        self.target_mapping = target_mapping

        for k, v in self.target_mapping.items():
            if not isinstance(v, list):
                self.target_mapping[k] = [v]
        if len(self.target_mapping) == 0:
            raise ValueError("No samples found.")

        if not isinstance(self.path, (list, set)):
            self.path = [self.path]

        self.meta_path = None
        meta_path = None
        if meta_path is not None:
            self.meta = pd.read_csv(meta_path, index_col=0)
            self.meta = self.meta.astype({lat_key: "float32", lon_key: "float32"})
            self.lat_key = lat_key
            self.lon_key = lon_key

        self.shards = self.__init_shards(self.path)
        self.length = len(self.target_mapping)
        
    @staticmethod
    def __init_shards(path: Union[str, Path]) -> list:
        shards = []
        for i, p in enumerate(path):
            shards_re = r"shard_(\d+).msg"
            shards_index = [
                int(re.match(shards_re, x).group(1))
                for x in os.listdir(p)
                if re.match(shards_re, x)
            ]
            shards.extend(
                [
                    {
                        "path_index": i,
                        "path": p,
                        "shard_index": s,
                        "shard_path": os.path.join(p, f"shard_{s}.msg"),
                    }
                    for s in shards_index
                ]
            )
        if len(shards) == 0:
            raise ValueError("No shards found")
        return shards

    def _process_sample(self, x):
        # prepare image and target value

        # decode and initial resize if necessary
        img = Image.open(BytesIO(x[self.key_img_encoded]))
        if img.mode != "RGB":
            img = img.convert("RGB")

        if img.width > 320 and img.height > 320:
            img = torchvision.transforms.Resize(320)(img)

        # apply all user specified image transformations
        if self.transformation is not None:
            img = self.transformation(img)

        _id = x[self.key_img_id].decode("utf-8")

        if self.meta_path is None:
            return img, x["target"], _id
        else:
            meta = self.meta.loc[_id]
            return img, x["target"], meta[self.lat_key], meta[self.lon_key], _id

    def __iter__(self):

        shard_indices = list(range(len(self.shards)))

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(shard_indices)

        worker_info = torch.utils.data.get_worker_info()

        if worker_info is not None:

            def split_list(alist, splits=1):
                length = len(alist)
                return [
                    alist[i * length // splits : (i + 1) * length // splits]
                    for i in range(splits)
                ]

            shard_indices_split = split_list(shard_indices, worker_info.num_workers)[
                worker_info.id
            ]

        else:
            shard_indices_split = shard_indices

        cache = []

        for shard_index in shard_indices_split:
            shard = self.shards[shard_index]

            with open(
                os.path.join(shard["path"], f"shard_{shard['shard_index']}.msg"), "rb"
            ) as f:
                unpacker = msgpack.Unpacker(
                    f, max_buffer_size=1024 * 1024 * 1024, raw=True
                )
                for x in unpacker:
                    if x is None:
                        continue

                    # valid dataset sample?
                    _id = x[self.key_img_id].decode("utf-8")
                    try:
                        # set target value dynamically
                        if len(self.target_mapping[_id]) == 1:
                            x["target"] = self.target_mapping[_id][0]
                        else:
                            x["target"] = self.target_mapping[_id]
                    except KeyError:
                        # reject sample
                        # print(f'reject {_id} {type(_id)}')
                        continue

                    if len(cache) < self.cache_size:
                        cache.append(x)

                    if len(cache) == self.cache_size:

                        if self.shuffle:
                            random.shuffle(cache)
                        while cache:
                            yield self._process_sample(cache.pop())
        if self.shuffle:
            random.shuffle(cache)

        while cache:
            yield self._process_sample(cache.pop())

    def __len__(self):
        return self.length
    
class EncodedImagesDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_images,mapping):
        self.encoded_images = encoded_images  
        self.mapping = mapping

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        id = list(self.mapping.keys())[idx]
        encoded_image = self.encoded_images[id]
        label = self.mapping[id]
        return encoded_image, label,id