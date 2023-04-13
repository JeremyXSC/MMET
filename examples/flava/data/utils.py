# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List
import os

import requests
from datasets import concatenate_datasets, load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from flava.definitions import HFDatasetInfo
from PIL import Image, UnidentifiedImageError


DATASETS_USER_AGENT = get_datasets_user_agent()


def build_datasets_from_info(dataset_infos: List[HFDatasetInfo], split: str = "train"):
    dataset_list = []    
    for dataset_info in dataset_infos:
        print(dataset_info.key)
        print(dataset_info.subset)
        print(dataset_info.split_key_mapping[split])
        print(dataset_info.extra_kwargs)

        #import pdb; pdb.set_trace()
        current_dataset = load_dataset(
            dataset_info.key,
            dataset_info.subset,
            split=dataset_info.split_key_mapping[split],
            use_auth_token=True,
            **dataset_info.extra_kwargs,
        )
        # import pdb; pdb.set_trace()
        # current_dataset = current_dataset.select([i for i in range(8)])
        #import pdb; pdb.set_trace()
        if dataset_info.remove_columns is not None:
            current_dataset = current_dataset.remove_columns(
                dataset_info.remove_columns
            )
        if dataset_info.rename_columns is not None:
            for rename in dataset_info.rename_columns:
                current_dataset = current_dataset.rename_column(rename[0], rename[1])

        dataset_list.append(current_dataset)

    return concatenate_datasets(dataset_list)


def fetch_single_image_from_url(image_url, timeout, retries=0, sleep_timer=0):
    for _ in range(retries + 1):
        try:
            image = Image.open(
                requests.get(
                    image_url,
                    stream=True,
                    headers={"user-agent": DATASETS_USER_AGENT},
                    timeout=timeout,
                ).raw
            )
            break
        except (requests.exceptions.ConnectionError, UnidentifiedImageError):
            image = None
            time.sleep(sleep_timer)

    return image

def fetch_single_image_None(image_url, timeout, retries=0, sleep_timer=0):
    for _ in range(retries + 1):
        
        image = None
        time.sleep(sleep_timer)

    return image
    
def fetch_single_image(image_url, timeout, retries=0, sleep_timer=0):
    #import pdb;pdb.set_trace()
    for _ in range(retries + 1):
        try:
            #import pdb; pdb.set_trace()
            #print(image_url)
            image_name = image_url.split("/")[-1]
            #print(image_name)
            #print(type(image_name))
            #tmp = Image.open(os.path.join('/cluster/home/guanmengyuan/multimodal-main/examples/red_caps_download/jellyfish',image_name)).convert('RGB')
            #print(tmp)

            directory = '/cluster/home/guanmengyuan/multimodal-main/examples/red_caps_download/jellyfish'
            image_path = os.path.join(directory, image_name)
            #print(image_path)

            image = Image.open(image_path).convert('RGB')
            #print(image)
            break
        except (OSError):
            #print('OSError')
            image = None
            time.sleep(sleep_timer)
        '''
        image_name = image_url.split("/")[-1]
        directory = '/cluster/home/guanmengyuan/multimodal-main/examples/red_caps_download/jellyfish'
        image_path = os.path.join(directory, image_name)
        if os.path.exists (image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = None
        '''

    #print("image------",image)
        
    return image

def fetch_images(batch, num_threads, timeout=None, retries=0, sleep_timer=0):
# def fetch_images(batch):
    if "image" in batch:
        # This dataset already has "image" defined.
        return batch
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(
            executor.map(
                partial(
                    fetch_single_image,
                    timeout=timeout,
                    retries=retries,
                    sleep_timer=sleep_timer,
                ),
                batch["image_url"],
            )
        )
    '''
    img = []
    for image_url in batch["image_url"]:
        image_name = image_url.split("/")[-1]
        directory = '/cluster/home/guanmengyuan/multimodal-main/examples/red_caps_download/jellyfish'
        image_path = os.path.join(directory, image_name)
        if os.path.exists (image_path):
            image = Image.open(image_path).convert('RGB')
        else:
            image = None
        # img.append(fetch_single_image(image_url, 0))
        img.append(image)
    batch["image"] = img
    '''
    return batch
