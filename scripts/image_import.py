import requests
import os
import pandas as pd
import re
from PIL import Image
from shutil import move
from argparse import ArgumentParser
from pathlib import Path
from requests.exceptions import ConnectionError, HTTPError
import hashlib
import hmac
import base64
import urllib.parse as urlparse
def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--data_path",
        type=Path
    )
    args.add_argument(
        "--split",
        type=str,
        default = "train"
    )
    args.add_argument(
        "--api_key",
        type=str
    )
    args.add_argument(
        "--base_url",
        type=str
    )
    args.add_argument(
        "--secret",
        type=str
    )

    return args.parse_args()

def extract_ids(input_string):
    '''
    Extract the pano id and the number idnciating the part of 360 panorama from the image id
    :param input_string: str, image id
    '''
    id_match = re.match(r'^(.{22})', input_string)
    number_match = re.search(r'_(\d+).png', input_string)
    id_value = id_match.group(1) if id_match else None
    number_value = number_match.group(1) if number_match else None
    return id_value, number_value

def load_data(data_path,split,api_key,base_url,secret):
    '''
    Load the images from the Google Street View API
    :param data_path: str, path to the data
    :param split: str, split of the dataset
    :param api_key: str, street vGoogle Street View API
    :param base_url: str, base url of the Google Street View API
    :param secret: str, secret key for the API
    '''
    in_dir = os.path.join(data_path, split)  
    out_dir = os.path.join(in_dir, "images")
    images  = pd.read_csv(os.path.join(in_dir, f"{split}.csv"))
    params = {
        'size': '224x224',
        'fov': '90',
        'pitch': '0',
        'key': api_key
    }
    csv_rows = []
    for i in range(len(images)):
        id, number = extract_ids(images['IMG_ID'][i])
        params['pano'] = id
        # Set the heading degree based on the number of the image
        headings = [45, 135, 225, 315]
        params['heading'] = headings[int(number)]
        url = urlparse.urlparse(base_url + urlparse.urlencode(params))
        url_to_sign = url.path + "?" + url.query
        decoded_key = base64.urlsafe_b64decode(secret)
        # Create a signature using the HMAC-SHA1 algorithm
        signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)
        encoded_signature = base64.urlsafe_b64encode(signature.digest())
        original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query
        # Add the signature to the URL
        signed_url = original_url + "&signature=" + encoded_signature.decode()
        try:
                response = requests.get(signed_url)
                response.raise_for_status()  
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, images['IMG_ID'][i]), 'wb') as f:
                    f.write(response.content)
                if i % 100 == 0:
                    print(f"Downloaded {i}/{len(images)} images from {split} split.")
                csv_rows.append(images.iloc[i])
        except ConnectionError as e:
            print(f'ConnectionError occurred while fetching image: {e}')
        except HTTPError as e:
            print(f'HTTPError occurred while fetching image: {e.response.status_code}')
        except Exception as e:
            print(f'An unexpected error occurred: {e}')
    csv_outfile = os.path.join(in_dir, f"{split}.csv")
    df_out = pd.DataFrame(csv_rows,columns=['IMG_ID'])
    df_out.to_csv(os.path.join(in_dir,csv_outfile), index=False)

if __name__ == "__main__":
    args = parse_args()
    load_data(args.data_path,args.split,args.api_key,args.base_url,args.secret)
