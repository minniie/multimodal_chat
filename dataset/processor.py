import glob
import json
import os

import torch

from util.text import clean_uttr, remove_empty_uttr
from util.image import load_image_from_url


class PhotochatProcessor():
    
    def __init__(self):
        self.data_for_image_retriever = {
            "train_set": [], "dev_set": [], "test_set": []
        }
        self.data_for_response_generator = {
            "train_set": [], "dev_set": [], "test_set": []
        }
        self.images = []

    def split(self, path):
        # iterate through raw dataset
        file_path_list = sorted(glob.glob(path+"/*/**"))
        for file_path in file_path_list:
            with open(file_path) as f:
                data = json.load(f)
            for datum in data:
                if "train" in file_path:
                    curr_set = "train_set"
                elif "dev" in file_path:
                    curr_set = "dev_set"
                elif "test" in file_path:
                    curr_set = "test_set"

                # make dialog alternate btw two speakers
                dialog_alternate = []
                for turn in datum["dialogue"]:
                    if not dialog_alternate:
                        dialog_alternate.append(turn)
                    else:
                        if turn["user_id"] == dialog_alternate[-1]["user_id"]:
                            dialog_alternate[-1]["message"] += " " + turn["message"] 
                        else:
                            dialog_alternate.append(turn)
                    if turn["share_photo"]:
                        share_photo_idx = len(dialog_alternate)-1
                dialog_alternate = [d["message"] for d in dialog_alternate]
                dialog_alternate = clean_uttr(dialog_alternate)
                
                # data for image retriever
                dialog_trunc = remove_empty_uttr(dialog_alternate[:share_photo_idx+1])
                self.data_for_image_retriever[curr_set].append((dialog_trunc, datum["photo_url"]))

                # data for response generator
                dialog_alternate = remove_empty_uttr(dialog_alternate)
                dialog_per_uttr = [dialog_alternate[:i+1] for i in range(len(dialog_alternate))]
                self.data_for_response_generator[curr_set].extend(dialog_per_uttr)

        print(f"{'*'*10} image retriever")
        print(f"{'*'*5} train set: {len(self.data_for_image_retriever['train_set'])}")
        print(f"{'*'*5} dev set: {len(self.data_for_image_retriever['dev_set'])}")
        print(f"{'*'*5} test set: {len(self.data_for_image_retriever['test_set'])}")

        print(f"{'*'*10} response generator")
        print(f"{'*'*5} train set: {len(self.data_for_response_generator['train_set'])}")
        print(f"{'*'*5} dev set: {len(self.data_for_response_generator['dev_set'])}")
        print(f"{'*'*5} test set: {len(self.data_for_response_generator['test_set'])}")

    def load(self, raw_path, processed_path, processor):
        # load processed images
        if os.path.exists(processed_path):
            images_processed = torch.load(processed_path)
        
        # iterate through raw dataset and save processed images
        else:
            image_urls = []
            file_path_list = sorted(glob.glob(raw_path+"/*/**"))
            for file_path in file_path_list:
                with open(file_path) as f:
                    data = json.load(f)
                image_urls.extend(d["photo_url"] for d in data)
            image_urls = image_urls[:10]
            images = [load_image_from_url(u) for u in image_urls]
            images_processed = processor(images=images, return_tensors="pt")
            images_processed = images_processed.pixel_values
            torch.save(images_processed, processed_path)            
        
        self.images = images_processed