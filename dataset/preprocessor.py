import glob
import json
import requests
import time
import warnings
import psutil
from sys import getsizeof
import numpy as np

from PIL import Image

from util.text import clean_uttr, remove_empty_uttr


IMAGE_DIM = 224


class PhotochatPreprocessor():
    
    def __init__(self):
        self.data_for_image_retriever = {
            "train_set": [], "dev_set": [], "test_set": []
        }
        self.data_for_response_generator = {
            "train_set": [], "dev_set": [], "test_set": []
        }
        self.preprocess()

    def download(self):
        pass

    def preprocess(self):
        # exclude images with any exceptions or warnings
        warnings.simplefilter("error")
        n_except_image = 0

        # iterate through raw dataset
        start_time = time.time()
        start_mem = psutil.virtual_memory().percent
        file_path_list = sorted(glob.glob("dataset/dummy/*/**"))
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
                try:
                    image = Image.open(requests.get(datum["photo_url"], stream=True).raw).convert('RGB').resize((IMAGE_DIM, IMAGE_DIM))
                except Exception as e:
                    n_except_image += 1
                    print(f"Exception #{n_except_image}: {e}")
                    continue
                self.data_for_image_retriever[curr_set].append((dialog_trunc, image))

                # data for response generator
                dialog_alternate = remove_empty_uttr(dialog_alternate)
                dialog_per_uttr = [dialog_alternate[:i+1] for i in range(len(dialog_alternate))]
                self.data_for_response_generator[curr_set].extend(dialog_per_uttr)
        
        warnings.simplefilter("default")
        print(f"{'*'*10} dataset preprocessing time: {(time.time()-start_time)/60:.3f} sec")
        print(f"{'*'*10} dataset memory: {(psutil.virtual_memory().percent-start_mem):.3f} %")

        print(f"{'*'*10} image retriever")
        print(f"{'*'*5} train set: {len(self.data_for_image_retriever['train_set'])}")
        print(f"{'*'*5} dev set: {len(self.data_for_image_retriever['dev_set'])}")
        print(f"{'*'*5} test set: {len(self.data_for_image_retriever['test_set'])}")

        print(f"{'*'*10} response generator")
        print(f"{'*'*5} train set: {len(self.data_for_response_generator['train_set'])}")
        print(f"{'*'*5} dev set: {len(self.data_for_response_generator['dev_set'])}")
        print(f"{'*'*5} test set: {len(self.data_for_response_generator['test_set'])}")