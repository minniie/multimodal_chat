import os
import glob
import json
import requests

from PIL import Image


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
        file_path_list = sorted(glob.glob("dataset/photochat/*/**"))
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
                    # TODO: image retriever format
                    # images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
                    self.data_for_image_retriever[curr_set].append(
                        [uttr["message"] for uttr in datum["dialogue"]]
                    )
                    self.data_for_response_generator[curr_set].append(
                        [uttr["message"] for uttr in datum["dialogue"]]
                    )
        print(f"{'*'*5} train set: {len(self.data_for_image_retriever['train_set'])}")
        print(f"{'*'*5} dev set: {len(self.data_for_image_retriever['dev_set'])}")
        print(f"{'*'*5} test set: {len(self.data_for_image_retriever['test_set'])}")


# p = PhotochatPreprocessor()
# p.preprocess()