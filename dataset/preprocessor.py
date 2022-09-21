import glob
import json

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
                    # TODO: dataset for image retriever
                    # images = [Image.open(requests.get(url, stream=True).raw) for url in urls]
                    self.data_for_image_retriever[curr_set].append(
                        [uttr["message"] for uttr in datum["dialogue"]]
                    )

                    # dataset for response generator
                    curr_dialog = []
                    curr_user_id = -1
                    for turn in datum["dialogue"]:
                        if not turn["message"]:
                            continue
                        if curr_user_id == turn["user_id"]:
                            curr_dialog[-1] += " " + turn["message"]
                        else:
                            curr_dialog.append(turn["message"])
                        curr_user_id = turn["user_id"]
                    for i in range(len(curr_dialog)):  
                        self.data_for_response_generator[curr_set].append(curr_dialog[:i+1])

        print(f"{'*'*5} train set: {len(self.data_for_response_generator['train_set'])}")
        print(f"{'*'*5} dev set: {len(self.data_for_response_generator['dev_set'])}")
        print(f"{'*'*5} test set: {len(self.data_for_response_generator['test_set'])}")


# p = PhotochatPreprocessor()
# p.preprocess()