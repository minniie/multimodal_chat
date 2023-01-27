from PIL import Image
from torchvision.transforms import ToTensor
from torchvision import transforms
# from torchvision import transforms
import numpy as np
import torch
from transformers import OFATokenizer, OFAModel
import requests

from util.image import load_image_from_url
from util.text import join_dialog
# from generate import sequence_generator

"""
torch.Size([1, 3, 384, 384])
"""

urls = [
    "https://farm3.staticflickr.com/2674/5850229113_4fe05d5265_z.jpg",
    "http://images.cocodataset.org/val2017/000000039769.jpg"
]
ckpt_dir = 'OFA-Sys/ofa-base'

# mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
# resolution = 384
# patch_resize_transform = transforms.Compose([
#     lambda image: image.convert("RGB"),
#     transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
#     transforms.ToTensor(), 
#     transforms.Normalize(mean=mean, std=std)
# ])  

# img = Image.open(requests.get(urls[0], stream=True).raw)
# patch_img = patch_resize_transform(img).unsqueeze(0)
# print(patch_img.size())

images = [load_image_from_url(url) for url in urls]
tf_toTensor = ToTensor()
tensor_list = []
for i in images:
    tensor_list.append(tf_toTensor(i))
tensor_batch = torch.stack(tensor_list)
print(tensor_batch.size())

tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
model = OFAModel.from_pretrained(ckpt_dir, use_cache=True)
print(tokenizer.pad_token)

dialog = [["hi", "how are you?", "not bad, can you recommend me a restaurant?"], ["what is in this picture?"]]
context = [join_dialog(d, tokenizer.eos_token) + tokenizer.eos_token for d in dialog]
print(f"context\n{context}")

inputs = tokenizer(context, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
print(inputs.input_ids.size())
# img = Image.open(path_to_image)
# patch_img = patch_resize_transform(img).unsqueeze(0)


dummy = model(
    input_ids=inputs.input_ids,
    attention_mask=inputs.attention_mask,
    patch_images=tensor_batch,
    decoder_input_ids=inputs.input_ids, # labels
)
print(dummy.logits.size())



gen = model.generate(
    input_ids=inputs.input_ids,
    patch_images=tensor_batch,
    max_length=64,
    num_beams=1, 
    do_sample=True
)
print(gen)

print(tokenizer.batch_decode(gen, skip_special_tokens=True))
