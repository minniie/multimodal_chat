from PIL import Image
import requests
from transformers import AutoProcessor, BlipForQuestionAnswering

model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = AutoProcessor.from_pretrained("Salesforce/blip-vqa-base")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# training
text = "How many cats are in the picture?"
label = "2"
inputs = processor(images=image, text=text, return_tensors="pt")
labels = processor(text=label, return_tensors="pt").input_ids

inputs["labels"] = labels
outputs = model(**inputs)
loss = outputs.loss
loss.backward()

# inference
text = "How many cats are in the picture?"
inputs = processor(images=image, text=text, return_tensors="pt")
outputs = model.generate(**inputs)
print(processor.decode(outputs[0], skip_special_tokens=True))