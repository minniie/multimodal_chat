import argparse

import torch
from flask import Flask, request, render_template

from demo.config import (
    ServeConfig,
    ImageRetrieverConfig,
    ResponseGeneratorConfig,
    DataConfig
)
from model.image_retriever import ImageRetriever
from model.response_generator import ResponseGenerator
from dataset.processor import PhotochatProcessor
from util.text import truncate_dialog
from util.resource import set_device, get_device_util


app = Flask(__name__)


@app.route("/", methods=["GET"])
def run():
    return render_template("main.html")


@app.route("/send", methods=["POST"])
def send():
    req_data = request.get_json(force=True)
    context = req_data["context"]
    context = truncate_dialog(context, max_context_len=12)
    
    # get response
    response = response_generator.inference(context)

    # get top 1 image
    probs_per_image = image_retriever.inference(device, context, images)
    idx = torch.argmax(probs_per_image, dim=-1).item()
    image_url = images[1][idx]
    
    res_data = {"bot_response": response, "image_url": image_url}
    
    return res_data


if __name__ == "__main__":
    # configs
    serve_config = ServeConfig()
    image_retriever_config = ImageRetrieverConfig()
    response_generator_config = ResponseGeneratorConfig()
    data_config = DataConfig()

    # set device
    device = set_device()

    # load image retriever
    print(f"{'*'*10} Loading image retriever")
    image_retriever = ImageRetriever(
        device=device,
        retriever_finetuned_path=image_retriever_config.model_path
    )

    # load images
    print(f"{'*'*10} Loading images")
    images = image_retriever.load_images(
        device=device,
        dataset_path=data_config.images_dataset_path,
        encoding_path=data_config.images_encoding_path
    )

    # load response generator
    print(f"{'*'*10} Loading response generator")
    response_generator = ResponseGenerator(
        device=device,
        generator_text_decoder_path=response_generator_config.model_path
    )

    # get device util
    print(f"{'*'*10} Device util after loading all models and data")
    get_device_util()

    # args for app
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9810)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--debug", action="store_true", default=False)
    args, _ = parser.parse_known_args()
    app.run(
        port=args.port, 
        host=args.host, 
        debug=args.debug, 
        ssl_context=None
    )