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
from util.io import save_dialog
from util.text import truncate_dialog
from util.time import get_date
from util.resource import set_device, get_device_util


app = Flask(__name__)


@app.route("/", methods=["GET"])
def run():
    return render_template("main.html")


@app.route("/evaluation", methods=["GET"])
def evaluation():
    return render_template("evaluation.html")


@app.route("/send", methods=["POST"])
def send():
    req_data = request.get_json(force=True)
    context = req_data["context"]
    context = truncate_dialog(context, max_context_len=12)

    # get top 1 image
    image_url = None
    if image_retriever_config.use_model:
        probs_per_image = image_retriever.inference(context, images)
        val, idx = torch.topk(probs_per_image, 1)
        print(f"top 1 probability: {val.item()}")
        if val.item() > 0.2:
            image_url = images[1][idx]
    
    # get response
    response = response_generator.inference(context, image_url)
    
    # return data
    res_data = {"image_url": image_url, "bot_response": response}
    return res_data


@app.route("/save", methods=["POST"])
def save():
    req_data = request.get_json(force=True)

    # preprocess data
    context = req_data["context"]
    context, response = context[:-1], context[-1]
    evaluations = req_data["evaluations"]
    name = req_data["name"]
    date = get_date()

    # save annotated dialog
    dialog = {
        "context": context,
        "response": response,
        "evaluations": evaluations
    }
    workload = save_dialog(dialog, data_config.evaluation_path, name, date, "turns.json")
    
    # return data
    res_data = {"workload": workload}
    return res_data


@app.route("/done", methods=["POST"])
def done():
    req_data = request.get_json(force=True)

    # preprocess data
    context = req_data["context"]
    username = req_data["username"]
    date = get_date()

    # save annotated dialog
    dialog = split_context(context, DomainVar.DELIMITER)
    workload = save_dialogs(dialog, data_config.eval_dir_path, username, date, DomainVar.DONE_PIPELINE_FILE_NAME)
 
    res_data = {"workload": workload}
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
        retriever_image_encoder_path=image_retriever_config.image_encoder_path,
        retriever_text_encoder_path=image_retriever_config.text_encoder_path,
        retriever_finetuned_path=image_retriever_config.finetuned_path
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
        generator_image_encoder_path=response_generator_config.image_encoder_path,
        generator_text_decoder_path=response_generator_config.text_decoder_path,
        generator_finetuned_path=response_generator_config.finetuned_path
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