import argparse

from flask import Flask, request, render_template

from demo.config import (
    ServeConfig,
    ImageRetrieverConfig,
    ResponseGeneratorConfig
)
from model.response_generator import ResponseGenerator
from util.text import truncate_context


app = Flask(__name__)


@app.route("/", methods=["GET"])
def run():
    return render_template("main.html")


@app.route("/send", methods=["POST"])
def send():
    req_data = request.get_json(force=True)
    context = req_data["context"]
    context = truncate_context(context, max_context_len=12)
    response = response_generator.inference(context)
    res_data = {"bot_response": response}
    
    return res_data


if __name__ == "__main__":
    # configs
    serve_config = ServeConfig()
    image_retriever_config = ImageRetrieverConfig()
    response_generator_config = ResponseGeneratorConfig()

    # load response generator
    print(f"{'*'*10} Loading response generator")
    response_generator = ResponseGenerator(response_generator_config.generator_model_path)

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