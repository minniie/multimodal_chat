import argparse

from flask import Flask, request, render_template

from demo.config import (
    ServeConfig,
    ImageRetrieverConfig,
    ResponseGeneratorConfig
)
from model.response_generator import ResponseGenerator
from util.text import (
    truncate_context, 
    join_context, 
    split_context,
    strip_context
)


USER_PREFIX = "User: "
MODEL_PREFIX = "Bot: "


app = Flask(__name__)


@app.route("/", methods=["GET"])
def run():
    return render_template("main.html")


@app.route("/send", methods=["POST"])
def send():
    req_data = request.get_json(force=True)

    # preprocess data 
    context_displayed = req_data["context"]
    context_displayed += "\n" + USER_PREFIX + req_data["message"]
    context_displayed = context_displayed.strip()
    # print(context_displayed)

    context = strip_context(context_displayed, prefixes=[USER_PREFIX, MODEL_PREFIX])
    context = split_context(context, sep="\n")
    context = truncate_context(context, max_context_len=12)
    response = response_generator.inference(context)
    
    # return output
    context_displayed = context_displayed + "\n" + MODEL_PREFIX + response
    res_data = {"context": context_displayed}
    
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