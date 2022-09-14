import argparse

from flask import Flask, render_template


app = Flask(__name__)


@app.route("/", methods=["GET"])
def run():
    return render_template("index.html")


if __name__ == "__main__":
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