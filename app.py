"""自然语言处理流水线平台 —— Flask 入口。

启动方式::

    python app.py            # 默认 http://127.0.0.1:8000
    python app.py --port 9000
"""

from __future__ import annotations

import argparse
import os
import sys

from flask import Flask, redirect, send_from_directory

# 保证以本目录为基准导入包
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from pipeline import PipelineEngine          # noqa: E402
from storage import StoreRegistry             # noqa: E402
from web import api                           # noqa: E402


def create_app(data_root: str | None = None) -> Flask:
    app = Flask(__name__, static_folder="static", static_url_path="/static")

    if data_root is None:
        data_root = os.path.join(BASE_DIR, "data")
    os.makedirs(data_root, exist_ok=True)

    # 应用级单例，供蓝图通过 current_app.config 访问
    registry = StoreRegistry(data_root, shard_size=100)
    engine = PipelineEngine().register_builtin()
    app.config["DATA_ROOT"] = data_root
    app.config["STORE_REGISTRY"] = registry
    app.config["PIPELINE_ENGINE"] = engine
    app.config["JSON_AS_ASCII"] = False

    app.register_blueprint(api)

    # 页面路由
    @app.get("/")
    def index():
        return redirect("/page/corpus")

    @app.get("/page/<path:name>")
    def page(name: str):
        if not name.endswith(".html"):
            name = name + ".html"
        return send_from_directory(os.path.join(BASE_DIR, "static", "pages"), name)

    return app


def main():
    parser = argparse.ArgumentParser(description="NLP 流水线平台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data", default=None, help="数据存储目录")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app(args.data)
    print(f"* NLP 流水线平台已启动: http://{args.host}:{args.port}")
    print(f"* 数据目录: {app.config['DATA_ROOT']}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
