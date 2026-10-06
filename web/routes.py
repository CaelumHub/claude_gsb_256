"""Flask API 路由。

把所有 NLP 能力、存储与流水线编排暴露为 REST 接口，
前端 10 个页面通过 ``fetch`` 调用这些接口。
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Optional

from flask import Blueprint, current_app, jsonify, request

from nlp import (get_constituency_parser, get_embeddings, get_keywords, get_ner,
                 get_parser, get_segmenter, get_sentiment, get_summarizer,
                 get_tagger, get_translator, ENTITY_TYPE_NAMES, TAG_NAMES,
                 DEP_REL_NAMES, PHRASE_NAMES, POLARITY_NAMES)
from nlp.lexicon import STOPWORDS
from nlp.similarity import DEFAULT_THRESHOLD
from storage import StoreRegistry


api = Blueprint("api", __name__, url_prefix="/api")


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _registry() -> StoreRegistry:
    return current_app.config["STORE_REGISTRY"]


def _engine():
    return current_app.config["PIPELINE_ENGINE"]


def _similarity():
    return current_app.config["SIMILARITY_SERVICE"]


def _models_dir() -> str:
    import os
    path = os.path.join(current_app.config["DATA_ROOT"], "models")
    os.makedirs(path, exist_ok=True)
    return path


def _store_result(task: str, text: str, result: dict,
                  corpus_id: Optional[str] = None) -> str:
    record = {"text": text, "result": result, "created_at": time.time()}
    if corpus_id:
        record["corpus_id"] = corpus_id
    return _registry().task(task).insert(record)


def _payload() -> dict:
    data = request.get_json(silent=True) or {}
    return data


def _resolve_text(data: dict) -> tuple[str, Optional[str]]:
    """从请求中取文本：优先 text，其次 corpus_id。"""
    if data.get("text"):
        return data["text"], data.get("corpus_id")
    corpus_id = data.get("corpus_id")
    if corpus_id:
        record = _registry().task("corpus").get(corpus_id)
        if record:
            return record.get("text", ""), corpus_id
        return "", corpus_id
    return "", None


def _clean(text: str, remove_stopwords: bool = True) -> dict:
    text = re.sub(r"\s+", " ", text).strip()
    seg = get_segmenter()
    words = seg.cut(text)
    if remove_stopwords:
        kept = [w for w in words if w not in STOPWORDS]
    else:
        kept = words
    removed = len(words) - len(kept)
    return {
        "text": text,
        "cleaned": " ".join(kept),
        "tokens": kept,
        "original_tokens": words,
        "removed_stopwords": removed,
    }


# ---------------------------------------------------------------------------
# 状态
# ---------------------------------------------------------------------------

@api.get("/status")
def status():
    return jsonify({
        "ok": True,
        "version": "1.0.0",
        "tasks": _registry().tasks(),
        "time": time.time(),
    })


@api.get("/meta")
def meta():
    """给前端提供标签集合与可配置参数。"""
    return jsonify({
        "tag_names": TAG_NAMES,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
        "entity_type_names": ENTITY_TYPE_NAMES,
        "polarity_names": POLARITY_NAMES,
        "directions": [{"id": "zh2en", "name": "中文 → 英文"},
                       {"id": "en2zh", "name": "英文 → 中文"}],
    })


# ---------------------------------------------------------------------------
# 语料库管理
# ---------------------------------------------------------------------------

@api.get("/corpus")
def list_corpus():
    records = _registry().task("corpus").all()
    items = [{
        "id": r.get("id"),
        "name": r.get("name", "未命名"),
        "length": len(r.get("text", "")),
        "created_at": r.get("created_at"),
        "preview": r.get("text", "")[:80],
    } for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"corpora": items})


@api.post("/corpus")
def create_corpus():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "语料内容不能为空"}), 400
    record = {
        "name": data.get("name") or f"语料_{int(time.time())}",
        "text": text,
        "created_at": time.time(),
    }
    rid = _registry().task("corpus").insert(record)
    _index_hook(rid, text)
    return jsonify({"id": rid, "ok": True})


def _index_hook(rid: str, text: str) -> None:
    """语料写入后同步相似度索引；失败不影响主流程（索引可自愈）。"""
    try:
        _similarity().on_corpus_added(rid, text)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.warning("相似度索引更新失败（将在下次检索时自愈）: %s", exc)


@api.post("/corpus/upload")
def upload_corpus():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "未接收到文件"}), 400
    raw = file.read()
    text = None
    for enc in ("utf-8", "gbk", "gb18030", "utf-16"):
        try:
            text = raw.decode(enc)
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if text is None:
        return jsonify({"error": "无法解码文件内容"}), 400
    name = data_name = file.filename or "上传文件"
    record = {"name": name, "text": text.strip(), "created_at": time.time()}
    rid = _registry().task("corpus").insert(record)
    _index_hook(rid, record["text"])
    return jsonify({"id": rid, "name": name, "length": len(text), "ok": True})


@api.get("/corpus/<cid>")
def get_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    return jsonify(record)


@api.delete("/corpus/<cid>")
def delete_corpus(cid: str):
    ok = _registry().task("corpus").delete(cid)
    if ok:
        try:
            _similarity().on_corpus_deleted(cid)
        except Exception as exc:  # noqa: BLE001
            current_app.logger.warning("相似度索引清理失败（将在下次检索时自愈）: %s", exc)
    return jsonify({"ok": ok})


@api.post("/corpus/<cid>/clean")
def clean_corpus(cid: str):
    record = _registry().task("corpus").get(cid)
    if not record:
        return jsonify({"error": "语料不存在"}), 404
    data = _payload()
    result = _clean(record.get("text", ""), data.get("remove_stopwords", True))
    _store_result("clean", record.get("text", ""), result, corpus_id=cid)
    return jsonify(result)


# ---------------------------------------------------------------------------
# 相似检索与全库查重
# ---------------------------------------------------------------------------

@api.post("/similarity/search")
def similarity_search():
    """对任意文本（或库内文档）检索最相似的语料。

    跨全部分片汇总打分后全局排序；排序键为 ``(-score, id)`` 全序，
    ``offset/limit`` 翻页连续、不重不漏。
    """
    data = _payload()
    text = (data.get("text") or "").strip()
    corpus_id = data.get("corpus_id")
    if not text and not corpus_id:
        return jsonify({"error": "缺少查询文本或 corpus_id"}), 400
    try:
        limit = min(max(int(data.get("limit", 10)), 1), 100)
        offset = max(int(data.get("offset", 0)), 0)
        min_score = min(max(float(data.get("min_score", 0.0)), 0.0), 1.0)
    except (TypeError, ValueError):
        return jsonify({"error": "limit/offset/min_score 参数非法"}), 400
    try:
        result = _similarity().search(
            text=text or None, corpus_id=corpus_id,
            limit=limit, offset=offset, min_score=min_score,
            include_self=bool(data.get("include_self")))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(result)


@api.post("/dedup/run")
def dedup_run():
    """对整个语料库跑一轮查重，把高度相似的文档归成一组。"""
    data = _payload()
    try:
        threshold = float(data.get("threshold", DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        return jsonify({"error": "threshold 参数非法"}), 400
    if not 0.0 < threshold <= 1.0:
        return jsonify({"error": "threshold 需在 (0, 1] 区间"}), 400
    return jsonify(_similarity().run_dedup(threshold))


@api.get("/dedup/groups")
def dedup_groups():
    """查看最近一次查重的归组结果（stale=true 表示语料已变化，建议重跑）。"""
    return jsonify(_similarity().groups())


@api.post("/similarity/reindex")
def similarity_reindex():
    """全量重建相似度索引（特征口径升级或索引异常时使用）。"""
    return jsonify(_similarity().reindex())


# ---------------------------------------------------------------------------
# 分词与词性标注
# ---------------------------------------------------------------------------

@api.post("/segment")
def segment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    seg = get_segmenter()
    words = seg.cut(text)
    result = {"words": words, "count": len(words)}
    rid = _store_result("segment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/pos")
def pos_tag():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    tagger = get_tagger()
    tokens = [[w, t] for w, t in tagger.tag(text)]
    result = {"tokens": tokens, "tag_names": TAG_NAMES}
    rid = _store_result("pos", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 句法分析
# ---------------------------------------------------------------------------

@api.post("/parse")
def parse():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    dep = get_parser().parse(text)
    const = get_constituency_parser().parse(text)
    result = {
        "dependency": dep,
        "constituency": const,
        "dep_rel_names": DEP_REL_NAMES,
        "phrase_names": PHRASE_NAMES,
    }
    rid = _store_result("parse", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 命名实体识别与标注
# ---------------------------------------------------------------------------

@api.post("/ner")
def ner():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    entities = get_ner().recognize(text)
    result = {"entities": entities, "entity_type_names": ENTITY_TYPE_NAMES}
    rid = _store_result("ner", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


@api.post("/ner/annotate")
def ner_annotate():
    data = _payload()
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    record = {
        "text": text,
        "entities": data.get("entities", []),
        "note": data.get("note", ""),
        "created_at": time.time(),
    }
    rid = _registry().task("annotation").insert(record)
    return jsonify({"id": rid, "ok": True})


@api.get("/ner/annotations")
def ner_annotations():
    records = _registry().task("annotation").all()
    return jsonify({"annotations": records})


# ---------------------------------------------------------------------------
# 情感分析
# ---------------------------------------------------------------------------

@api.post("/sentiment")
def sentiment():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_sentiment().analyze(text)
    result["polarity_name"] = POLARITY_NAMES.get(result["polarity"], "")
    rid = _store_result("sentiment", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 文本摘要
# ---------------------------------------------------------------------------

@api.post("/summary")
def summary():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_summarizer().summarize(
        text, ratio=data.get("ratio", 0.3),
        max_sentences=data.get("max_sentences"))
    rid = _store_result("summary", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 机器翻译（模拟）
# ---------------------------------------------------------------------------

@api.post("/translate")
def translate():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_translator().translate(text, direction=data.get("direction", "zh2en"))
    rid = _store_result("translate", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 关键词提取
# ---------------------------------------------------------------------------

@api.post("/keywords")
def keywords():
    data = _payload()
    text, cid = _resolve_text(data)
    if not text:
        return jsonify({"error": "缺少文本"}), 400
    result = get_keywords().extract(text, top_k=data.get("top_k", 10),
                                    method=data.get("method", "hybrid"))
    rid = _store_result("keywords", text, result, corpus_id=cid)
    result["id"] = rid
    return jsonify(result)


# ---------------------------------------------------------------------------
# 词向量
# ---------------------------------------------------------------------------

def _embedding_path() -> str:
    import os
    return os.path.join(_models_dir(), "embeddings.json")


@api.post("/embeddings/train")
def train_embeddings():
    data = _payload()
    corpus_ids = data.get("corpus_ids")
    store = _registry().task("corpus")
    if corpus_ids:
        texts = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
    else:
        texts = [r["text"] for r in store.all() if not r.get("_deleted")]
    if not texts:
        return jsonify({"error": "没有可用语料，请先上传语料"}), 400

    emb = get_embeddings()
    emb.train(texts, vocab_size=data.get("vocab_size", 200),
              dim=data.get("dim", 20), window=data.get("window", 5),
              min_count=data.get("min_count", 1))

    payload = {
        "vocab": emb.vocab,
        "vectors": emb.vectors,
        "dim": emb.dim,
        "trained_at": time.time(),
        "corpus_count": len(texts),
    }
    with open(_embedding_path(), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return jsonify(emb.stats())


@api.get("/embeddings/vectors")
def embeddings_vectors():
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    if not emb.vectors:
        return jsonify({"error": "尚未训练词向量"}), 404
    n_clusters = int(request.args.get("clusters", 5))
    proj = emb.project_2d()
    clusters = emb.cluster(n_clusters)
    return jsonify({
        "points": [{"word": w, "x": round(p[0], 4), "y": round(p[1], 4),
                    "cluster": clusters.get(w, 0)} for w, p in proj.items()],
        "stats": emb.stats(),
    })


@api.get("/embeddings/neighbors")
def embeddings_neighbors():
    word = request.args.get("word", "")
    k = int(request.args.get("k", 10))
    emb = get_embeddings()
    if not emb.vectors:
        _load_embeddings()
        emb = get_embeddings()
    return jsonify({"word": word, "neighbors": emb.nearest(word, k)})


def _load_embeddings():
    import os
    path = _embedding_path()
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        emb = get_embeddings()
        emb.vocab = data.get("vocab", [])
        emb.vectors = data.get("vectors", {})
        emb.dim = data.get("dim", 0)
    except (json.JSONDecodeError, OSError):
        pass


# ---------------------------------------------------------------------------
# 流水线配置与执行
# ---------------------------------------------------------------------------

@api.get("/pipeline/stages")
def pipeline_stages():
    return jsonify({"stages": _engine().list_stages()})


@api.post("/pipeline")
def save_pipeline():
    data = _payload()
    config = data.get("config") or data
    if not config.get("stages"):
        return jsonify({"error": "流水线至少需要一个阶段"}), 400
    name = config.get("name") or f"流水线_{int(time.time())}"
    record = {"name": name, "config": config, "created_at": time.time()}
    rid = _registry().task("pipeline_config").insert(record)
    return jsonify({"id": rid, "name": name, "ok": True})


@api.get("/pipeline")
def list_pipelines():
    records = _registry().task("pipeline_config").all()
    items = [{"id": r["id"], "name": r.get("name"), "config": r.get("config"),
              "created_at": r.get("created_at")}
             for r in records if not r.get("_deleted")]
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return jsonify({"pipelines": items})


@api.get("/pipeline/<pid>")
def get_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    return jsonify(record)


@api.post("/pipeline/preview")
def pipeline_preview():
    """对单条文本跑流水线（不持久化），供配置页预览。"""
    data = _payload()
    text = (data.get("text") or "").strip()
    config = data.get("config")
    if not text or not config:
        return jsonify({"error": "缺少文本或配置"}), 400
    try:
        result = _engine().build(config).run({"text": text})
        return jsonify({"ok": True, "output": result})
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 400


@api.post("/pipeline/<pid>/run")
def run_pipeline(pid: str):
    record = _registry().task("pipeline_config").get(pid)
    if not record:
        return jsonify({"error": "流水线不存在"}), 404
    config = record.get("config")
    data = _payload()

    run_id = uuid.uuid4().hex[:12]
    started = time.time()

    if data.get("batch"):
        # 批量：对语料库中的多篇文档执行
        corpus_ids = data.get("corpus_ids") or []
        store = _registry().task("corpus")
        docs = []
        if corpus_ids:
            docs = [store.get(c)["text"] for c in corpus_ids if store.get(c)]
        else:
            docs = [r["text"] for r in store.all() if not r.get("_deleted")]
        if not docs:
            return jsonify({"error": "没有可处理的文档"}), 400

        progress_state = {"done": 0, "total": len(docs)}

        def _progress(done, total):
            progress_state["done"] = done
            progress_state["total"] = total

        results = _engine().run_batch(
            config, docs, shared=data.get("shared"),
            max_workers=data.get("max_workers", 4),
            chunk_size=data.get("chunk_size", 16),
            progress=_progress)
        succeeded = sum(1 for r in results if r and r["ok"])
        failed = len(results) - succeeded
        run_record = {
            "run_id": run_id, "pipeline_id": pid, "batch": True,
            "doc_count": len(docs), "succeeded": succeeded, "failed": failed,
            "started": started, "finished": time.time(),
            "results": results,
        }
        rid = _registry().task("pipeline_run").insert(run_record)
        return jsonify({"run_id": run_id, "id": rid, "succeeded": succeeded,
                        "failed": failed, "doc_count": len(docs)})
    else:
        text = (data.get("text") or "").strip()
        if not text:
            return jsonify({"error": "缺少文本"}), 400
        try:
            output = _engine().build(config).run({"text": text})
            run_record = {
                "run_id": run_id, "pipeline_id": pid, "batch": False,
                "text": text, "output": output,
                "started": started, "finished": time.time(),
            }
            rid = _registry().task("pipeline_run").insert(run_record)
            return jsonify({"run_id": run_id, "id": rid, "ok": True,
                            "output": output})
        except Exception as exc:  # noqa: BLE001
            return jsonify({"ok": False, "error": str(exc)}), 400


@api.get("/pipeline/run/<run_id>")
def get_pipeline_run(run_id: str):
    records = _registry().task("pipeline_run").query(
        where=[("run_id", "eq", run_id)])
    if not records:
        return jsonify({"error": "执行记录不存在"}), 404
    return jsonify(records[0])


# ---------------------------------------------------------------------------
# 结果查询（分片合并与查询）
# ---------------------------------------------------------------------------

@api.get("/results")
def list_result_tasks():
    registry = _registry()
    tasks = []
    for name in registry.tasks():
        if name in ("corpus", "pipeline_config", "annotation"):
            continue
        stats = registry.task(name).stats()
        tasks.append(stats)
    return jsonify({"tasks": tasks})


@api.get("/results/<task>")
def query_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    store = registry.task(task)
    where = []
    for key in ("type", "corpus_id"):
        val = request.args.get(key)
        if val:
            where.append((key, "eq", val))
    order_by = request.args.get("order_by")
    order = request.args.get("order", "desc")
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", 0, type=int)
    records = store.query(where=where or None, order_by=order_by,
                          order=order, limit=limit, offset=offset)
    return jsonify({
        "task": task,
        "count": len(records),
        "stats": store.stats(),
        "records": records,
    })


@api.post("/results/<task>/compact")
def compact_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).compact())


@api.get("/results/<task>/merge")
def merge_results(task: str):
    registry = _registry()
    if task not in registry.tasks():
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(registry.task(task).merge())


@api.post("/results/compact_all")
def compact_all():
    return jsonify({"compacted": _registry().compact_all()})
