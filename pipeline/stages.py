"""内置流水线阶段：把 NLP 算法封装成可编排的阶段。"""

from __future__ import annotations

from nlp import (get_keywords, get_ner, get_parser, get_segmenter,
                 get_sentiment, get_summarizer, get_tagger, get_translator,
                 get_constituency_parser)
from nlp.lexicon import STOPWORDS

from .stage import Stage


def _clean(ctx, params):
    text = ctx.get("text", "")
    import re
    # 去空白、统一标点
    text = re.sub(r"\s+", " ", text).strip()
    if params.get("remove_stopwords", True):
        seg = get_segmenter()
        words = [w for w in seg.cut(text) if w not in STOPWORDS]
        return {"clean_text": " ".join(words)}
    return {"clean_text": text}


def _segment(ctx, params):
    seg = get_segmenter()
    text = ctx.get("clean_text") or ctx.get("text", "")
    return {"words": seg.cut(text)}


def _pos(ctx, params):
    tagger = get_tagger()
    text = ctx.get("clean_text") or ctx.get("text", "")
    return {"pos": [[w, t] for w, t in tagger.tag(text)]}


def _ner(ctx, params):
    ner = get_ner()
    text = ctx.get("clean_text") or ctx.get("text", "")
    return {"ner": ner.recognize(text)}


def _sentiment(ctx, params):
    text = ctx.get("clean_text") or ctx.get("text", "")
    return {"sentiment": get_sentiment().analyze(text)}


def _keywords(ctx, params):
    text = ctx.get("clean_text") or ctx.get("text", "")
    return {"keywords": get_keywords().extract(text, top_k=params.get("top_k", 10))}


def _summary(ctx, params):
    text = ctx.get("clean_text") or ctx.get("text", "")
    return {"summary": get_summarizer().summarize(
        text, ratio=params.get("ratio", 0.3),
        max_sentences=params.get("max_sentences"))}


def _translate(ctx, params):
    text = ctx.get("clean_text") or ctx.get("text", "")
    return {"translation": get_translator().translate(
        text, direction=params.get("direction", "zh2en"))}


def _parse(ctx, params):
    text = ctx.get("clean_text") or ctx.get("text", "")
    dep = get_parser().parse(text)
    const = get_constituency_parser().parse(text)
    return {"parse": {"dependency": dep, "constituency": const}}


BUILTIN_STAGES = [
    Stage("clean", _clean, inputs=["text"], outputs=["clean_text"],
          description="文本清洗：去空白、去停用词", params={"remove_stopwords": True}),
    Stage("segment", _segment, inputs=["text", "clean_text"], outputs=["words"],
          description="中文分词"),
    Stage("pos", _pos, inputs=["text", "clean_text"], outputs=["pos"],
          description="词性标注"),
    Stage("ner", _ner, inputs=["text", "clean_text"], outputs=["ner"],
          description="命名实体识别"),
    Stage("sentiment", _sentiment, inputs=["text", "clean_text"], outputs=["sentiment"],
          description="情感分析"),
    Stage("keywords", _keywords, inputs=["text", "clean_text"], outputs=["keywords"],
          description="关键词提取", params={"top_k": 10}),
    Stage("summary", _summary, inputs=["text", "clean_text"], outputs=["summary"],
          description="文本摘要", params={"ratio": 0.3}),
    Stage("translate", _translate, inputs=["text", "clean_text"], outputs=["translation"],
          description="机器翻译（模拟）", params={"direction": "zh2en"}),
    Stage("parse", _parse, inputs=["text", "clean_text"], outputs=["parse"],
          description="句法分析"),
]
