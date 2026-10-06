"""NLP 算法包：分词、词性标注、句法分析、NER、情感、摘要、翻译、关键词、词向量。

对外暴露线程安全的惰性单例，避免为每次请求重建模型。
"""

from __future__ import annotations

import threading

from .segmenter import Segmenter
from .pos import POSTagger, TAG_NAMES, TAGSET
from .parser import DependencyParser, ConstituencyParser, DEP_REL_NAMES, PHRASE_NAMES
from .ner import NERExtractor, ENTITY_TYPE_NAMES
from .sentiment import SentimentAnalyzer, POLARITY_NAMES
from .summarizer import Summarizer
from .translator import Translator
from .keywords import KeywordExtractor
from .embeddings import WordEmbeddings
from . import lexicon, text, hmm, similarity

__all__ = [
    "Segmenter", "POSTagger", "DependencyParser", "ConstituencyParser",
    "NERExtractor", "SentimentAnalyzer", "Summarizer", "Translator",
    "KeywordExtractor", "WordEmbeddings",
    "TAG_NAMES", "TAGSET", "DEP_REL_NAMES", "PHRASE_NAMES", "ENTITY_TYPE_NAMES",
    "POLARITY_NAMES", "lexicon", "text", "hmm", "similarity",
    "get_segmenter", "get_tagger", "get_parser", "get_ner", "get_sentiment",
    "get_summarizer", "get_translator", "get_keywords", "get_embeddings",
]


_lock = threading.Lock()
_instances: dict = {}


def _singleton(name: str, factory):
    with _lock:
        if name not in _instances:
            _instances[name] = factory()
        return _instances[name]


def get_segmenter() -> Segmenter:
    return _singleton("segmenter", Segmenter)


def get_tagger() -> POSTagger:
    return _singleton("tagger", POSTagger)


def get_parser() -> DependencyParser:
    return _singleton("dep_parser", DependencyParser)


def get_constituency_parser() -> ConstituencyParser:
    return _singleton("const_parser", ConstituencyParser)


def get_ner() -> NERExtractor:
    return _singleton("ner", NERExtractor)


def get_sentiment() -> SentimentAnalyzer:
    return _singleton("sentiment", SentimentAnalyzer)


def get_summarizer() -> Summarizer:
    return _singleton("summarizer", Summarizer)


def get_translator() -> Translator:
    return _singleton("translator", Translator)


def get_keywords() -> KeywordExtractor:
    return _singleton("keywords", KeywordExtractor)


def get_embeddings() -> WordEmbeddings:
    return _singleton("embeddings", WordEmbeddings)
