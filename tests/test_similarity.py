"""相似检索与全库查重的单元测试。

运行：``python -m unittest discover -s tests -v``
覆盖：特征提取、相似度口径（同义改写 / 语序调整 / 摘录 / 主题相近）、
MinHash 候选对、并查集归组、服务层（跨分片检索、稳定分页、
增量维护、归组稳定性、口径一致性）、HTTP API。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp import get_segmenter
from nlp.lexicon import SYNONYM_GROUPS, SYNONYM_MAP
from nlp.similarity import (DEFAULT_THRESHOLD, DocFeatures, all_pairs,
                            candidate_pairs, cluster, content_tokens,
                            extract_features, normalize_text, score_features)
from storage import StoreRegistry
from web.similarity import SimilarityService


# ---------------------------------------------------------------------------
# 测试语料
# ---------------------------------------------------------------------------

BASE = ("人工智能技术正在深刻地改变着我们的生活方式。计算机视觉、自然语言处理和机器学习等方法"
        "被广泛应用于医疗、教育和金融等领域。数据的积累与算法的进步相互促进，大幅提高了系统的"
        "智能水平。未来，这些技术还将帮助人类解决更多复杂的问题，创造更大的社会价值。")

# 同义改写：计算机→电脑、方法→办法、数据→资料、提高→提升、技术→技艺、帮助→协助、改变→更改
REWORDED = ("人工智能技术正在深刻地更改着我们的生活方式。电脑视觉、自然语言处理和机器学习等办法"
            "被广泛应用于医疗、教育和金融等领域。资料的积累与算法的进步相互促进，大幅提升了系统的"
            "智能水平。未来，这些技艺还将协助人类解决更多复杂的问题，创造更大的社会价值。")

# 语序调整：句子顺序打乱
REORDERED = ("数据的积累与算法的进步相互促进，大幅提高了系统的智能水平。"
             "未来，这些技术还将帮助人类解决更多复杂的问题，创造更大的社会价值。"
             "人工智能技术正在深刻地改变着我们的生活方式。"
             "计算机视觉、自然语言处理和机器学习等方法被广泛应用于医疗、教育和金融等领域。")

# 局部摘录：原文前两句
EXCERPT = ("人工智能技术正在深刻地改变着我们的生活方式。"
           "计算机视觉、自然语言处理和机器学习等方法被广泛应用于医疗、教育和金融等领域。")

# 主题相近但内容不同
TOPIC_SIMILAR = ("深度学习模型通常需要大量的标注数据才能训练出理想的效果。神经网络通过反向传播"
                 "算法不断调整参数，在图像识别、语音合成等任务上取得了突破性进展。研究人员正在"
                 "探索更高效的训练策略，以降低模型对计算资源的依赖，让人工智能应用落地到更多场景。")

DISTRACTORS = [
    "今天的天气格外晴朗，气温回升到了二十度以上，公园里到处都是踏青的人群。",
    "本场足球比赛双方攻防转换极快，主队在下半场连入两球，最终锁定胜局。",
    "红烧肉的做法并不复杂：五花肉切块焯水，加冰糖炒出糖色，小火慢炖一小时即可。",
    "受利好消息影响，大盘高开高走，成交量明显放大，板块轮动加快。",
    "这次旅行我们沿着海岸线一路向南，途经三个渔村，品尝了最新鲜的海产。",
]


def _feat(text):
    return extract_features(text, get_segmenter())


# ---------------------------------------------------------------------------
# 算法层
# ---------------------------------------------------------------------------

class TestFeatures(unittest.TestCase):
    def test_synonym_groups_disjoint(self):
        seen = {}
        for i, group in enumerate(SYNONYM_GROUPS):
            for word in group:
                self.assertNotIn(word, seen,
                                 f"词 {word!r} 同时出现在组 {seen.get(word)} 与组 {i}")
                seen[word] = i
        self.assertEqual(len(SYNONYM_MAP), len(seen))

    def test_normalize_ignores_punct_and_case(self):
        a = normalize_text("人工智能，AI！")
        b = normalize_text("人工智能 ai ")
        self.assertEqual(a, b)

    def test_content_tokens_canonicalize_synonyms(self):
        tokens_a = set(content_tokens("计算机和互联网改变了生活", get_segmenter()))
        tokens_b = set(content_tokens("电脑和因特网改变了生活", get_segmenter()))
        self.assertEqual(tokens_a, tokens_b)

    def test_content_tokens_drop_stopwords(self):
        tokens = content_tokens("我们的产品非常好用", get_segmenter())
        self.assertNotIn("非常", tokens)
        self.assertNotIn("我们", tokens)

    def test_features_roundtrip(self):
        features = _feat(BASE)
        restored = DocFeatures.from_record(features.to_record())
        self.assertEqual(features.tokens, restored.tokens)
        self.assertEqual(features.shingles, restored.shingles)
        self.assertEqual([list(r) for r in features.signature],
                         [list(r) for r in restored.signature])


class TestSimilarityScore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = _feat(BASE)
        cls.reworded = _feat(REWORDED)
        cls.reordered = _feat(REORDERED)
        cls.excerpt = _feat(EXCERPT)
        cls.topic = _feat(TOPIC_SIMILAR)
        cls.distractor = _feat(DISTRACTORS[0])

    def test_identical_is_one(self):
        self.assertAlmostEqual(
            score_features(self.base, _feat(BASE))["score"], 1.0)

    def test_synonym_rewording_is_duplicate(self):
        score = score_features(self.base, self.reworded)["score"]
        self.assertGreaterEqual(score, DEFAULT_THRESHOLD)

    def test_reordering_is_duplicate(self):
        score = score_features(self.base, self.reordered)["score"]
        self.assertGreaterEqual(score, 0.85)

    def test_excerpt_is_duplicate(self):
        score = score_features(self.base, self.excerpt)["score"]
        self.assertGreaterEqual(score, DEFAULT_THRESHOLD)

    def test_topic_similar_is_not_duplicate(self):
        score = score_features(self.base, self.topic)["score"]
        self.assertLess(score, DEFAULT_THRESHOLD)

    def test_unrelated_is_low(self):
        score = score_features(self.base, self.distractor)["score"]
        self.assertLess(score, 0.3)

    def test_symmetric_and_deterministic(self):
        s1 = score_features(self.base, self.reworded)["score"]
        s2 = score_features(self.reworded, self.base)["score"]
        s3 = score_features(self.base, self.reworded)["score"]
        self.assertAlmostEqual(s1, s2)
        self.assertEqual(s1, s3)

    def test_empty_text_scores_zero(self):
        empty = _feat("")
        self.assertEqual(score_features(empty, self.base)["score"], 0.0)
        self.assertEqual(score_features(empty, empty)["score"], 0.0)


class TestCandidatesAndClustering(unittest.TestCase):
    def test_lsh_recalls_duplicates(self):
        docs = {"base": BASE, "reworded": REWORDED, "reordered": REORDERED,
                "excerpt": EXCERPT, "topic": TOPIC_SIMILAR}
        docs.update({f"d{i}": t for i, t in enumerate(DISTRACTORS)})
        signatures = {k: _feat(t).signature for k, t in docs.items()}
        pairs = candidate_pairs(signatures)
        for dup in ("reworded", "reordered", "excerpt"):
            self.assertTrue(any(dup in pair and "base" in pair for pair in pairs),
                            f"LSH 漏掉了 base~{dup} 候选对")

    def test_all_pairs_count(self):
        self.assertEqual(len(all_pairs(["a", "b", "c"])), 3)

    def test_cluster_star_groups(self):
        # a-b(0.9)、b-c(0.8) -> 以度数最高的 b 为中心的星形组
        groups = cluster([("a", "b", 0.9), ("b", "c", 0.8)])
        self.assertEqual(groups, [("b", ["a", "b", "c"])])

    def test_cluster_breaks_transitive_chains(self):
        # A~B 0.9、C~D 0.9、B~C 0.65：单链会串成 {A,B,C,D}，
        # 星形聚类应得到 {A,B} 与 {C,D} 两组（B-C 边两端已各自入组则跳过）
        groups = cluster([("a", "b", 0.9), ("c", "d", 0.9), ("b", "c", 0.65)])
        self.assertEqual(groups, [("b", ["a", "b"]), ("c", ["c", "d"])])

    def test_cluster_multi_excerpt_shares_source(self):
        # 同一原文的两段互不重叠的摘录：彼此不相似但都与原文相似，
        # 应聚到以原文为中心的一组（单链/全连接都会给出错误结果）
        groups = cluster([("src", "x1", 0.8), ("src", "x2", 0.8)])
        self.assertEqual(groups, [("src", ["x1", "x2", "src"])])

    def test_cluster_ignores_singletons(self):
        self.assertEqual(cluster([]), [])


# ---------------------------------------------------------------------------
# 服务层
# ---------------------------------------------------------------------------

class TestSimilarityService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # 分片大小设为 5，10 篇文档 -> 2 个分片，验证跨分片汇总
        self.registry = StoreRegistry(self.tmp, shard_size=5)
        self.service = SimilarityService(self.registry)
        self.corpus = self.registry.task("corpus")
        self.ids = {}
        for key, text in [("base", BASE), ("reworded", REWORDED),
                          ("reordered", REORDERED), ("excerpt", EXCERPT),
                          ("topic", TOPIC_SIMILAR)]:
            rid = self.corpus.insert({"name": key, "text": text,
                                      "created_at": 1.0})
            self.service.on_corpus_added(rid, text)
            self.ids[key] = rid
        for i, text in enumerate(DISTRACTORS):
            rid = self.corpus.insert({"name": f"d{i}", "text": text,
                                      "created_at": 1.0})
            self.service.on_corpus_added(rid, text)
            self.ids[f"d{i}"] = rid

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # -- 索引 -------------------------------------------------------------
    def test_index_built_incrementally(self):
        stats = self.service.ensure_index()
        self.assertEqual(stats["added"], 0)  # 钩子已建，无需补建
        self.assertEqual(self.registry.task("simindex").stats()["total"], 10)

    def test_index_self_heals(self):
        # 绕过服务直接删语料 -> ensure_index 自动清理索引
        rid = self.ids["d0"]
        self.corpus.delete(rid)
        stats = self.service.ensure_index()
        self.assertEqual(stats["removed"], 1)
        self.assertNotIn(rid, self.service._index_features())

    # -- 检索 -------------------------------------------------------------
    def test_search_ranks_duplicates_first(self):
        result = self.service.search(text=REWORDED, limit=10)
        top_ids = [r["corpus_id"] for r in result["results"][:4]]
        for key in ("base", "reordered", "excerpt"):
            self.assertIn(self.ids[key], top_ids)
        scores = {r["corpus_id"]: r["score"] for r in result["results"]}
        self.assertGreaterEqual(scores[self.ids["base"]], DEFAULT_THRESHOLD)
        self.assertLess(scores[self.ids["topic"]], DEFAULT_THRESHOLD)

    def test_search_by_corpus_id_excludes_self(self):
        result = self.service.search(corpus_id=self.ids["base"], limit=10)
        returned = [r["corpus_id"] for r in result["results"]]
        self.assertNotIn(self.ids["base"], returned)
        self.assertIn(self.ids["reworded"], returned[:3])

    def test_search_pagination_is_continuous(self):
        limit = 3
        seen, pages = [], 0
        offset = 0
        while True:
            result = self.service.search(text=BASE, limit=limit, offset=offset)
            seen.extend(r["corpus_id"] for r in result["results"])
            pages += 1
            offset += limit
            if offset >= result["total"]:
                break
            self.assertLess(pages, 10)
        self.assertEqual(len(seen), len(set(seen)), "翻页出现重复")
        self.assertEqual(set(seen), set(self.ids.values()), "翻页有遗漏")

    def test_search_min_score_filter(self):
        result = self.service.search(text=BASE, limit=10, min_score=0.6)
        self.assertTrue(all(r["score"] >= 0.6 for r in result["results"]))
        self.assertGreaterEqual(result["total"], 3)

    def test_metric_consistent_after_corpus_changes(self):
        before = self.service.search(text=REWORDED, limit=1)["results"][0]["score"]
        # 新增无关文档后，同一对文档的得分不变（口径与语料规模无关）
        rid = self.corpus.insert({"name": "extra", "text": DISTRACTORS[0] + "补充",
                                  "created_at": 1.0})
        self.service.on_corpus_added(rid, DISTRACTORS[0] + "补充")
        after = self.service.search(text=REWORDED, limit=1)["results"][0]["score"]
        self.assertEqual(before, after)

    # -- 查重 -------------------------------------------------------------
    def test_dedup_groups_duplicates(self):
        stats = self.service.run_dedup()
        self.assertEqual(stats["groups"], 1)
        groups = self.service.groups()["groups"]
        members = set(groups[0]["members"][i]["corpus_id"]
                      for i in range(groups[0]["size"]))
        expect = {self.ids[k] for k in ("base", "reworded", "reordered", "excerpt")}
        self.assertEqual(members, expect)

    def test_dedup_gid_stable_across_changes(self):
        self.service.run_dedup()
        gid_before = self.service.groups()["groups"][0]["gid"]

        # 新增无关文档 -> 重跑 -> 组 id 与成员不变
        rid = self.corpus.insert({"name": "extra", "text": DISTRACTORS[1],
                                  "created_at": 1.0})
        self.service.on_corpus_added(rid, DISTRACTORS[1])
        self.service.run_dedup()
        groups = self.service.groups()["groups"]
        self.assertEqual(groups[0]["gid"], gid_before)
        self.assertEqual(groups[0]["size"], 4)

        # 删除组内一篇 -> 组 id 不变，成员减一
        self.corpus.delete(self.ids["excerpt"])
        self.service.on_corpus_deleted(self.ids["excerpt"])
        self.service.run_dedup()
        groups = self.service.groups()["groups"]
        self.assertEqual(groups[0]["gid"], gid_before)
        self.assertEqual(groups[0]["size"], 3)

    def test_delete_hook_updates_groups(self):
        self.service.run_dedup()
        self.corpus.delete(self.ids["excerpt"])
        self.service.on_corpus_deleted(self.ids["excerpt"])
        groups = self.service.groups()["groups"]
        self.assertEqual(groups[0]["size"], 3)
        # 删到不足两篇 -> 组解散
        for key in ("reworded", "reordered", "base"):
            self.corpus.delete(self.ids[key])
            self.service.on_corpus_deleted(self.ids[key])
        self.assertEqual(self.service.groups()["groups"], [])

    def test_groups_stale_flag(self):
        self.service.run_dedup()
        self.assertFalse(self.service.groups()["stale"])
        rid = self.corpus.insert({"name": "new", "text": DISTRACTORS[2],
                                  "created_at": 1.0})
        self.service.on_corpus_added(rid, DISTRACTORS[2])
        self.assertTrue(self.service.groups()["stale"])
        self.service.run_dedup()
        self.assertFalse(self.service.groups()["stale"])

    def test_dedup_scales_with_lsh(self):
        """超过全配对阈值后走 LSH 路径，且能找回植入的重复对。"""
        import random
        rng = random.Random(7)
        templates = DISTRACTORS + [TOPIC_SIMILAR]
        for i in range(160):
            text = templates[i % len(templates)] + f"补充说明第{i}条。"
            # 随机插入噪声句，避免大面积人造重复
            if i % 3 == 0:
                text += rng.choice(DISTRACTORS)
            rid = self.corpus.insert({"name": f"bulk{i}", "text": text,
                                      "created_at": 1.0})
            self.service.on_corpus_added(rid, text)
        stats = self.service.run_dedup()
        self.assertEqual(stats["candidate_mode"], "lsh")
        groups = self.service.groups()["groups"]
        base_group = [g for g in groups
                      if any(m["corpus_id"] == self.ids["base"]
                             for m in g["members"])]
        self.assertTrue(base_group, "LSH 路径下未找回原有重复组")


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class TestSimilarityAPI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        from app import create_app
        self.app = create_app(self.tmp)
        self.client = self.app.test_client()
        self.ids = {}
        for key, text in [("base", BASE), ("reworded", REWORDED),
                          ("topic", TOPIC_SIMILAR)]:
            resp = self.client.post("/api/corpus",
                                    json={"name": key, "text": text})
            self.assertEqual(resp.status_code, 200)
            self.ids[key] = resp.get_json()["id"]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_search_endpoint(self):
        # 用库内文档做查询（自动排除自身），最相似的应是其改写版
        resp = self.client.post("/api/similarity/search",
                                json={"corpus_id": self.ids["reworded"],
                                      "limit": 5})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["results"][0]["corpus_id"], self.ids["base"])
        self.assertGreaterEqual(data["results"][0]["score"], DEFAULT_THRESHOLD)

    def test_search_pagination_via_api(self):
        r1 = self.client.post("/api/similarity/search",
                              json={"text": BASE, "limit": 2, "offset": 0})
        r2 = self.client.post("/api/similarity/search",
                              json={"text": BASE, "limit": 2, "offset": 2})
        ids1 = [r["corpus_id"] for r in r1.get_json()["results"]]
        ids2 = [r["corpus_id"] for r in r2.get_json()["results"]]
        self.assertFalse(set(ids1) & set(ids2), "翻页出现重复")
        self.assertEqual(len(ids1) + len(ids2), r1.get_json()["total"])

    def test_search_requires_input(self):
        resp = self.client.post("/api/similarity/search", json={})
        self.assertEqual(resp.status_code, 400)

    def test_dedup_endpoints(self):
        resp = self.client.post("/api/dedup/run", json={"threshold": 0.6})
        self.assertEqual(resp.status_code, 200)
        stats = resp.get_json()
        self.assertEqual(stats["groups"], 1)
        resp = self.client.get("/api/dedup/groups")
        data = resp.get_json()
        self.assertFalse(data["stale"])
        members = {m["corpus_id"] for m in data["groups"][0]["members"]}
        self.assertEqual(members, {self.ids["base"], self.ids["reworded"]})

    def test_delete_corpus_updates_groups(self):
        self.client.post("/api/dedup/run", json={})
        self.client.delete(f"/api/corpus/{self.ids['reworded']}")
        data = self.client.get("/api/dedup/groups").get_json()
        self.assertEqual(data["groups"], [])

    def test_reindex_endpoint(self):
        resp = self.client.post("/api/similarity/reindex")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["total"], 3)


if __name__ == "__main__":
    unittest.main()
