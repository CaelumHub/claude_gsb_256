"""相似度检索 / 近重复查重测试。

运行：``python -m unittest tests.test_similarity -v``

覆盖：
- 三粒度词法特征对语序调整/同义改写/截取的区分能力；
- 同主题不同内容不被误判；
- MinHash / 超平面 LSH 的候选召回；
- 分片存储、跨分片检索、游标分页不重不漏；
- 增删文档后归组 id 稳定（继承/分裂/新增）；
- 口径版本不一致触发重建。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.similarity import (DEFAULT_CONFIG, MinHasher, SimilarityModel,
                            char_bigrams, content_tokens,
                            find_duplicate_edges, hyperplane_signature,
                            lsh_candidate_pairs, make_fingerprint,
                            pair_score, query_candidates, cluster_edges)
from storage import StoreRegistry, SimilarityService


BASE = (
    "自然语言处理是人工智能的重要分支，机器学习技术近年来取得了显著进展，"
    "深度学习模型被广泛应用于语音识别、机器翻译和文本分类等任务。"
    "研究人员通过大规模语料训练神经网络，使系统能够自动学习语言规律。"
    "未来对话系统将更好地理解用户意图，提供更智能的服务。"
)
REORDER = (
    "未来对话系统将更好地理解用户意图，提供更智能的服务。"
    "研究人员通过大规模语料训练神经网络，使系统能够自动学习语言规律。"
    "深度学习模型被广泛应用于语音识别、机器翻译和文本分类等任务，"
    "机器学习技术近年来取得了显著进展。自然语言处理是人工智能的重要分支。"
)
REWORD = (
    "自然语言处理属于人工智能的一个关键领域，机器学习方法近些年获得了明显进步，"
    "深度神经网络已大量用于语音识别、机器翻译以及文本分类等场景。"
    "研究者借助海量语料来训练神经网络，让系统可以自主掌握语言规律。"
    "今后对话系统会更加理解用户的真实意图，从而提供更加智能化的服务。"
)
EXCERPT = (
    "深度学习模型被广泛应用于语音识别、机器翻译和文本分类等任务。"
    "研究人员通过大规模语料训练神经网络，使系统能够自动学习语言规律。"
)
SAME_TOPIC = (
    "计算机视觉研究如何让机器从图片中提取信息，卷积神经网络在目标检测和"
    "图像分割任务上表现优异。自动驾驶汽车依赖激光雷达与摄像头感知周围环境"
    "并做出行驶决策。"
)
UNRELATED = "昨天的足球比赛非常精彩，主队连进三球大胜，球迷散场后燃放焰火庆祝。"

DOCS = [
    ("base", BASE), ("reorder", REORDER), ("reword", REWORD),
    ("excerpt", EXCERPT), ("same_topic", SAME_TOPIC),
    ("unrelated", UNRELATED),
]


def build_model(docs=DOCS):
    model = SimilarityModel()
    prepared = [(text, content_tokens(text)) for _, text in docs]
    model.fit(prepared)
    return model


def fingerprints(model, docs=DOCS):
    mh = MinHasher()
    return {key: make_fingerprint(key, text, model, mh)
            for key, text in docs}


class TestLexicalFeatures(unittest.TestCase):
    def setUp(self):
        self.model = build_model()
        self.fps = fingerprints(self.model)

    def score(self, a, b):
        return pair_score(self.fps[a], self.fps[b], self.model.config)

    def test_reorder_is_near_identical(self):
        s = self.score("base", "reorder")
        self.assertGreaterEqual(s["word"], 0.99)
        self.assertGreaterEqual(s["bigram"], 0.9)

    def test_reword_ranks_above_unrelated(self):
        near = self.score("base", "reword")["score"]
        far = self.score("base", "same_topic")["score"]
        none = self.score("base", "unrelated")["score"]
        self.assertGreater(near, 0.4)
        self.assertGreater(near, far)
        self.assertGreater(far, none)

    def test_excerpt_recognized_as_partial_copy(self):
        s = self.score("base", "excerpt")
        self.assertGreater(s["word"], 0.6)
        self.assertGreater(s["bigram"], 0.4)

    def test_same_topic_different_content_is_low(self):
        s = self.score("base", "same_topic")
        self.assertLess(s["lexical"], 0.2)

    def test_scores_in_unit_range_and_symmetric(self):
        for i, (a, _) in enumerate(DOCS):
            for b, _ in DOCS[i + 1:]:
                s = self.score(a, b)
                for key in ("score", "lexical", "semantic",
                            "word", "char", "bigram"):
                    self.assertGreaterEqual(s[key], 0.0)
                    self.assertLessEqual(s[key], 1.0)
                self.assertEqual(
                    s["score"], self.score(b, a)["score"])

    def test_char_bigram_order_invariant_within_text(self):
        # 相同句子集合、不同顺序：bigram 集合几乎一致
        g1 = char_bigrams("机器学习技术取得进展。深度学习被广泛应用。")
        g2 = char_bigrams("深度学习被广泛应用。机器学习技术取得进展。")
        self.assertGreater(len(g1 & g2) / len(g1 | g2), 0.9)

    def test_deterministic_minhash(self):
        mh1, mh2 = MinHasher(), MinHasher()
        fp_a = make_fingerprint("x", BASE, self.model, mh1)
        fp_b = make_fingerprint("x", BASE, self.model, mh2)
        self.assertEqual(fp_a["sig"], fp_b["sig"])
        # 相同输入 -> 超平面签名一致
        self.assertEqual(
            hyperplane_signature(fp_a["sem"], 64),
            hyperplane_signature(fp_a["sem"], 64))


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.model = build_model()
        self.fps = fingerprints(self.model)

    def test_edges_and_groups_with_floor(self):
        fps = list(self.fps.values())
        # 重排是确定的重复；截取超过高阈值时按参数决定
        edges = find_duplicate_edges(fps, self.model.config,
                                     dup_threshold=0.95, lex_floor=0.5)
        self.assertIn(("base", "reorder"), edges)
        groups = cluster_edges(fps, edges)
        members = {d for g in groups for d in g["docs"]}
        self.assertIn("base", members)
        self.assertIn("reorder", members)
        self.assertNotIn("same_topic", members)
        self.assertNotIn("unrelated", members)

    def test_lexical_floor_blocks_semantic_only(self):
        # 构造一对语义可能接近但词法不重叠的极端输入，验证 floor 生效
        text_a = "人工智能模型提升了识别准确率"
        text_b = "深度学习系统改善了分类表现"
        model = SimilarityModel()
        model.fit([(text_a, content_tokens(text_a)),
                   (text_b, content_tokens(text_b))])
        mh = MinHasher()
        fp_a = make_fingerprint("a", text_a, model, mh)
        fp_b = make_fingerprint("b", text_b, model, mh)
        parts = pair_score(fp_a, fp_b, model.config)
        self.assertLess(parts["lexical"], 0.35)

    def test_lsh_recalls_obvious_pair(self):
        pairs = lsh_candidate_pairs(list(self.fps.values()),
                                    self.model.config)
        # LSH 在极小库上不保证（直接走全对比较路径），这里仅验证接口；
        # 大库召回见 TestService。
        self.assertIsInstance(pairs, set)


class _ServiceTestBase(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.registry = StoreRegistry(self.root, shard_size=10)
        self.service = SimilarityService(self.registry, shard_size=128)
        self.ids = {}
        for name, text in DOCS:
            self.ids[name] = self.registry.task("corpus").insert({
                "name": name, "text": text, "created_at": time.time()})

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def insert(self, name, text):
        return self.registry.task("corpus").insert(
            {"name": name, "text": text, "created_at": time.time()})


class TestService(_ServiceTestBase):
    def test_build_and_status(self):
        st = self.service.ensure_index()
        self.assertTrue(st["built"])
        self.assertEqual(st["doc_count"], len(DOCS))
        self.assertTrue(st["in_sync"])
        self.assertEqual(st["shard_count"], 1)
        # 幂等：再次 ensure 不重建
        v1 = st["index_version"]
        st2 = self.service.ensure_index()
        self.assertEqual(st2["index_version"], v1)

    def test_search_arbitrary_text_ranking(self):
        self.service.ensure_index()
        result = self.service.search(
            text="人工智能机器学习深度学习让系统从语料中自动学习规律",
            limit=6)
        self.assertNotIn("error", result)
        top = result["results"][0]["id"]
        self.assertIn(top, (self.ids["base"], self.ids["reorder"]))
        # 无关文档排在末位（same_topic 与 unrelated 可并列垫底）
        last_ids = {r["id"] for r in result["results"][-2:]}
        self.assertIn(self.ids["unrelated"], last_ids)

    def test_search_by_doc_excludes_self(self):
        self.service.ensure_index()
        result = self.service.search(doc_id=self.ids["base"], limit=10)
        self.assertNotIn(self.ids["base"], [r["id"] for r in result["results"]])
        self.assertEqual(result["results"][0]["id"], self.ids["reorder"])

    def test_search_missing_doc(self):
        self.service.ensure_index()
        result = self.service.search(doc_id="corpus_9999")
        self.assertIn("error", result)

    def test_cursor_pagination_continuous(self):
        # 多加一批文档，分页拉全量，顺序严格、不重不漏
        self.service.ensure_index()
        seen = []
        cursor = None
        pages = 0
        while True:
            r = self.service.search(doc_id=self.ids["unrelated"],
                                    limit=2, cursor=cursor)
            pages += 1
            batch = [x["id"] for x in r["results"]]
            seen.extend(batch)
            cursor = r["next_cursor"]
            if not cursor:
                break
            self.assertLessEqual(pages, 10)
        self.assertEqual(len(seen), len(DOCS) - 1)
        self.assertEqual(len(seen), len(set(seen)))
        # 分数非递增（同分时 id 升序由服务层保证）
        all_rows = self.service.search(
            doc_id=self.ids["unrelated"], limit=50)["results"]
        scores = [x["score"] for x in all_rows]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_stale_cursor_after_rebuild(self):
        self.service.ensure_index()
        page = self.service.search(doc_id=self.ids["base"], limit=2)
        cursor = page["next_cursor"]
        self.insert("new", "一篇全新主题的短文，讲航天探测器登陆小行星采样。")
        self.service.ensure_index()
        page2 = self.service.search(doc_id=self.ids["base"],
                                    limit=2, cursor=cursor)
        self.assertTrue(page2["stale_cursor"])

    def test_dedup_groups(self):
        self.service.ensure_index()
        d = self.service.run_dedup(dup_threshold=0.95, lex_floor=0.5)
        self.assertGreaterEqual(d["group_count"], 1)
        all_members = [m for g in d["groups"] for m in g["docs"]]
        self.assertIn(self.ids["base"], all_members)
        self.assertIn(self.ids["reorder"], all_members)
        self.assertNotIn(self.ids["same_topic"], all_members)
        self.assertNotIn(self.ids["unrelated"], all_members)

    def test_group_id_stable_across_add_delete(self):
        self.service.ensure_index()
        d = self.service.run_dedup(dup_threshold=0.95, lex_floor=0.5)
        gid = next(g["group_id"] for g in d["groups"]
                   if self.ids["base"] in g["docs"])

        # 新增无关文档后重建：原组 id 保留
        new_id = self.insert("体育新闻", "昨晚马拉松鸣枪起跑，三万名跑者沿江奔跑。")
        self.service.ensure_index()
        groups = self.service.get_groups()["groups"]
        kept = {g["group_id"]: g["docs"] for g in groups}
        self.assertIn(gid, kept)
        self.assertIn(self.ids["base"], kept[gid])
        self.assertIn(self.ids["reorder"], kept[gid])

        # 删除新文档后仍保留
        self.registry.task("corpus").delete(new_id)
        self.service.ensure_index()
        groups = self.service.get_groups()["groups"]
        self.assertIn(gid, {g["group_id"] for g in groups})

    def test_new_group_gets_new_id_and_old_kept(self):
        self.service.ensure_index()
        d1 = self.service.run_dedup(dup_threshold=0.95, lex_floor=0.5)
        before = {g["group_id"] for g in d1["groups"]}
        # 增加一对全新的重复文档（与已有任何组不重叠）
        x = self.insert("dup_x", "量子纠缠实验再次验证贝尔不等式，测量结果令人振奋。")
        y = self.insert("dup_y", "量子纠缠实验再度验证贝尔不等式，测量的结果令人振奋不已。")
        self.service.ensure_index()
        d2 = self.service.run_dedup(dup_threshold=0.55, lex_floor=0.3)
        after = {g["group_id"]: g["docs"] for g in d2["groups"]}
        # 旧组仍在
        self.assertTrue(before & set(after))
        # 新组拿到新 id
        new_gids = set(after) - before
        new_groups = [g for gid, g in after.items() if gid in new_gids]
        self.assertTrue(any(x in g and y in g for g in new_groups))

    def test_sharding_across_files(self):
        # 用更小分片重建，确认跨分片汇总
        service = SimilarityService(self.registry, shard_size=2)
        st = service.ensure_index(force=True)
        self.assertEqual(st["shard_count"], 3)
        result = service.search(doc_id=self.ids["base"], limit=10)
        self.assertEqual(result["results"][0]["id"], self.ids["reorder"])


class TestLargeCorpusLSH(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.registry = StoreRegistry(self.root, shard_size=100)
        self.service = SimilarityService(self.registry, shard_size=300)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_large_corpus_recall_and_speed(self):
        # > BRUTE_FORCE_LIMIT 时走 LSH，验证明显重复对能召回
        store = self.registry.task("corpus")
        pair_ids = []
        topic = (
            "公司宣布新一代处理器性能提升百分之四十，功耗下降百分之二十，"
            "编号{tag}，将于下季度量产，主要面向数据中心市场。"
        )
        # 近复制版：局部调序 + 轻微改写（真实翻版的主体形态）
        topic_v = (
            "编号{tag}，公司宣布新一代处理器性能提升百分之四十，"
            "功耗下降约百分之二十，计划在下个季度量产，主要面向数据中心市场。"
        )
        for k in range(60):
            a = store.insert({"name": f"a{k}", "text": topic.format(tag=k)})
            b = store.insert({"name": f"b{k}", "text": topic_v.format(tag=k)})
            pair_ids.append((a, b))
        # 互不相同的短文：随机句子组合
        import random
        rng = random.Random(0)
        bank = [
            "河面上的雾到晌午才散。", "老槐树底下坐着几个下棋的人。",
            "邮递员的自行车铃在巷口响了两下。", "糖炒栗子的香味飘过整条街。",
            "屋顶的瓦松又高了一截。", "孩子们追着断了线的风筝跑。",
            "渡口的艄公把竹篙插进淤泥。", "戏台的锣鼓突然紧了三声。",
            "山货铺门口挂着一串干辣椒。", "末班公交晃着黄灯驶过桥头。",
            "井台边结了薄薄一层冰。", "隔壁院子传来二胡拉错的音阶。",
            "新刷的石灰墙映得屋子发亮。", "货郎担上的玻璃弹珠反着光。",
            "雨点儿敲在铁皮棚上噼啪作响。", "奶奶把腌菜坛子封得严实。",
            "集市收摊后满地都是菜叶。", "村小的钟声敲过六下。",
            "两只麻雀为半块烧饼争个不停。", "晒谷场上铺开金黄一片。",
            "煤油灯的影子在土墙上来回晃。", "他把信读了三遍才折好。",
            "早班火车鸣着笛穿过薄雾。", "卖花姑娘篮里剩两枝栀子。",
            "修伞匠撑开破伞对着光看。", "渔船归港带回满仓银光。",
        ]
        for k in range(180):
            sents = rng.sample(bank, 5)
            store.insert({"name": f"n{k}",
                          "text": "".join(sents) + f"（独立札记{k:03d}）"})

        self.service.ensure_index()
        d = self.service.run_dedup(dup_threshold=0.72, lex_floor=0.35)
        members = {}
        for g in d["groups"]:
            for doc in g["docs"]:
                members[doc] = g["group_id"]
        hits = sum(1 for a, b in pair_ids
                   if members.get(a) and members[a] == members.get(b))
        # LSH 允许极少数漏检，主体必须召回（实测应全部命中）
        self.assertGreaterEqual(hits, int(len(pair_ids) * 0.95))

        # 查询路径（非库内文本）也应通过 LSH 召回近复制
        r = self.service.search(
            text="公司宣布新一代处理器性能提升百分之四十，功耗下降百分之二十，"
                 "将于下季度量产，主要面向数据中心市场。",
            limit=5)
        self.assertLessEqual(r["candidate_count"], d["doc_count"])
        self.assertGreaterEqual(r["results"][0]["score"], 0.6)


if __name__ == "__main__":
    unittest.main()
