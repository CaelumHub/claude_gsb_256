"""机器翻译（模拟）。

基于小型平行词典的短语/词级翻译，属于「规则 + 词典」的演示实现，
并非神经机器翻译。支持 中文 -> 英文 与 英文 -> 中文 双向。

实现要点：
- 用分词器切分中文，按最长匹配优先查短语表，再查词表；
- 英文按空格分词，逐词查反向词典；
- 附少量结构规则（如时间/地点状语后置、英文大小写与冠词）。
"""

from __future__ import annotations

import re
from typing import Optional

from .segmenter import Segmenter


# 平行词典：中文 -> 英文（词与常见短语）
ZH_EN = {
    # 常用词
    "我": "I", "你": "you", "他": "he", "她": "she", "它": "it",
    "我们": "we", "你们": "you", "他们": "they",
    "是": "is", "有": "have", "在": "in", "和": "and", "的": "of",
    "喜欢": "like", "爱": "love", "想": "want", "说": "say", "看": "see",
    "学习": "learn", "工作": "work", "生活": "life", "研究": "research",
    "开发": "develop", "使用": "use", "实现": "implement", "提供": "provide",
    "支持": "support", "帮助": "help", "需要": "need", "可以": "can",
    "今天": "today", "明天": "tomorrow", "昨天": "yesterday", "现在": "now",
    "未来": "future", "时间": "time", "世界": "world", "国家": "country",
    "中国": "China", "美国": "America", "日本": "Japan", "北京": "Beijing",
    "上海": "Shanghai", "公司": "company", "大学": "university",
    "学校": "school", "老师": "teacher", "学生": "student", "朋友": "friend",
    "家庭": "family", "孩子": "child", "人": "people", "人类": "human",
    "社会": "society", "经济": "economy", "科技": "technology",
    "技术": "technology", "计算机": "computer", "互联网": "internet",
    "人工智能": "artificial intelligence", "数据": "data", "信息": "information",
    "系统": "system", "平台": "platform", "网络": "network", "软件": "software",
    "硬件": "hardware", "程序": "program", "算法": "algorithm", "模型": "model",
    "语言": "language", "自然语言": "natural language", "文本": "text",
    "新闻": "news", "文章": "article", "问题": "problem", "方法": "method",
    "结果": "result", "过程": "process", "目标": "goal", "功能": "function",
    "性能": "performance", "质量": "quality", "速度": "speed", "效率": "efficiency",
    "成本": "cost", "价格": "price", "市场": "market", "产品": "product",
    "服务": "service", "用户": "user", "客户": "customer", "设计": "design",
    "管理": "manage", "分析": "analyze", "教育": "education", "健康": "health",
    "环境": "environment", "资源": "resource", "能源": "energy", "金融": "finance",
    "银行": "bank", "投资": "investment", "股票": "stock", "基金": "fund",
    # 常用短语
    "你好": "hello", "谢谢": "thank you", "再见": "goodbye", "对不起": "sorry",
    "早上好": "good morning", "晚上好": "good evening",
    "机器学习": "machine learning", "深度学习": "deep learning",
    "神经网络": "neural network", "大数据": "big data",
    "云计算": "cloud computing", "物联网": "internet of things",
    "我喜欢": "I like", "非常好": "very good", "很好": "very good",
    "很大": "very big", "很重要": "very important",
    # 动词/形容词
    "好": "good", "坏": "bad", "大": "big", "小": "small", "多": "many",
    "少": "few", "高": "high", "低": "low", "长": "long", "短": "short",
    "新": "new", "旧": "old", "快": "fast", "慢": "slow", "重要": "important",
    "简单": "simple", "复杂": "complex", "正确": "correct", "错误": "wrong",
    "准确": "accurate", "稳定": "stable", "安全": "safe", "快速": "fast",
    "高效": "efficient", "智能": "intelligent", "自动": "automatic",
    # 数量/时间
    "一": "one", "二": "two", "三": "three", "四": "four", "五": "five",
    "六": "six", "七": "seven", "八": "eight", "九": "nine", "十": "ten",
    "年": "year", "月": "month", "日": "day", "小时": "hour", "分钟": "minute",
}

# 反向词典：英文 -> 中文（自动构建 + 少量手工补充）
EN_ZH: dict[str, str] = {}
for _zh, _en in ZH_EN.items():
    EN_ZH.setdefault(_en.lower(), _zh)


class Translator:
    """模拟翻译器。"""

    def __init__(self, segmenter: Optional[Segmenter] = None):
        self.segmenter = segmenter or Segmenter()
        # 按长度降序，保证最长匹配优先
        self._zh_phrases = sorted(ZH_EN.keys(), key=len, reverse=True)

    def translate(self, text: str, direction: str = "zh2en") -> dict:
        direction = direction.lower()
        if direction in ("zh2en", "zh-en", "zh"):
            target = self._zh_to_en(text)
            source_lang, target_lang = "zh", "en"
        else:
            target = self._en_to_zh(text)
            source_lang, target_lang = "en", "zh"
        return {
            "source": text,
            "translation": target,
            "direction": f"{source_lang}->{target_lang}",
            "engine": "dictionary-based (simulated)",
        }

    # -- 中文 -> 英文 -----------------------------------------------------
    def _zh_to_en(self, text: str) -> str:
        words = self.segmenter.cut(text)
        out = []
        i = 0
        n = len(words)
        # 先做最长短语匹配（跨多个分词结果合并成短语），未命中再逐词翻译
        while i < n:
            matched_len = 0
            for length in range(min(4, n - i), 0, -1):
                phrase = "".join(words[i:i + length])
                if phrase in ZH_EN:
                    out.append(ZH_EN[phrase])
                    matched_len = length
                    break
            if matched_len:
                i += matched_len
            else:
                w = words[i]
                if w not in "，。！？、；：\"\"''（）《》【】,.!?;: \n":
                    out.append(ZH_EN.get(w, w))
                i += 1
        translation = " ".join(out)
        return self._postprocess_en(translation)

    @staticmethod
    def _postprocess_en(s: str) -> str:
        # 句首大写，句尾加句号
        s = s.strip()
        if not s:
            return s
        s = s[0].upper() + s[1:]
        if s and s[-1] not in ".!?":
            s += "."
        return s

    # -- 英文 -> 中文 -----------------------------------------------------
    def _en_to_zh(self, text: str) -> str:
        tokens = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?|[.,!?]", text)
        out = []
        for tok in tokens:
            low = tok.lower().strip(".,!?")
            if low in EN_ZH:
                out.append(EN_ZH[low])
            elif tok in ".,!?":
                out.append(tok)
            else:
                out.append(tok)
        return "".join(out)
