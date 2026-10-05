/* ============================================================
   公共脚本：导航注入、API 封装、工具函数
   ============================================================ */

const PAGES = [
  { file: "corpus.html",    name: "语料库管理",   desc: "上传与清洗" },
  { file: "segment.html",   name: "分词与词性标注", desc: "切词 + POS" },
  { file: "parse.html",     name: "句法分析树",   desc: "依存 / 成分树" },
  { file: "ner.html",       name: "命名实体识别", desc: "NER 与标注" },
  { file: "sentiment.html", name: "情感分析",     desc: "正负面分类" },
  { file: "summary.html",   name: "文本摘要",     desc: "抽取式摘要" },
  { file: "translate.html", name: "机器翻译",     desc: "模拟翻译" },
  { file: "keywords.html",  name: "关键词提取",   desc: "TF-IDF + TextRank" },
  { file: "embedding.html", name: "词向量可视化", desc: "降维投影" },
  { file: "pipeline.html",  name: "流水线配置",   desc: "编排与执行" },
];

const PAGE_NAMES = {
  corpus: "语料库管理", segment: "分词与词性标注", parse: "句法分析树",
  ner: "命名实体识别", sentiment: "情感分析", summary: "文本摘要",
  translate: "机器翻译", keywords: "关键词提取", embedding: "词向量可视化",
  pipeline: "流水线配置与执行",
};

// 中文标签集（与后端 /api/meta 一致，离线可用）
const TAG_NAMES = {
  n: "名词", v: "动词", a: "形容词", d: "副词", r: "代词", p: "介词",
  c: "连词", u: "助词", m: "数词", q: "量词", t: "时间词", f: "方位词",
  ns: "地名", nr: "人名", nt: "机构", x: "其他",
};

const ENTITY_NAMES = {
  PERSON: "人名", LOCATION: "地名", ORGANIZATION: "机构", TIME: "时间",
  DATE: "日期", NUMBER: "数字", MONEY: "金额", PERCENT: "百分比",
};

const DEP_REL_NAMES = {
  root: "核心", nsubj: "主语", dobj: "宾语", amod: "定语修饰", nummod: "数量修饰",
  advmod: "状语修饰", conj: "并列", mark: "助词", case: "介词", clf: "量词",
  lobj: "方位", tmod: "时间修饰", compound: "复合词", punct: "标点", attr: "补语",
  dep: "其它",
};

const PHRASE_NAMES = {
  S: "句子", NP: "名词短语", VP: "动词短语", AP: "形容词短语", PP: "介词短语",
  ADVP: "副词短语", P: "介词", CONJ: "连词", U: "助词", TP: "时间短语",
};

/* ---------- 导航注入 ---------- */
function renderNav(activeFile) {
  const host = document.getElementById("site-sidebar");
  if (!host) return;
  let links = PAGES.map((p, i) => {
    const cls = p.file === activeFile ? ' class="active"' : "";
    return `<a href="/page/${p.file}"${cls}><span class="idx">${String(i + 1).padStart(2, "0")}</span>${p.name}</a>`;
  }).join("");
  host.innerHTML = `
    <div class="brand">NLP 流水线平台<small>自然语言处理</small></div>
    <nav>${links}</nav>`;
}

/* ---------- API 封装 ---------- */
async function api(path, options = {}) {
  const opts = { headers: {}, ...options };
  if (opts.body && typeof opts.body === "object" && !(opts.body instanceof FormData)) {
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(opts.body);
  }
  const resp = await fetch(path, opts);
  let data = null;
  try { data = await resp.json(); } catch (_) { /* ignore */ }
  if (!resp.ok) {
    const msg = (data && data.error) || `请求失败 (${resp.status})`;
    throw new Error(msg);
  }
  return data;
}

/* ---------- 工具 ---------- */
function $(sel, root = document) { return root.querySelector(sel); }
function $$(sel, root = document) { return Array.from(root.querySelectorAll(sel)); }

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function toast(msg, type = "ok") {
  let el = $(".toast");
  if (!el) {
    el = document.createElement("div");
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = msg;
  el.className = `toast ${type}`;
  requestAnimationFrame(() => el.classList.add("show"));
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove("show"), 2800);
}

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", { hour12: false });
}

function setLoading(btn, on, text = "处理中…") {
  if (!btn) return;
  if (on) {
    btn.dataset.label = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = `<span class="spinner"></span>${text}`;
  } else {
    btn.disabled = false;
    btn.innerHTML = btn.dataset.label || btn.innerHTML;
  }
}

/* 把带实体偏移的文本高亮显示 */
function highlightEntities(text, entities) {
  if (!entities || !entities.length) return esc(text);
  const sorted = entities.slice().sort((a, b) => a.start - b.start);
  let html = "", last = 0;
  for (const e of sorted) {
    html += esc(text.slice(last, e.start));
    html += `<span class="ent-${e.type}" title="${ENTITY_NAMES[e.type] || e.type}">${esc(text.slice(e.start, e.end))}</span>`;
    last = e.end;
  }
  html += esc(text.slice(last));
  return html;
}

/* 词性 token 渲染 */
function renderTokens(tokens) {
  return tokens.map(([w, t]) =>
    `<span class="token"><span class="t-word">${esc(w)}</span><span class="pos-tag pos-${t}">${t} ${TAG_NAMES[t] || ""}</span></span>`
  ).join("");
}

/* 页面元数据自动填充标题 */
function renderPageTitle(key, desc) {
  const t = $(".page-title");
  if (t && PAGE_NAMES[key]) t.textContent = PAGE_NAMES[key];
  const d = $(".page-desc");
  if (d && desc) d.textContent = desc;
}
