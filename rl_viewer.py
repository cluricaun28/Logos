#!/usr/bin/env python3
"""
RL Live Viewer — read-only Wikipedia-style browser for a Logos Reference Library.
No build step: renders markdown off-disk on request, so it is never stale.
Cross-linking (wikilinks + backlinks) is the primary feature. Search reads the
same v2 hybrid index the agents use (~/.hermes/rl_index.db / rl_index_fts),
with legacy search.db/rl_fts and in-memory title matching as fallbacks, so
human + agent views agree.

Per-user: point RL_ROOT at one user's reference-library; run one instance per user.
Read-only: no write endpoints.
"""
import os, re, time, html, sqlite3, sys, threading, math
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from markdown import markdown
import yaml

# Accept both RL_ROOT (original) and RLV_ROOT (systemd unit convention used
# by the per-user fleet units) — RLV_ROOT wins when both are set.
RL_ROOT = Path(os.environ.get("RLV_ROOT") or os.environ.get("RL_ROOT", str(Path.home() / ".logos" / "reference-library"))).resolve()
SEARCH_DB = RL_ROOT / "search.db"
PORT = int(os.environ.get("RLV_PORT", "8090"))
USER_LABEL = os.environ.get("RLV_USER", "RL")

# ---------------------------------------------------------------- link index
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def _top_cat(key):
    """Top-level category of a no-.md page key."""
    parts = key.split("/")
    return parts[0] if len(parts) > 1 else "(root)"

def _index_maps(md_files, root):
    """Pass 1: resolution maps + files + display titles + category counts."""
    path_index = {}    # "system/foo" -> "system/foo.md"
    name_index = {}    # "foo" (stem)  -> "system/foo.md"  (first wins)
    title_index = {}   # slug of display title -> path
    files = {}         # relpath(without .md) -> relpath(with .md)
    titles = {}        # key(no .md) -> display title (for title-substring fallback)
    cat_counts = {}
    for p in md_files:
        rel = p.relative_to(root).as_posix()
        key = rel[:-3]  # strip .md
        files[key] = rel
        path_index.setdefault(key, rel)
        stem = Path(rel).stem
        name_index.setdefault(stem, rel)
        parts = p.relative_to(root).parts
        top = parts[0] if len(parts) > 1 else "(root)"
        cat_counts[top] = cat_counts.get(top, 0) + 1
        # frontmatter title
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            txt = ""
        disp = stem.replace("-", " ").title()
        m = re.match(r"^---\s*\n(.*?)\n---", txt, re.S)
        if m:
            try:
                fm = yaml.safe_load(m.group(1)) or {}
                if isinstance(fm, dict) and fm.get("title"):
                    title_index.setdefault(_slug(str(fm["title"])), rel)
                    disp = str(fm["title"])
            except Exception:
                pass
        titles[key] = disp
    return path_index, name_index, title_index, files, cat_counts, titles

def _scan_links(md_files, path_index, name_index, title_index, root):
    """Pass 2: wikilink scan -> (backlinks, out_edges, cat_matrix).

    Single source of truth for link resolution: backlinks and out_edges are
    built from the SAME resolved edges, so the graph views can never disagree
    with the per-page '⬅ Backlinks (N)' section. cat_matrix counts link
    INSTANCES per (src_top_cat, dst_top_cat) pair."""
    backlinks = {}     # target key(no .md) -> {source keys}
    out_edges = {}     # source key(no .md) -> {target keys}
    cat_matrix = {}    # (src_top, dst_top) -> instance count
    for p in md_files:
        rel = p.relative_to(root).as_posix()
        src = rel[:-3]
        src_top = _top_cat(src)
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        for m in WIKILINK_RE.finditer(txt):
            target = m.group(1).split("|")[0].strip()
            resolved = _resolve_target(target, path_index, name_index, title_index)
            if resolved:
                dst = resolved[:-3]
                backlinks.setdefault(dst, set()).add(src)  # key by no-.md path
                out_edges.setdefault(src, set()).add(dst)
                k = (src_top, _top_cat(dst))
                cat_matrix[k] = cat_matrix.get(k, 0) + 1
    return backlinks, out_edges, cat_matrix

def build_index():
    """Scan all .md files -> resolution maps + link maps + home data.

    Returns 10-tuple: (path_index, name_index, title_index, files, backlinks,
    cat_counts, recent, titles, out_edges, cat_matrix). cat_counts/recent are
    computed HERE (background thread) so the home page never rglobs in the
    request path."""
    md_files = [p for p in RL_ROOT.rglob("*.md")
                if p.is_file() and "static" not in p.parts and "assets" not in p.parts]
    path_index, name_index, title_index, files, cat_counts, titles = _index_maps(md_files, RL_ROOT)
    backlinks, out_edges, cat_matrix = _scan_links(md_files, path_index, name_index, title_index, RL_ROOT)
    # recent updates (top 14 by mtime) — computed here, not per home request
    recent = []
    for p in md_files:
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.relative_to(RL_ROOT).as_posix()
        key = rel[:-3]
        recent.append((key, titles.get(key, p.stem.replace("-", " ").title()),
                       st.st_mtime,
                       p.parent.relative_to(RL_ROOT).as_posix() or "root"))
    recent.sort(key=lambda r: r[2], reverse=True)
    recent = recent[:14]
    return (path_index, name_index, title_index, files, backlinks,
            cat_counts, recent, titles, out_edges, cat_matrix)

def _resolve_target(target, path_index, name_index, title_index):
    t = target.strip()
    if not t:
        return None
    if "/" in t:
        cand = t
        if cand in path_index: return path_index[cand]
        if (cand + ".md") in path_index: return path_index[cand + ".md"]
        return None
    if t in name_index: return name_index[t]
    if _slug(t) in title_index: return title_index[_slug(t)]
    return None

IDX = build_index()
(PATH_INDEX, NAME_INDEX, TITLE_INDEX, FILES, BACKLINKS, CAT_COUNTS, RECENT, FILES_TITLES,
 OUT_EDGES, CAT_MATRIX) = IDX
_last_scan = time.time()

def reload_if_stale(force=False):
    global IDX, PATH_INDEX, NAME_INDEX, TITLE_INDEX, FILES, BACKLINKS, CAT_COUNTS, RECENT, FILES_TITLES, OUT_EDGES, CAT_MATRIX, _last_scan, _INDEX_GEN
    # Refresh normally happens in _index_loop (background, every 120s).
    # Synchronous rebuild is a rare safety net (loop died / >5min stale).
    if force or time.time() - _last_scan > 300:
        IDX = build_index()
        PATH_INDEX, NAME_INDEX, TITLE_INDEX, FILES, BACKLINKS, CAT_COUNTS, RECENT, FILES_TITLES, OUT_EDGES, CAT_MATRIX = IDX
        _last_scan = time.time()
        _INDEX_GEN += 1

def _index_loop():
    """Background: keep the link/home index fresh without blocking requests."""
    global IDX, PATH_INDEX, NAME_INDEX, TITLE_INDEX, FILES, BACKLINKS, CAT_COUNTS, RECENT, FILES_TITLES, OUT_EDGES, CAT_MATRIX, _last_scan, _INDEX_GEN
    while True:
        time.sleep(120)
        try:
            IDX = build_index()
            PATH_INDEX, NAME_INDEX, TITLE_INDEX, FILES, BACKLINKS, CAT_COUNTS, RECENT, FILES_TITLES, OUT_EDGES, CAT_MATRIX = IDX
            _last_scan = time.time()
            _INDEX_GEN += 1
        except Exception:
            pass

threading.Thread(target=_index_loop, daemon=True).start()

app = FastAPI()

# ---------------------------------------------------------------- rendering
CSS = """
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2128;--fg:#e6edf3;--dim:#8b949e;
--acc:#58a6ff;--acc2:#3fb950;--miss:#d29922;--line:#30363d;--code:#79c0ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
a{color:var(--acc);text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:920px;margin:0 auto;padding:24px 20px 80px}
header.top{position:sticky;top:0;background:rgba(13,17,23,.95);backdrop-filter:blur(6px);
border-bottom:1px solid var(--line);z-index:10}
.bar{max-width:920px;margin:0 auto;display:flex;gap:14px;align-items:center;padding:12px 20px}
.brand{font-weight:700;font-size:18px;white-space:nowrap}
.brand .user{color:var(--acc2)}
.search{flex:1;display:flex}
.search input{width:100%;background:var(--panel);border:1px solid var(--line);
color:var(--fg);border-radius:8px;padding:8px 12px;font-size:15px}
.search input:focus{outline:none;border-color:var(--acc)}
h1{font-size:26px;margin:8px 0 4px}
h2{font-size:20px;margin:24px 0 8px;border-bottom:1px solid var(--line);padding-bottom:6px}
h3{font-size:17px;margin:20px 0 6px}
.meta{color:var(--dim);font-size:13px;margin:2px 0 14px}
.meta code{color:var(--acc);background:var(--panel);padding:1px 6px;border-radius:5px}
.crumbs{font-size:13px;color:var(--dim);margin-bottom:12px}
.crumbs a{color:var(--dim)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin:10px 0}
table{border-collapse:collapse;width:100%;margin:12px 0;font-size:14px}
th,td{border:1px solid var(--line);padding:7px 10px;text-align:left}
th{background:var(--panel2)}
code{color:var(--code);background:var(--panel);padding:1px 5px;border-radius:5px;font-size:.92em}
pre{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow:auto}
pre code{background:none;padding:0}
blockquote{border-left:3px solid var(--acc);margin:12px 0;padding:4px 14px;color:var(--dim);background:var(--panel)}
.wikilink.missing{color:var(--miss);border-bottom:1px dashed var(--miss);cursor:help}
.backlinks{margin-top:28px;padding-top:14px;border-top:2px solid var(--line)}
.backlinks h2{color:var(--acc2);margin-top:0}
.backlinks ul{margin:6px 0 0;padding-left:20px}
.catgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:10px;margin:14px 0}
.catgrid a{display:block;background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:12px 14px;font-weight:600}
.catgrid a:hover{border-color:var(--acc);text-decoration:none}
.catgrid .n{display:block;color:var(--dim);font-weight:400;font-size:12px;margin-top:4px}
ol.plist,ul.plist{list-style:none;padding:0}
ol.plist li,ul.plist li{padding:6px 0;border-bottom:1px solid var(--line)}
ol.plist li a{font-weight:600}
.tag{display:inline-block;background:var(--panel2);border:1px solid var(--line);
border-radius:20px;padding:1px 9px;font-size:12px;color:var(--dim);margin:0 4px 4px 0}
.snip{color:var(--dim);font-size:13px;margin:2px 0 0}
.snip b{color:var(--fg)}
.banner{border-color:var(--acc2)}
.banner a{font-weight:700;font-size:16px}
.pnav{display:flex;gap:14px;align-items:center;margin:6px 0 0;color:var(--dim);font-size:13px}
.pnav a{font-weight:600}
.badge{font-size:11px;color:var(--dim);border:1px solid var(--line);border-radius:20px;padding:0 8px}
footer{color:var(--dim);font-size:12px;margin-top:40px;text-align:center}
"""

def render_wikilinks(md_text):
    """Convert [[...]] to HTML links, resolving against the live index."""
    def repl(m):
        inner = m.group(1)
        target, label = (inner.split("|", 1) + [None])[:2]
        target = target.strip()
        label = (label or target).strip()
        resolved = _resolve_target(target, PATH_INDEX, NAME_INDEX, TITLE_INDEX)
        if resolved:
            key = resolved[:-3] if resolved.endswith(".md") else resolved
            disp = html.escape(label)
            return f'<a class="wikilink" href="/page/{key}">{disp}</a>'
        return (f'<a class="wikilink missing" title="page not found yet">'
                f'{html.escape(label)}</a>')
    return WIKILINK_RE.sub(repl, md_text)

def _frontmatter(text):
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.S)
    fm = {}
    if m:
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except Exception:
            fm = {}
        body = text[m.end():]
    else:
        body = text
    return (fm if isinstance(fm, dict) else {}), body

def render_page(path_key):
    reload_if_stale()
    rel = FILES.get(path_key)
    if not rel:
        return _404(path_key)
    f = RL_ROOT / rel
    try:
        text = f.read_text(errors="ignore")
    except Exception as e:
        return _404(path_key, str(e))
    fm, body = _frontmatter(text)
    body = render_wikilinks(body)
    html_body = markdown(body, extensions=["tables", "fenced_code", "nl2br", "toc"])
    title = fm.get("title") or Path(rel).stem.replace("-", " ").title()
    # metadata line
    meta_bits = []
    if fm.get("type"): meta_bits.append(f"type: {fm['type']}")
    if fm.get("category"): meta_bits.append(f"category: {fm['category']}")
    if fm.get("last_updated") or fm.get("updated"):
        meta_bits.append(f"updated: {fm.get('last_updated') or fm.get('updated')}")
    meta_bits.append(f"<code>{html.escape(rel)}</code>")
    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime))
    meta_bits.append(f"file mtime: {mtime}")
    meta = " &nbsp;·&nbsp; ".join(meta_bits)
    # description
    desc = fm.get("description") or ""
    # backlinks
    bl = sorted(p for p in BACKLINKS.get(path_key, []) if p != path_key)
    bl_html = ""
    if bl:
        items = "".join(
            f'<li><a href="/page/{b}">{html.escape(Path(b).stem.replace("-"," ").title())}</a></li>'
            for b in bl[:40])
        more = f'<li class="snip">+{len(bl)-40} more</li>' if len(bl) > 40 else ""
        bl_html = f'<div class="backlinks"><h2>⬅ Backlinks ({len(bl)})</h2><ul>{items}{more}</ul></div>'
    crumbs = " / ".join(Path(rel).parts[:-1]) or "root"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(title))} · {html.escape(USER_LABEL)} RL</title>
<style>{CSS}</style></head><body>
<header class="top"><div class="bar">
<span class="brand">{html.escape(USER_LABEL)} <span class="user">Reference Library</span></span>
<form class="search" action="/search" method="get"><input name="q" placeholder="Search this library…" value=""></form>
</div></header>
<div class="wrap">
<div class="crumbs"><a href="/">home</a> / {html.escape(crumbs)} / <a href="/page/{path_key}">{html.escape(Path(rel).stem)}</a></div>
<h1>{html.escape(str(title))}</h1>
<div class="meta">{meta}</div>
{'<p class="snip">'+html.escape(str(desc))+'</p>' if desc else ''}
<div class="card" style="padding:20px 22px">{html_body}</div>
{bl_html}
<footer>RL Live Viewer · rendered live from {html.escape(str(RL_ROOT))} · {mtime}</footer>
</div></body></html>"""

def _404(key, extra=""):
    return HTMLResponse(f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}</style></head>
<body><div class="wrap"><h1>Page not found</h1>
<p class="snip">{html.escape(str(key))} {html.escape(extra)}</p>
<a href="/">← back to home</a></div></body></html>""", status_code=404)

# ---------------------------------------------------------------- home / browse
CATEGORIES = ["people","organizations","technology","events","places","projects",
              "methods","reasoning-methods","topics","ideas","law","history",
              "evidence","skills","tools","system","books","library","sources",
              "communication-delivery"]

@app.get("/", response_class=HTMLResponse)
def home():
    reload_if_stale()
    # category counts + recents are precomputed in build_index() (refreshed by
    # the background _index_loop) — no per-request disk scans here.
    cards = "".join(
        f'<a href="/browse/{name}">{html.escape(name.replace("-"," ").title())}<span class="n">{n} pages</span></a>'
        for name, n in sorted(CAT_COUNTS.items(), key=lambda x: -x[1])
        if n and name != "(root)")
    recent = "".join(
        f'<li><a href="/page/{key}">{html.escape(title)}</a>'
        f'<div class="snip">{time.strftime("%b %d %H:%M", time.localtime(mtime))} · {html.escape(parent)}</div></li>'
        for key, title, mtime, parent in RECENT)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(USER_LABEL)} Reference Library</title>
<style>{CSS}</style></head><body>
<header class="top"><div class="bar">
<span class="brand">{html.escape(USER_LABEL)} <span class="user">Reference Library</span></span>
<form class="search" action="/search" method="get"><input name="q" placeholder="Search {len(FILES):,} pages…"></form>
</div></header>
<div class="wrap">
<h1>{html.escape(USER_LABEL)}'s Reference Library</h1>
<div class="meta">{len(FILES):,} pages · {len(BACKLINKS):,} cross-linked · live, no build step</div>
<h2>Browse by category</h2>
<div class="catgrid">{cards}</div>
<h2>Recently updated</h2>
<div class="card"><ul class="plist">{recent}</ul></div>
<footer>RL Live Viewer · rendered live from {html.escape(str(RL_ROOT))}</footer>
</div></body></html>"""

@app.get("/browse/{category:path}", response_class=HTMLResponse)
def browse(category: str, request: Request):
    """Folder-style browse: subdirectories at this level (drill down) + files at
    THIS level only (paginated), so big trees (e.g. a large jobs tree, 6k pages)
    navigate instead of dumping one wall of links. Titles come from the
    in-memory index — no per-file disk reads in the request path. A folder with
    an index.md/INDEX.md gets a 'start here' banner pointing at it."""
    reload_if_stale()
    cat = category.strip("/")
    d = RL_ROOT / cat
    if not d.is_dir():
        return _404(category)
    # subdirectories (skip static/assets/dotfiles) with subtree page counts
    subdirs = []
    for p in sorted(d.iterdir(), key=lambda x: x.name.lower()):
        if p.is_dir() and not p.name.startswith(".") and p.name not in ("static", "assets"):
            n = sum(1 for f in p.rglob("*.md") if f.is_file())
            if n:
                subdirs.append((p.name, n))
    # files at this level only — titles from the in-memory index
    files = []
    for p in d.iterdir():
        if p.is_file() and p.suffix == ".md":
            key = f"{cat}/{p.name}"[:-3]
            files.append((FILES_TITLES.get(key) or p.stem.replace("-", " ").title(), key))
    files.sort(key=lambda x: x[0].lower())
    # 'start here' banner: index/INDEX.md in this folder
    banner = next((f"{cat}/{c}" for c in ("INDEX", "index") if f"{cat}/{c}" in FILES), None)
    # pagination (files at this level; subdirs always listed in full)
    PAGE = 400
    pages = max(1, (len(files) + PAGE - 1) // PAGE)
    try:
        pnum = int(request.query_params.get("p", "1"))
    except (ValueError, TypeError):
        pnum = 1
    pnum = max(1, min(pnum, pages))
    chunk = files[(pnum - 1) * PAGE : pnum * PAGE]

    crumbs = '<a href="/">home</a>'
    parts = cat.split("/")
    for i in range(len(parts) - 1):
        crumbs += f' / <a href="/browse/{"/".join(parts[:i+1])}">{html.escape(parts[i])}</a>'
    crumbs += f" / {html.escape(parts[-1])}"

    subs_html = ""
    if subdirs:
        cards = "".join(
            f'<a href="/browse/{cat}/{name}">{html.escape(name.replace("-", " ").title())}'
            f'<span class="n">{n:,} pages</span></a>' for name, n in subdirs)
        subs_html = f'<h2>Folders</h2><div class="catgrid">{cards}</div>'

    banner_html = ""
    if banner:
        bt = html.escape(FILES_TITLES.get(banner) or Path(banner).stem.replace("-", " ").title())
        banner_html = (f'<div class="card banner"><a href="/page/{banner}">📖 Start here: {bt}</a>'
                       f'<div class="snip">index page for this folder</div></div>')

    items = "".join(f'<li><a href="/page/{k}">{html.escape(t)}</a></li>' for t, k in chunk)
    pnav = ""
    if pages > 1:
        prev = f'<a href="?p={pnum - 1}">← Prev</a>' if pnum > 1 else ""
        nxt = f'<a href="?p={pnum + 1}">Next →</a>' if pnum < pages else ""
        pnav = (f'<div class="pnav">{prev} <span>page {pnum} of {pages} · '
                f'{len(files):,} files</span> {nxt}</div>')

    total = len(files) + sum(n for _, n in subdirs)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style>
<title>{html.escape(cat.replace("-", " ").title())} · {html.escape(USER_LABEL)} RL</title></head><body>
<header class="top"><div class="bar">
<span class="brand">{html.escape(USER_LABEL)} <span class="user">RL</span></span>
<form class="search" action="/search" method="get"><input name="q" placeholder="Search…"></form>
</div></header><div class="wrap">
<div class="crumbs">{crumbs}</div>
<h1>{html.escape(cat.replace("-", " ").title())} <span class="badge">{total:,} pages</span></h1>
{banner_html}{subs_html}
<h2>Files <span class="badge">{len(files):,}</span></h2>
<div class="card"><ul class="plist">{items or '<li class="snip">no files in this folder</li>'}</ul></div>
{pnav}
<footer>RL Live Viewer</footer></div></body></html>"""

@app.get("/page/{path:path}", response_class=HTMLResponse)
def page(path: str):
    return render_page(path)

# ---------------------------------------------------------------- search
# The v2 hybrid index (rl_index.db — the same one the agents query via
# reference_library_search) is the primary source. Legacy search.db/rl_fts
# (per-page FTS, frozen 2026-08-16) is a fallback if the v2 DB is missing.
SEARCH_DB = RL_ROOT / "search.db"                      # legacy v1
SEARCH_DB_V2 = RL_ROOT.parent / "rl_index.db"          # v2, agent-shared

_SEARCH_STOP = {"the","a","an","of","on","for","to","in","and","or","what",
                "is","are","was","were","right","now","where","which","how",
                "do","does","did","with","about","at","by","it","its",
                "this","that","i","my","our","you","your"}

def _fts_tokens(q: str):
    toks = [t for t in re.split(r"\s+", q) if t]
    cleaned = [t for t in toks if len(t) >= 2 and t.lower() not in _SEARCH_STOP]
    return cleaned or toks

def _q(tok: str) -> str:
    return '"' + tok.replace('"', '""') + '"'

def _fts_search_v2(con, q: str, toks):
    """Hybrid-index search: AND -> OR -> de-pluralized, title-boosted ranking."""
    attempts = [
        " AND ".join(_q(t) for t in toks),
        " OR ".join(_q(t) for t in toks),
    ]
    # de-pluralize pass: gpus->gpu, cards->card (helps unicode61, no stemming)
    dep = [t[:-1] if t.lower().endswith("s") and len(t) > 3 else t for t in toks]
    if dep != toks:
        attempts += [
            " AND ".join(_q(t) for t in dep),
            " OR ".join(_q(t) for t in dep),
        ]
    sql = ("SELECT file_path, title, snippet(rl_index_fts, 2, '<b>', '</b>', '…', 24) "
           "FROM rl_index_fts WHERE rl_index_fts MATCH ? "
           "ORDER BY bm25(rl_index_fts, 0.1, 3.0, 1.0, 0.1) LIMIT 25")
    for attempt in attempts:
        try:
            rows = con.execute(sql, (attempt,)).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            continue
    return []

def _fts_search_v1(con, q: str, toks):
    """Legacy per-page index (search.db/rl_fts) — columns (path, title, ...)."""
    attempts = [
        " AND ".join(_q(t) for t in toks),
        " OR ".join(_q(t) for t in toks),
    ]
    sql = ("SELECT path, title, snippet(rl_fts, 2, '<b>', '</b>', '…', 24) "
           "FROM rl_fts WHERE rl_fts MATCH ? ORDER BY rank LIMIT 25")
    for attempt in attempts:
        try:
            rows = con.execute(sql, (attempt,)).fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            continue
    return []

def _title_search(q: str, toks):
    """In-memory last resort: display-title substring (no DB needed)."""
    needle = q.lower().strip()
    hits = []
    for key, title in FILES_TITLES.items():
        tl = title.lower()
        if needle and needle in tl:
            hits.append((key + ".md", title, f"matched in <b>title</b>"))
        else:
            for t in toks:
                if t.lower() in tl:
                    hits.append((key + ".md", title, f"matched in <b>title</b>"))
                    break
        if len(hits) >= 25:
            break
    return hits

@app.get("/search", response_class=HTMLResponse)
def search(request: Request):
    q = (request.query_params.get("q") or "").strip()
    results = []
    total_pages = len(FILES)
    toks = _fts_tokens(q)
    if q:
        rows = []
        # v2 (agent-shared) index first
        if SEARCH_DB_V2.exists():
            try:
                con = sqlite3.connect(f"file:{SEARCH_DB_V2}?mode=ro", uri=True)
                try:
                    n_docs = con.execute("SELECT COUNT(*) FROM rl_index_fts_docsize").fetchone()[0]
                    if n_docs >= 100:  # v2 index is populated
                        rows = _fts_search_v2(con, q, toks)
                finally:
                    con.close()
            except Exception:
                rows = []
        # legacy fallback (v2 missing/empty)
        if not rows and SEARCH_DB.exists():
            try:
                con = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
                try:
                    rows = _fts_search_v1(con, q, toks)
                finally:
                    con.close()
            except Exception as e:
                rows = [(f"(search error: {e})", "", "")]
        # in-memory title fallback
        if not rows:
            rows = _title_search(q, toks)
        results = rows
    items = ""
    for path, title, snip in results:
        key = path[:-3] if path.endswith(".md") else path
        key = key.lstrip("/")
        exists = key in FILES
        link = f'<a href="/page/{key}">' if exists else '<span class="wikilink missing" title="indexed but file missing">'
        t = title or Path(key).stem.replace("-"," ").title()
        cls = "" if exists else ' style="opacity:.6"'
        items += (f'<div class="card"{cls}><a href="/page/{key}" style="font-weight:700">{html.escape(str(t))}</a> '
                  f'<div class="snip"><code>{html.escape(str(path))}</code></div>'
                  f'<div class="snip">{snip}</div></div>')
    if not q:
        items = '<p class="snip">Type a search term above.</p>'
    elif not results:
        items = f'<p class="snip">No results for “{html.escape(q)}”.</p>'
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}</style>
<title>Search: {html.escape(q)} · {html.escape(USER_LABEL)} RL</title></head><body>
<header class="top"><div class="bar">
<span class="brand">{html.escape(USER_LABEL)} <span class="user">RL</span></span>
<form class="search" action="/search" method="get"><input name="q" placeholder="Search…" value="{html.escape(q)}"></form>
</div></header><div class="wrap">
<h1>Search <span class="badge">{len(results)} results</span></h1>
{items}
<footer>searching the live FTS5 index (same as the agents) · {total_pages:,} pages</footer>
</div></body></html>"""

# ---------------------------------------------------------------- graph views
# Read-only views of the wikilink structure: /graph (core topology),
# /graph/ego (neighborhood), /graph/heat (category-pair matrix). Fully
# offline: inline SVG + one small inline <script> for pan/zoom, ZERO external
# references. Layout is server-computed and DETERMINISTIC (seed 42), and each
# page's HTML is cached per index generation (content is static between the
# background index rebuilds, and the seeded layout is stable across restarts).
# Edges come from the SAME resolved wikilinks as the per-page backlinks — the
# graph can never disagree with '⬅ Backlinks (N)'.

MEGA_TREES = ("religion/bible", "crenshaw/jobs")  # archive/britannica removed 10/19: tree relocated to FL, 8 stubs remain
GRAPH_NODE_CAP = 1500          # beyond this, drop degree-1 nodes (count is shown)
_INDEX_GEN = 0                 # bumped on every index rebuild (graph cache key)
_GRAPH_CACHE = {}              # cache key -> rendered HTML
_GRAPH_LOCK = threading.Lock()

def _mega_of(key):
    """Mega-tree prefix this no-.md key belongs to, or None."""
    for pre in MEGA_TREES:
        if key == pre or key.startswith(pre + "/"):
            return pre
    return None

def collapse_mega(node_set, edge_pairs, all_files):
    """Collapse each MEGA_TREES subtree into ONE labeled node.

    Returns (nodes, edges, mega_pages): nodes/edges are sorted lists of
    no-.md keys (internal mega-tree edges vanish into the node's link weight),
    mega_pages = {prefix: total pages in that tree} (from all_files)."""
    norm = {}
    for k in node_set:
        m = _mega_of(k)
        norm[k] = m if m else k
    mega_pages = {pre: sum(1 for k in all_files if _mega_of(k) == pre)
                  for pre in MEGA_TREES}
    pairs = set()
    for a, b in edge_pairs:
        na, nb = norm.get(a, a), norm.get(b, b)
        if na != nb:
            pairs.add((na, nb) if na < nb else (nb, na))
    nodes = sorted(set(norm.values()) | {p for p, n in mega_pages.items() if n})
    return nodes, sorted(pairs), mega_pages

def ego_graph(root_key, out_edges, backlinks, hops):
    """Neighborhood of root_key over EXPLICIT wikilinks (out-edges ∪ inverted
    backlinks — same edge set), undirected, BFS up to `hops` (1..4).

    Returns (node_set, edge_pairs); edge pairs are sorted 2-tuples whose both
    endpoints are inside node_set."""
    hops = max(1, min(int(hops), 4))
    adj = {}
    for s, dsts in out_edges.items():
        adj.setdefault(s, set()).update(dsts)
        for d in dsts:
            adj.setdefault(d, set()).add(s)
    seen = {root_key}
    frontier = {root_key}
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            for v in adj.get(u, ()):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    edges = set()
    for u in seen:
        for v in adj.get(u, ()):
            if v in seen and v != u:
                edges.add((u, v) if u < v else (v, u))
    return seen, edges

def graph_core(node_set, edge_pairs, all_files, max_nodes=GRAPH_NODE_CAP):
    """Shared pipeline for /graph and /graph/ego: mega-tree collapse, then —
    if the visible node count exceeds the cap — keep degree>=2 nodes + mega
    nodes and report the omitted count.

    Returns (nodes, edges, omitted, mega_pages, mega_links, deg) where
    mega_links[pre] = distinct edge pairs touching that tree (internal pairs
    count once) and deg is the degree in the FINAL (filtered) graph."""
    nodes, edges, mega_pages = collapse_mega(node_set, edge_pairs, all_files)
    mega_links = {pre: 0 for pre in MEGA_TREES}
    for a, b in edge_pairs:
        ma, mb = _mega_of(a), _mega_of(b)
        if ma or mb:
            mega_links[ma or mb] += 1
    omitted = 0
    if len(nodes) > max_nodes:
        deg0 = {}
        for a, b in edges:
            deg0[a] = deg0.get(a, 0) + 1
            deg0[b] = deg0.get(b, 0) + 1
        keep = [n for n in nodes if _mega_of(n) or deg0.get(n, 0) >= 2]
        keep_set = set(keep)
        edges = [(a, b) for a, b in edges if a in keep_set and b in keep_set]
        omitted = len(nodes) - len(keep)
        nodes = keep
    deg = {}
    for a, b in edges:
        deg[a] = deg.get(a, 0) + 1
        deg[b] = deg.get(b, 0) + 1
    return nodes, edges, omitted, mega_pages, mega_links, deg

def build_graph_data(root_dir):
    """Pure graph-data builder (tests run this on a temp fixture tree without
    starting the app): same two-pass scan + resolution as build_index(), over
    an arbitrary root directory.

    Returns {"out_edges", "backlinks", "cat_matrix", "files", "cat_counts"}."""
    root = Path(root_dir)
    md_files = [p for p in root.rglob("*.md")
                if p.is_file() and "static" not in p.parts and "assets" not in p.parts]
    path_index, name_index, title_index, files, cat_counts, titles = _index_maps(md_files, root)
    backlinks, out_edges, cat_matrix = _scan_links(md_files, path_index, name_index, title_index, root)
    return {"out_edges": out_edges, "backlinks": backlinks, "cat_matrix": cat_matrix,
            "files": files, "cat_counts": cat_counts}

def _fr_layout(nodes, edges, root_key=None, iterations=100):
    """Deterministic Fruchterman–Reingold layout (seed 42). Returns
    {key: (x, y)} rounded to 0.1 — byte-stable across calls and restarts for
    the same (nodes, edges). numpy is lazy-imported; without it falls back to
    a deterministic phyllotaxis circle (still offline, still stable)."""
    n = len(nodes)
    if n == 0:
        return {}
    order = sorted(nodes)
    idx = {k: i for i, k in enumerate(order)}
    GOLD = 2.399963229728653
    try:
        import numpy as np
    except ImportError:
        return {k: (round(300.0 * math.cos(GOLD * i), 1),
                    round(300.0 * math.sin(GOLD * i), 1))
                for i, k in enumerate(order)}
    rnd = np.random.RandomState(42)
    P = np.empty((n, 2))
    for i in range(n):
        r = 30.0 * math.sqrt(i + 1)
        a = GOLD * i
        P[i] = (r * math.cos(a) + rnd.uniform(-8.0, 8.0),
                r * math.sin(a) + rnd.uniform(-8.0, 8.0))
    i0 = idx.get(root_key)
    if i0 is not None:
        P[i0] = (0.0, 0.0)
    E = [(idx[a], idx[b]) for a, b in edges if a in idx and b in idx]
    k = 16.0
    temp = 120.0
    if E:
        ea = np.array([a for a, b in E], dtype=int)
        eb = np.array([b for a, b in E], dtype=int)
    for _ in range(iterations):
        d = P[:, None, :] - P[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", d, d)
        np.fill_diagonal(dist2, 1e18)
        disp = ((k * k / dist2)[:, :, None] * d).sum(axis=1)          # repulsion k²/d
        if E:                                                         # attraction d²/k
            de = P[ea] - P[eb]
            de_dist = np.sqrt((de * de).sum(axis=1)) + 1e-9
            fe = (de_dist * de_dist / k)[:, None] * de / de_dist[:, None]
            np.add.at(disp, ea, -fe)
            np.add.at(disp, eb, fe)
        dl = np.sqrt((disp * disp).sum(axis=1)) + 1e-9
        P += disp * np.minimum(1.0, temp / dl)[:, None]
        if i0 is not None:
            P[i0] = (0.0, 0.0)
        temp *= 0.96
    P -= P.mean(axis=0)
    mx = max(float(np.abs(P[:, 0]).max()), 1e-6)
    my = max(float(np.abs(P[:, 1]).max()), 1e-6)
    P *= min(620.0 / mx, 430.0 / my, 3.0)
    return {k: (round(float(x), 1), round(float(y), 1)) for k, (x, y) in zip(order, P)}

GRAPH_JS = """
(function(){
var svg=document.getElementById('gsvg');if(!svg)return;
var vp=svg.querySelector('#viewport');if(!vp)return;
var x=0,y=0,s=1;
function ap(){vp.setAttribute('transform','translate('+x+','+y+') scale('+s+')');}
svg.addEventListener('wheel',function(e){e.preventDefault();
var r=svg.getBoundingClientRect();
var mx=(e.clientX-r.left)*(1400/r.width);
var my=(e.clientY-r.top)*(900/r.height);
var ns=Math.min(12,Math.max(0.15,s*(e.deltaY<0?1.15:1/1.15)));
x=mx-(mx-x)*(ns/s);y=my-(my-y)*(ns/s);s=ns;ap();
},{passive:false});
var drag=null;
svg.addEventListener('mousedown',function(e){drag=[e.clientX,e.clientY,x,y];svg.style.cursor='grabbing';});
window.addEventListener('mousemove',function(e){if(!drag)return;
var r=svg.getBoundingClientRect();
x=drag[2]+(e.clientX-drag[0])*(1400/r.width);
y=drag[3]+(e.clientY-drag[1])*(900/r.height);ap();});
window.addEventListener('mouseup',function(){drag=null;svg.style.cursor='grab';});
})();
"""

def _graph_svg(nodes, edges, deg, pos, mega_pages, mega_links, root_key, links):
    """Render the node/edge set as inline SVG (dark theme, no external refs)."""
    elines = []
    for a, b in edges:
        if a not in pos or b not in pos:
            continue
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        elines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}"/>')
    edges_g = f'<g stroke="#30363d" stroke-opacity=".55" stroke-width=".8">{"".join(elines)}</g>'
    nitems = []
    for k in nodes:
        if k not in pos:
            continue
        x, y = pos[k]
        d = deg.get(k, 0)
        m = _mega_of(k)
        if m:
            r = min(26.0, 13.0 + 1.8 * math.log10(max(mega_pages.get(m, 1), 1)))
            fill = "#d29922"
            label = f"{m.split('/')[-1]} · {mega_pages.get(m, 0):,} pp · {mega_links.get(m, 0):,} lk"
            show = True
        else:
            r = min(14.0, 2.5 + 2.2 * math.log10(1 + d))
            fill = "#3fb950" if k == root_key else "#58a6ff"
            label = Path(k).stem.replace("-", " ").title()
            show = d >= 5
        if show:
            t = label if len(label) <= 30 else label[:29] + "…"
            tspan = (f'<text x="{x}" y="{y + r + 11}" text-anchor="middle" '
                     f'fill="#8b949e" font-size="10">{html.escape(t)}</text>')
        else:
            tspan = ""
        tip = html.escape(f"{k} · degree {d}")
        nitems.append(
            f'<a href="{links[k]}"><circle cx="{x}" cy="{y}" r="{r:.1f}" fill="{fill}" '
            f'stroke="#0d1117" stroke-width="1"><title>{tip}</title></circle>{tspan}</a>')
    return (f'<svg id="gsvg" viewBox="-700 -450 1400 900" xmlns="http://www.w3.org/2000/svg">'
            f'<g id="viewport">{edges_g}{"".join(nitems)}</g></svg>')

def _graph_page(title, header, svg_html, footer_note):
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<style>{CSS}
.gframe{{width:100%;height:calc(100vh - 190px);min-height:520px;background:var(--panel);
border:1px solid var(--line);border-radius:10px;overflow:hidden}}
.gframe svg{{width:100%;height:100%;display:block;cursor:grab}}
.gframe svg text{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
paint-order:stroke;stroke:#0d1117;stroke-width:2.5px;stroke-linejoin:round}}
.gframe a circle{{cursor:pointer}}
.legend{{color:var(--dim);font-size:12px;margin:10px 2px}}
.legend .sw{{display:inline-block;width:10px;height:10px;border-radius:50%;margin:0 4px 0 12px;vertical-align:-1px}}
</style></head><body>
<header class="top"><div class="bar">
<span class="brand">{html.escape(USER_LABEL)} <span class="user">Reference Library</span></span>
<form class="search" action="/search" method="get"><input name="q" placeholder="Search this library…" value=""></form>
</div></header>
<div class="wrap">
{header}
<div class="gframe">{svg_html}</div>
<div class="legend"><span class="sw" style="background:#58a6ff"></span>page
<span class="sw" style="background:#d29922"></span>mega-tree (collapsed)
<span class="sw" style="background:#3fb950"></span>ego root
&nbsp;·&nbsp; drag to pan · wheel to zoom · click a node to open its page</div>
<footer>RL Live Viewer · rendered live from {html.escape(str(RL_ROOT))} · {footer_note}</footer>
</div>
<script>{GRAPH_JS}</script>
</body></html>"""

def _render_graph_page(node_set, edge_pairs, root_key, title, header):
    """graph_core -> layout -> svg -> full page, cached per index generation."""
    global _INDEX_GEN
    ckey = ("ego" if root_key else "core", _INDEX_GEN, root_key)
    with _GRAPH_LOCK:
        cached = _GRAPH_CACHE.get(ckey)
    if cached:
        return HTMLResponse(cached)
    nodes, edges, omitted, mega_pages, mega_links, deg = graph_core(node_set, edge_pairs, FILES)
    pos = _fr_layout(nodes, edges, root_key=root_key)
    links = {k: (f"/browse/{_mega_of(k)}" if _mega_of(k) else f"/page/{k}") for k in nodes}
    svg = _graph_svg(nodes, edges, deg, pos, mega_pages, mega_links, root_key, links)
    page = _graph_page(title, header, svg, "graph view")
    with _GRAPH_LOCK:
        _GRAPH_CACHE[ckey] = page
    return HTMLResponse(page)

@app.get("/graph", response_class=HTMLResponse)
def graph():
    """Core topology: one node per page with degree >= 1, the three mega-trees
    collapsed to single labeled nodes; if the visible count exceeds the cap,
    degree-1 nodes are omitted (count stated in the header)."""
    reload_if_stale()
    node_set = {k for k in FILES if k in OUT_EDGES or k in BACKLINKS}
    edge_pairs = set()
    for s, dsts in OUT_EDGES.items():
        for d in dsts:
            if d != s:
                edge_pairs.add((s, d) if s < d else (d, s))
    nodes, edges, omitted, mega_pages, _, _ = graph_core(node_set, edge_pairs, FILES)
    omitted_note = (f' <span class="badge">{omitted:,} degree-1 nodes omitted '
                    f'(view them via /graph/ego)</span>' if omitted else "")
    mega_bits = " ".join(
        f'<span class="badge" style="color:var(--miss)">{m}: {mega_pages.get(m, 0):,} pages'
        f' · {sum(1 for a, b in edge_pairs if _mega_of(a) == m or _mega_of(b) == m):,} links</span>'
        for m in MEGA_TREES if mega_pages.get(m, 0))
    header = (f'<h1>Graph <span class="badge">{len(nodes):,} nodes · {len(edges):,} edges</span>'
              f'{omitted_note}</h1>'
              f'<div class="meta">deterministic server-computed layout (seed 42) · node size ∝ degree · '
              f'labels at degree ≥ 5 · offline, no external scripts {mega_bits}</div>')
    return _render_graph_page(node_set, edge_pairs, None, "Graph · " + USER_LABEL + " RL", header)

@app.get("/graph/ego", response_class=HTMLResponse)
def graph_ego(request: Request):
    """Neighborhood of one page over explicit wikilinks (in- AND out-edges),
    ?path=<page>&hops=N (default 2, cap 4)."""
    reload_if_stale()
    path = (request.query_params.get("path") or "").strip().lstrip("/")
    if path.endswith(".md"):
        path = path[:-3]
    if path not in FILES:
        return _404(f"graph/ego {path}")
    try:
        hops = int(request.query_params.get("hops", "2"))
    except (TypeError, ValueError):
        hops = 2
    hops = max(1, min(hops, 4))
    node_set, edge_pairs = ego_graph(path, OUT_EDGES, BACKLINKS, hops)
    node_set = node_set | {path}
    title = FILES_TITLES.get(path) or Path(path).stem.replace("-", " ").title()
    header = (f'<h1>Ego · {html.escape(str(title))} <span class="badge">{hops} hops</span></h1>'
              f'<div class="meta"><code>{html.escape(path)}</code> · '
              f'{len(BACKLINKS.get(path, ()))} in · {len(OUT_EDGES.get(path, ()))} out · '
              f'following explicit wikilinks both directions</div>')
    return _render_graph_page(node_set, edge_pairs, path, f"Ego: {title} · {USER_LABEL} RL", header)

@app.get("/graph/heat", response_class=HTMLResponse)
def graph_heat():
    """Category-pair edge matrix: server-rendered, color-shaded, no JS."""
    reload_if_stale()
    cats = sorted((c for c in CAT_COUNTS if CAT_COUNTS.get(c)),
                  key=lambda c: (-(sum(v for (s, d), v in CAT_MATRIX.items() if s == c or d == c)), c))
    max_c = max(CAT_MATRIX.values()) if CAT_MATRIX else 0
    log_max = math.log10(1 + max_c) if max_c else 1.0

    def cell(c):
        if not c:
            return ""
        a = 0.10 + 0.85 * (math.log10(1 + c) / log_max)
        return f' style="background:rgba(88,166,255,{a:.2f})"'

    col_hdr = "".join(f"<th>{html.escape(c)}</th>" for c in cats)
    rows = []
    for rc in cats:
        tds = "".join(
            f'<td{cell(CAT_MATRIX.get((rc, cc), 0))}>'
            f'{CAT_MATRIX.get((rc, cc), 0) or ""}</td>' for cc in cats)
        tot = sum(v for (s, d), v in CAT_MATRIX.items() if s == rc or d == rc)
        rows.append(f"<tr><th>{html.escape(rc)}</th>{tds}<td><b>{tot}</b></td></tr>")
    table = (f'<table style="font-size:12px"><tr><th></th>{col_hdr}<th>total</th></tr>'
             + "".join(rows) + "</table>")
    header = (f'<h1>Link Heatmap <span class="badge">{len(CAT_MATRIX)} active pairs · '
              f'{sum(CAT_MATRIX.values()):,} links</span></h1>'
              f'<div class="meta">rows = source top-level category · columns = destination · '
              f'count = resolved wikilink instances · darker = more links (log scale)</div>')
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Link Heatmap · {html.escape(USER_LABEL)} RL</title>
<style>{CSS}</style></head><body>
<header class="top"><div class="bar">
<span class="brand">{html.escape(USER_LABEL)} <span class="user">Reference Library</span></span>
<form class="search" action="/search" method="get"><input name="q" placeholder="Search this library…" value=""></form>
</div></header>
<div class="wrap" style="max-width:1500px">
{header}
<div class="card" style="overflow-x:auto;max-width:none">{table}</div>
<footer>RL Live Viewer · rendered live from {html.escape(str(RL_ROOT))} · graph view</footer>
</div></body></html>"""

if __name__ == "__main__":
    import uvicorn
    print(f"RL Live Viewer: {RL_ROOT} on :{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
