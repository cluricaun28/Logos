#!/usr/bin/env python3
"""
RL Live Viewer — read-only Wikipedia-style browser for a Logos Reference Library.
No build step: renders markdown off-disk on request, so it is never stale.
Cross-linking (wikilinks + backlinks) is the primary feature. Search reuses the
exact FTS5 index the agents use (search.db / rl_fts), so human + agent views agree.

Per-user: point RL_ROOT at one user's reference-library; run one instance per user.
Read-only: no write endpoints.
"""
import os, re, time, html, sqlite3, sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from markdown import markdown
import yaml

# Accept both RL_ROOT (original) and RLV_ROOT (systemd unit convention used
# by the per-user fleet units) — RLV_ROOT wins when both are set.
RL_ROOT = Path(os.environ.get("RLV_ROOT") or os.environ.get("RL_ROOT", "/home/exx/.hermes/reference-library")).resolve()
SEARCH_DB = RL_ROOT / "search.db"
PORT = int(os.environ.get("RLV_PORT", "8090"))
USER_LABEL = os.environ.get("RLV_USER", "RL")

# ---------------------------------------------------------------- link index
WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def build_index():
    """Scan all .md files -> resolution maps + backlink map."""
    path_index = {}    # "system/foo" -> "system/foo.md"
    name_index = {}    # "foo" (stem)  -> "system/foo.md"  (first wins)
    title_index = {}   # slug of display title -> path
    files = {}         # relpath(without .md) -> relpath(with .md)
    backlinks = {}     # target-path -> [source-paths]
    md_files = [p for p in RL_ROOT.rglob("*.md")
                if p.is_file() and "static" not in p.parts and "assets" not in p.parts]
    for p in md_files:
        rel = p.relative_to(RL_ROOT).as_posix()
        key = rel[:-3]  # strip .md
        files[key] = rel
        path_index.setdefault(key, rel)
        stem = Path(rel).stem
        name_index.setdefault(stem, rel)
        # frontmatter title
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            txt = ""
        m = re.match(r"^---\s*\n(.*?)\n---", txt, re.S)
        if m:
            try:
                fm = yaml.safe_load(m.group(1)) or {}
                if isinstance(fm, dict) and fm.get("title"):
                    title_index.setdefault(_slug(str(fm["title"])), rel)
            except Exception:
                pass
    # backlinks: scan every file for wikilinks
    for p in md_files:
        rel = p.relative_to(RL_ROOT).as_posix()
        src = rel[:-3]
        try:
            txt = p.read_text(errors="ignore")
        except Exception:
            continue
        for m in WIKILINK_RE.finditer(txt):
            target = m.group(1).split("|")[0].strip()
            resolved = _resolve_target(target, path_index, name_index, title_index)
            if resolved:
                backlinks.setdefault(resolved[:-3], set()).add(src)  # key by no-.md path
    return path_index, name_index, title_index, files, backlinks

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
PATH_INDEX, NAME_INDEX, TITLE_INDEX, FILES, BACKLINKS = IDX
_last_scan = time.time()

def reload_if_stale(force=False):
    global IDX, PATH_INDEX, NAME_INDEX, TITLE_INDEX, FILES, BACKLINKS, _last_scan
    # cheap staleness: rescan if forced or every 60s (keeps cross-links live)
    if force or time.time() - _last_scan > 60:
        IDX = build_index()
        PATH_INDEX, NAME_INDEX, TITLE_INDEX, FILES, BACKLINKS = IDX
        _last_scan = time.time()

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
    # count pages per category dir
    cats = []
    for d in sorted(p for p in RL_ROOT.iterdir() if p.is_dir() and p.name not in ("static","assets",".git")):
        n = sum(1 for _ in d.rglob("*.md"))
        if n:
            cats.append((d.name, n))
    loose = [p for p in RL_ROOT.glob("*.md")]
    cards = "".join(
        f'<a href="/browse/{name}">{html.escape(name.replace("-"," ").title())}<span class="n">{n} pages</span></a>'
        for name, n in sorted(cats, key=lambda x:-x[1]))
    # recent updates (top 12 by mtime)
    mdlist = []
    for p in RL_ROOT.rglob("*.md"):
        if p.is_file() and "static" not in p.parts:
            mdlist.append(p)
    mdlist.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    recent = "".join(
        f'<li><a href="/page/{p.relative_to(RL_ROOT).as_posix()[:-3]}">'
        f'{html.escape(p.stem.replace("-"," ").title())}</a>'
        f'<div class="snip">{time.strftime("%b %d %H:%M", time.localtime(p.stat().st_mtime))} · {p.parent.relative_to(RL_ROOT).as_posix() or "root"}</div></li>'
        for p in mdlist[:14])
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

@app.get("/browse/{category}", response_class=HTMLResponse)
def browse(category: str):
    reload_if_stale()
    d = RL_ROOT / category
    if not d.is_dir():
        return _404(category)
    pages = []
    for p in d.rglob("*.md"):
        if p.is_file():
            rel = p.relative_to(RL_ROOT).as_posix()
            fm, _ = _frontmatter(p.read_text(errors="ignore")[:2000])
            title = fm.get("title") or p.stem.replace("-"," ").title()
            pages.append((rel[:-3], title, time.localtime(p.stat().st_mtime)))
    pages.sort(key=lambda x: (x[0].count("/"), x[1].lower()))
    items = "".join(
        f'<li><a href="/page/{k}">{html.escape(t)}</a></li>' for k, t, _ in pages)
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>{CSS}</style>
<title>{html.escape(category)} · {html.escape(USER_LABEL)} RL</title></head><body>
<header class="top"><div class="bar">
<span class="brand">{html.escape(USER_LABEL)} <span class="user">RL</span></span>
<form class="search" action="/search" method="get"><input name="q" placeholder="Search…"></form>
</div></header><div class="wrap">
<div class="crumbs"><a href="/">home</a> / {html.escape(category)}</div>
<h1>{html.escape(category.replace("-"," ").title())} <span class="badge">{len(pages)} pages</span></h1>
<div class="card"><ul class="plist">{items or '<li class="snip">no pages</li>'}</ul></div>
<footer>RL Live Viewer</footer></div></body></html>"""

@app.get("/page/{path:path}", response_class=HTMLResponse)
def page(path: str):
    return render_page(path)

# ---------------------------------------------------------------- search
@app.get("/search", response_class=HTMLResponse)
def search(request: Request):
    q = (request.query_params.get("q") or "").strip()
    results = []
    total_pages = len(FILES)
    if q and SEARCH_DB.exists():
        try:
            con = sqlite3.connect(f"file:{SEARCH_DB}?mode=ro", uri=True)
            cur = con.cursor()
            # build a safe FTS5 query: AND of double-quoted tokens
            toks = [t for t in re.split(r"\s+", q) if t]
            fts_q = " AND ".join('"'+t.replace('"','')+ '"' for t in toks)
            try:
                rows = cur.execute(
                    "SELECT path, title, snippet(rl_fts,2,'<b>','</b>','…',24) "
                    "FROM rl_fts WHERE rl_fts MATCH ? ORDER BY rank LIMIT 25",
                    (fts_q,)).fetchall()
            except sqlite3.OperationalError:
                rows = cur.execute(
                    "SELECT path, title, snippet(rl_fts,2,'<b>','</b>','…',24) "
                    "FROM rl_fts WHERE rl_fts MATCH ? ORDER BY rank LIMIT 25",
                    (' OR '.join('"'+t.replace('"','')+'" ' for t in toks))).fetchall()
            con.close()
            results = rows
        except Exception as e:
            results = [(f"(search error: {e})", "", "")]
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

if __name__ == "__main__":
    import uvicorn
    print(f"RL Live Viewer: {RL_ROOT} on :{PORT}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
