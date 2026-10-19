"""Tests for rl_viewer.py graph views (owner 10/16: code changes ship WITH their tests).

Covers:
(a) fixture tree -> out-edges and backlinks both captured, mutually consistent
    (edges == backlinks inverted), exact expected edge set, category matrix
(b) mega-tree collapse (fixture pages under the first MEGA_TREES prefix fold
    into a single labeled node; internal edges vanish; page count preserved)
(c) ego extraction honors hops=1 vs hops=2 (exact node sets)
(d) GET /graph -> 200, contains "<svg", zero external http(s) refs; layout +
    page are deterministic (seeded)
(e) GET /graph/heat -> 200 + category table
Plus: GET /graph/ego route (200 with neighbors / 404 for unknown path).

rl_viewer is a flat repo-root module with import-time index build, so the
fixture does a FRESH import with RLV_ROOT pointed at a small temp tree, and
unloads the module afterwards so it cannot leak into other test modules.
"""

import html
import os
import sys
from pathlib import Path

import pytest


FIXTURE_FILES = {
    "alpha.md": "Alpha\n\nLinks [[beta]] and [[crenshaw/finances/qbo]].\n",
    "beta.md": "Beta\n",
    "crenshaw/finances/qbo.md": "QBO\n\n[[alpha]] [[crenshaw/jobs/e1]]\n",
    "crenshaw/jobs/e1.md": "E1\n\n[[crenshaw/finances/qbo]]\n",
    "crenshaw/jobs/e2.md": "E2 (no links)\n",
    "religion/bible/a.md": "A\n\n[[religion/bible/b]]\n",
    "religion/bible/b.md": "B\n\n[[crenshaw/jobs/e1]]\n",
}

# Expected resolved edges (no-.md keys):
#   alpha -> beta ; alpha -> crenshaw/finances/qbo
#   crenshaw/finances/qbo -> alpha ; crenshaw/finances/qbo -> crenshaw/jobs/e1
#   crenshaw/jobs/e1 -> crenshaw/finances/qbo
#   religion/bible/a -> religion/bible/b
#   religion/bible/b -> crenshaw/jobs/e1
# NOTE: crenshaw/jobs/e1 sits inside the "crenshaw/jobs" MEGA_TREES prefix, so
# graph views collapse it into the mega node — that is intended behavior.
EXPECTED_OUT_EDGES = {
    "alpha": {"beta", "crenshaw/finances/qbo"},
    "crenshaw/finances/qbo": {"alpha", "crenshaw/jobs/e1"},
    "crenshaw/jobs/e1": {"crenshaw/finances/qbo"},
    "religion/bible/a": {"religion/bible/b"},
    "religion/bible/b": {"crenshaw/jobs/e1"},
}


def _write_fixture(root: Path):
    for rel, text in FIXTURE_FILES.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


@pytest.fixture(scope="module")
def rl(tmp_path_factory):
    """Fresh rl_viewer import over a small fixture tree (module scope).

    Yields (rl_viewer_module, TestClient, fixture_root)."""
    root = tmp_path_factory.mktemp("rl_fixture")
    _write_fixture(root)
    os.environ["RLV_ROOT"] = str(root)
    os.environ["RLV_PORT"] = "8090"
    sys.modules.pop("rl_viewer", None)
    import rl_viewer  # flat import — repo root is on sys.path (conftest)
    from fastapi.testclient import TestClient
    try:
        yield rl_viewer, TestClient(rl_viewer.app), root
    finally:
        sys.modules.pop("rl_viewer", None)


def _edge_pairs(out_edges):
    return {(s, d) if s < d else (d, s)
            for s, dsts in out_edges.items() for d in dsts if d != s}


class TestBuildGraphData:
    def test_out_edges_exact(self, rl):
        rlv, _, root = rl
        data = rlv.build_graph_data(str(root))
        assert data["out_edges"] == EXPECTED_OUT_EDGES

    def test_edges_and_backlinks_mutually_consistent(self, rl):
        """(a) edges == backlinks inverted — the graph can never disagree
        with the per-page '⬅ Backlinks (N)' section."""
        rlv, _, root = rl
        data = rlv.build_graph_data(str(root))
        oe, bl = data["out_edges"], data["backlinks"]
        for s, dsts in oe.items():
            for d in dsts:
                assert s in bl[d], f"backlink missing: {s} -> {d}"
        for d, srcs in bl.items():
            for s in srcs:
                assert d in oe[s], f"out-edge missing: {s} -> {d}"
        # exact inversion
        for s in oe:
            assert oe[s] == {d for d, srcs in bl.items() if s in srcs}

    def test_unlinked_page_absent(self, rl):
        rlv, _, root = rl
        data = rlv.build_graph_data(str(root))
        assert "crenshaw/jobs/e2" not in data["out_edges"]
        assert "crenshaw/jobs/e2" not in data["backlinks"]

    def test_cat_matrix_counts(self, rl):
        rlv, _, root = rl
        cm = rlv.build_graph_data(str(root))["cat_matrix"]
        assert cm[("(root)", "(root)")] == 1         # alpha -> beta
        assert cm[("(root)", "crenshaw")] == 1       # alpha -> qbo
        assert cm[("crenshaw", "(root)")] == 1       # qbo -> alpha
        assert cm[("crenshaw", "crenshaw")] == 2     # qbo -> e1, e1 -> qbo
        assert cm[("religion", "religion")] == 1   # a -> b (internal)
        assert cm[("religion", "crenshaw")] == 1    # b -> e1
        assert sum(cm.values()) == 7                 # 7 resolved wikilinks


class TestMegaCollapse:
    def test_mega_tree_collapses_to_one_node(self, rl):
        """(b) fixture pages under a mega-tree prefix fold into ONE labeled
        node; internal edges vanish; page counts preserved."""
        rlv, _, root = rl
        data = rlv.build_graph_data(str(root))
        pre = rlv.MEGA_TREES[0]
        jobs_pre = "crenshaw/jobs"   # MEGA_TREES[1] — e1/e2 live under it
        assert pre == "religion/bible"
        node_set = {k for k in data["files"]
                    if k in data["out_edges"] or k in data["backlinks"]}
        edges_in = _edge_pairs(data["out_edges"])
        nodes, edges, mega_pages = rlv.collapse_mega(node_set, edges_in, set(data["files"]))
        # bible: a+b collapse to one node
        assert pre in nodes
        assert "religion/bible/a" not in nodes
        assert "religion/bible/b" not in nodes
        assert mega_pages[pre] == 2
        # jobs: e1 (the only linked jobs page) collapses to the mega node
        assert jobs_pre in nodes
        assert "crenshaw/jobs/e1" not in nodes
        assert mega_pages[jobs_pre] == 2            # e1 + e2 counted from all_files
        # cross-tree edges survive, pointing at the mega node (pairs stored sorted)
        assert (min(pre, jobs_pre), max(pre, jobs_pre)) in edges
        assert ("crenshaw/finances/qbo", jobs_pre) in edges
        # internal mega-tree edge vanished
        assert ("religion/bible/a", "religion/bible/b") not in edges
        # non-mega nodes untouched
        assert "alpha" in nodes and "beta" in nodes

    def test_graph_core_caps_and_reports_omitted(self, rl):
        rlv, _, root = rl
        data = rlv.build_graph_data(str(root))
        node_set = {k for k in data["files"]
                    if k in data["out_edges"] or k in data["backlinks"]}
        edges_in = _edge_pairs(data["out_edges"])
        # cap below node count: beta has degree 1 -> must be omitted,
        # mega node must survive, omitted count reported
        nodes, edges, omitted, mega_pages, mega_links, deg = rlv.graph_core(
            node_set, edges_in, set(data["files"]), max_nodes=4)
        assert "beta" not in nodes
        assert rlv.MEGA_TREES[0] in nodes
        assert omitted == 1
        assert all(a in set(nodes) and b in set(nodes) for a, b in edges)
        # mega link weight: a->b internal = 1 (the b->e1 cross edge is counted
        # under crenshaw/jobs: sorted pair puts that endpoint first, `ma or mb`
        # prefers it — a letter-order fixture quirk, not a code change)
        assert mega_links[rlv.MEGA_TREES[0]] == 1
        assert mega_links["crenshaw/jobs"] == 2     # qbo<->e1 internal + e1->b cross


class TestEgo:
    def test_hops_1_vs_2(self, rl):
        """(c) ego extraction honors hops: 1 -> direct neighbors only
        (out- AND in-edges); 2 -> one more ring (e1 reached via qbo)."""
        rlv, _, root = rl
        data = rlv.build_graph_data(str(root))
        oe, bl = data["out_edges"], data["backlinks"]
        n1, e1 = rlv.ego_graph("alpha", oe, bl, 1)
        assert n1 == {"alpha", "beta", "crenshaw/finances/qbo"}
        assert ("alpha", "beta") in e1 and ("alpha", "crenshaw/finances/qbo") in e1
        n2, e2 = rlv.ego_graph("alpha", oe, bl, 2)
        assert n2 == {"alpha", "beta", "crenshaw/finances/qbo", "crenshaw/jobs/e1"}
        assert len(e2) > len(e1)
        # every edge endpoint stays inside the node set
        assert all(a in n2 and b in n2 for a, b in e2)

    def test_hops_capped_at_4(self, rl):
        rlv, _, root = rl
        data = rlv.build_graph_data(str(root))
        n9, _ = rlv.ego_graph("alpha", data["out_edges"], data["backlinks"], 9)
        n4, _ = rlv.ego_graph("alpha", data["out_edges"], data["backlinks"], 4)
        assert n9 == n4


class TestRoutes:
    def test_graph_200_svg_offline(self, rl):
        """(d) /graph renders inline SVG with zero external references."""
        _, client, _ = rl
        r = client.get("/graph")
        assert r.status_code == 200
        assert "<svg" in r.text
        assert 'src="http' not in r.text
        assert "href=\"http" not in r.text
        # mega node present, collapsed (no per-page bible nodes linked)
        assert "religion/bible" in r.text
        assert "/page/religion/bible/a" not in r.text

    def test_graph_deterministic(self, rl):
        """Two calls -> byte-identical page; layout fn seeded (same input,
        same coordinates)."""
        rlv, client, _ = rl
        assert client.get("/graph").text == client.get("/graph").text
        a = rlv._fr_layout(["a/b", "c", "d/e"], [("a/b", "c"), ("c", "d/e")])
        b = rlv._fr_layout(["a/b", "c", "d/e"], [("a/b", "c"), ("c", "d/e")])
        assert a == b and len(a) == 3

    def test_ego_route(self, rl):
        _, client, _ = rl
        r = client.get("/graph/ego", params={"path": "alpha", "hops": 1})
        assert r.status_code == 200
        assert "<svg" in r.text
        assert '/page/beta' in r.text          # direct neighbor rendered
        assert '/page/crenshaw/jobs/e1' not in r.text  # 2-hop node excluded at hops=1
        assert 'src="http' not in r.text
        assert client.get("/graph/ego", params={"path": "nope", "hops": 1}).status_code == 404

    def test_heat_route(self, rl):
        """(e) /graph/heat is a server-rendered category table."""
        _, client, _ = rl
        r = client.get("/graph/heat")
        assert r.status_code == 200
        assert "<table" in r.text
        assert "<script" not in r.text          # no JS required
        for cat in ("crenshaw", "religion", "(root)"):
            assert html.escape(cat) in r.text
        # a known active cell (alpha -> qbo: (root) -> crenshaw, count 1)
        assert "rgba(88,166,255," in r.text

    def test_home_still_works(self, rl):
        """Existing routes unaffected by the graph additions."""
        _, client, _ = rl
        assert client.get("/").status_code == 200
        assert client.get("/page/alpha").status_code == 200
        assert "Backlinks (1)" in client.get("/page/alpha").text   # qbo -> alpha
