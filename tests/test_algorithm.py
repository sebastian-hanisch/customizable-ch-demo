"""Customizable CH gegen unabhängige Referenzen: networkx-Allpaar-Entfernungen für jede Ordnung, beide Abfragen, gerichtete und ungerichtete Graphen; Struktur (chordaler Abschluss, Dreiecke, Eliminationsbaum) und
Gewichte (kürzeste Wege über tiefere Knoten) werden aus den Definitionen nachgerechnet; teilweises Anpassen gegen vollständiges Anpassen."""

import heapq

import networkx as nx
import numpy as np
import pytest

import cc_cch as cc
import cc_ch as ch
from cc_graph import from_arcs, route_cost

INF = float("inf")


def _random_graph(n, m, seed, directed=False, zero=False):
    """Zusammenhängender (bei gerichteten Graphen: stark zusammenhängender) Zufallsgraph mit ganzzahligen Kosten 1 bis 9 (mit `zero`: auch 0)."""
    rng = np.random.default_rng(seed)
    lo = 0 if zero else 1
    arcs = {(int(rng.integers(0, i)), i): float(rng.integers(lo, 10)) for i in range(1, n)}
    if directed:
        arcs.update({(i, int(rng.integers(0, i))): float(rng.integers(lo, 10)) for i in range(1, n)})
    while len(arcs) < m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (u, v) not in arcs and (directed or (v, u) not in arcs):
            arcs[(u, v)] = float(rng.integers(lo, 10))
    return from_arcs(n, [(u, v, w) for (u, v), w in arcs.items()], rng.random((n, 2)), directed=directed)


def _apsp(g, weights=None):
    G = nx.DiGraph()                                                     # die Kanten stehen auch bei ungerichteten Graphen in beiden Richtungen einzeln im CSR
    G.add_nodes_from(range(g.n))
    w = g.weight if weights is None else weights
    for u in range(g.n):
        for k in range(g.indptr[u], g.indptr[u + 1]):
            v = int(g.indices[k])
            if not G.has_edge(u, v) or w[k] < G[u][v]["weight"]:
                G.add_edge(u, v, weight=float(w[k]))
    ref = dict(nx.all_pairs_dijkstra_path_length(G))
    return [[ref[s].get(t, INF) for t in range(g.n)] for s in range(g.n)]


CASES = [(n, m, seed) for n, m in ((6, 10), (12, 22), (25, 60)) for seed in range(4)]


def _order(g, kind, seed):
    return cc.make_order(g, kind, seed, ch.build_hierarchy(g, "lazy_edge_difference"))


def _cch(g, kind="nd", seed=0):
    c = cc.build_cch(g, _order(g, kind, seed))
    return c, cc.customize(c, g)


# --- Richtigkeit gegen alle Paare ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("directed", (False, True))
@pytest.mark.parametrize("kind", cc.ORDERS)
@pytest.mark.parametrize("n,m,seed", CASES)
def test_both_queries_answer_every_pair_exactly(directed, kind, n, m, seed):
    g = _random_graph(n, m, seed, directed)
    ref = _apsp(g)
    c, met = _cch(g, kind, seed)
    for s in range(n):
        for t in range(n):
            for q in (cc.cch_query(c, met, s, t), cc.cch_query_dijkstra(c, met, s, t)):
                assert q.dist == pytest.approx(ref[s][t]), (kind, s, t)
                if q.dist < INF:
                    assert q.route[0] == s and q.route[-1] == t and route_cost(g, q.route) == pytest.approx(q.dist), (kind, s, t)
                else:
                    assert q.route == [] and q.meet == -1


@pytest.mark.parametrize("directed", (False, True))
def test_zero_cost_edges_and_parallel_arcs_do_not_disturb(directed):
    g = _random_graph(14, 26, 3, directed, zero=True)
    ref = _apsp(g)
    for kind in cc.ORDERS:
        c, met = _cch(g, kind, 3)
        for s in range(g.n):
            for t in range(g.n):
                assert cc.cch_query(c, met, s, t).dist == pytest.approx(ref[s][t]) and cc.cch_query_dijkstra(c, met, s, t).dist == pytest.approx(ref[s][t])
    raw = from_arcs(3, [(0, 1, 5.0), (0, 1, 2.0), (1, 2, 0.0)], np.zeros((3, 2)), directed=True, clean=False)             # Parallelkanten bleiben (billigste gilt)
    c = cc.build_cch(raw, [0, 1, 2])
    met = cc.customize(c, raw)
    assert cc.cch_query(c, met, 0, 2).dist == 2.0 and cc.cch_query(c, met, 2, 0).dist == INF


def test_unreachable_pairs_and_the_same_node():
    g = from_arcs(4, [(0, 1, 1.0), (1, 2, 2.0)], np.zeros((4, 2)), directed=True)
    for kind in cc.ORDERS:
        c, met = _cch(g, kind, 0)
        for query in (cc.cch_query, cc.cch_query_dijkstra):
            assert query(c, met, 0, 2).dist == 3.0 and query(c, met, 2, 0).dist == INF and query(c, met, 0, 3).dist == INF
            assert query(c, met, 3, 3).dist == 0.0 and query(c, met, 3, 3).route == [3]


# --- Struktur: chordaler Abschluss, Dreiecke, Eliminationsbaum ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", cc.ORDERS)
@pytest.mark.parametrize("n,m,seed", CASES)
def test_structure_is_the_chordal_completion_with_the_right_triangles_and_tree(kind, n, m, seed):
    g = _random_graph(n, m, seed)
    c, _ = _cch(g, kind, seed)
    rk = c.rank
    edges = {(c.lo[e], c.hi[e]) for e in range(c.m)}
    assert len(edges) == c.m and all(rk[u] < rk[v] for u, v in edges)
    orig = {(u, v) if rk[u] < rk[v] else (v, u) for u in range(n) for v in g.out(u).tolist() if u != v}
    assert orig <= edges and c.counters["original_edges"] == len(orig) and c.counters["fill_edges"] == len(edges) - len(orig)
    assert all(rk[c.lo[e]] <= rk[c.lo[e + 1]] for e in range(c.m - 1))                                     # sortiert nach dem Rang des unteren Endknotens
    up = {v: sorted((b for a, b in edges if a == v), key=lambda x: rk[x]) for v in range(n)}
    for v in range(n):                                                                                     # höhere Nachbarn bilden eine Clique
        for i, a in enumerate(up[v]):
            for b in up[v][i + 1:]:
                assert (a, b) in edges or (b, a) in edges
        assert c.parent[v] == (up[v][0] if up[v] else -1)
        assert set(up[v]) <= set(c.chain(v))                                                               # alle höheren Nachbarn sind Vorfahren
    # Dreiecke aus der Definition nachgezählt
    brute = 0
    for e in range(c.m):
        u, v = c.lo[e], c.hi[e]
        xs = {x for x in range(n) if rk[x] < rk[u] and (x, u) in edges and (x, v) in edges}
        assert {t[0] for t in c.tri[e]} == xs and all(c.lo[t[1]] == t[0] and c.hi[t[1]] == u and c.lo[t[2]] == t[0] and c.hi[t[2]] == v for t in c.tri[e])
        brute += len(xs)
    assert c.triangles() == c.counters["triangles"] == brute
    roots = [v for v in range(n) if c.parent[v] < 0]
    assert len(roots) == 1 and c.height() == max(c.depth(v) for v in range(n))                              # zusammenhängend: ein Baum


@pytest.mark.parametrize("kind", cc.ORDERS)
def test_orders_are_permutations_and_reproducible(kind):
    g = _random_graph(30, 70, 2)
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    o = cc.make_order(g, kind, 3, h)
    assert sorted(o) == list(range(30)) and o == cc.make_order(g, kind, 3, h)
    with pytest.raises(ValueError):
        cc.make_order(g, "ring", 0, h)


def test_a_good_order_needs_fewer_edges_than_a_random_one_on_a_grid():
    from cc_scenario import city_network
    g = city_network(10, 3).graph
    e = {kind: cc.build_cch(g, _order(g, kind, 3)).m for kind in cc.ORDERS}
    assert e["nd"] < e["random"] and e["mindeg"] < e["random"] and e["nd"] < e["ch"]
    assert cc.build_cch(g, _order(g, "nd", 3)).height() < cc.build_cch(g, _order(g, "random", 3)).height()


# --- Gewichte -----------------------------------------------------------------------------------------------------------------------------------------------

def _lower_distance(g, rank, u, v, limit):
    """Kürzeste Entfernung u -> v (Dijkstra), das nur Zwischenknoten mit Rang unter `limit` benutzt (eigene Referenz für die Gewichte der CCH)."""
    dist = {u: 0.0}
    heap = [(0.0, u)]
    done = set()
    while heap:
        d, x = heapq.heappop(heap)
        if x in done:
            continue
        done.add(x)
        if x == v:
            return d
        if x != u and rank[x] >= limit:
            continue
        for k in range(g.indptr[x], g.indptr[x + 1]):
            y = int(g.indices[k])
            nd = d + g.weight[k]
            if nd < dist.get(y, INF):
                dist[y] = nd
                heapq.heappush(heap, (nd, y))
    return INF


@pytest.mark.parametrize("directed", (False, True))
@pytest.mark.parametrize("kind", ("nd", "mindeg", "random"))
def test_every_weight_is_the_shortest_path_over_lower_nodes(directed, kind):
    g = _random_graph(18, 34, 5, directed)
    c, met = _cch(g, kind, 5)
    rk = c.rank
    for e in range(c.m):
        u, v = c.lo[e], c.hi[e]
        limit = rk[u]
        assert met.fw[e] == pytest.approx(_lower_distance(g, rk, u, v, limit)), (e, "fw")
        assert met.bw[e] == pytest.approx(_lower_distance(g, rk, v, u, limit)), (e, "bw")


@pytest.mark.parametrize("directed", (False, True))
def test_weights_are_upper_bounds_of_the_true_distance_and_middles_explain_them(directed):
    g = _random_graph(20, 40, 6, directed)
    ref = _apsp(g)
    c, met = _cch(g, "nd", 6)
    for e in range(c.m):
        u, v = c.lo[e], c.hi[e]
        assert met.fw[e] >= ref[u][v] - 1e-9 and met.bw[e] >= ref[v][u] - 1e-9
        x = met.mid_fw[e]
        if x >= 0:
            assert met.fw[e] == pytest.approx(met.bw[c.eid[(x, u)]] + met.fw[c.eid[(x, v)]])
        else:
            assert met.fw[e] == met.orig_fw[e]
        y = met.mid_bw[e]
        if y >= 0:
            assert met.bw[e] == pytest.approx(met.bw[c.eid[(y, v)]] + met.fw[c.eid[(y, u)]])
    assert met.counters["triangles"] == c.triangles()


def test_customize_trace_lists_every_edge_in_order():
    g = _random_graph(15, 30, 2)
    c = cc.build_cch(g, _order(g, "nd", 2))
    met = cc.customize(c, g, trace=True)
    assert [t[0] for t in met.trace] == list(range(c.m)) and all(t[3] <= t[1] for t in met.trace) and sum(t[5] for t in met.trace) == c.triangles()
    assert cc.customize(c, g).trace == []


# --- Teilweises Anpassen -------------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("directed", (False, True))
@pytest.mark.parametrize("kind", ("nd", "mindeg"))
def test_partial_customization_equals_full_customization_after_any_change(directed, kind):
    g = _random_graph(25, 60, 7, directed)
    c = cc.build_cch(g, _order(g, kind, 7))
    met = cc.customize(c, g)
    rng = np.random.default_rng(1)
    w = g.weight.copy()
    for round_ in range(6):
        hit = rng.random(len(w)) < (0.03, 0.1, 0.3, 0.6, 0.05, 1.0)[round_]
        w2 = w.copy()
        w2[hit] = rng.integers(0, 12, int(hit.sum())).astype(float)                                          # Erhöhungen und Senkungen, auch Nullkosten
        part, cnt = cc.recustomize(c, g, met, w2)
        full = cc.customize(c, g, w2)
        assert part.fw == full.fw and part.bw == full.bw and part.orig_fw == full.orig_fw and part.orig_bw == full.orig_bw, round_
        assert cnt["triangles"] <= c.triangles() * 1.0 + 1e-9 and cnt["edges"] <= c.m
        ref = _apsp(g, w2)
        for s in range(0, 25, 3):
            for t in range(25):
                assert cc.cch_query(c, part, s, t).dist == pytest.approx(ref[s][t])
        met, w = part, w2


def test_partial_customization_of_nothing_does_nothing_and_touches_only_what_changed():
    g = _random_graph(20, 40, 3)
    c = cc.build_cch(g, _order(g, "nd", 3))
    met = cc.customize(c, g)
    same, cnt = cc.recustomize(c, g, met, g.weight.copy())
    assert same.fw == met.fw and cnt == {"triangles": 0, "edges": 0, "changed_arcs": 0, "touched": 0}
    w = g.weight.copy()
    w[0] += 5.0
    part, cnt = cc.recustomize(c, g, met, w)
    assert cnt["changed_arcs"] == 1 and cnt["touched"] == 1 and 1 <= cnt["edges"] < c.m and cnt["triangles"] < c.triangles()
    assert part.fw != met.fw or part.bw != met.bw or part.orig_fw != met.orig_fw or part.orig_bw != met.orig_bw


def test_the_old_metric_is_not_modified_by_partial_customization():
    g = _random_graph(15, 30, 4)
    c = cc.build_cch(g, _order(g, "nd", 4))
    met = cc.customize(c, g)
    before = (list(met.fw), list(met.bw), list(met.weights))
    w = g.weight * 2.0
    cc.recustomize(c, g, met, w)
    assert (met.fw, met.bw, met.weights) == before


# --- Abfragen ------------------------------------------------------------------------------------------------------------------------------------------------

def test_tree_query_steps_are_the_two_chains_and_the_dijkstra_query_settles_no_more_than_that():
    g = _random_graph(25, 60, 8)
    c, met = _cch(g, "nd", 8)
    for s in range(0, 25, 3):
        for t in range(1, 25, 4):
            if s == t:
                continue
            q = cc.cch_query(c, met, s, t)
            assert q.steps == len(c.chain(s)) + len(c.chain(t)) and q.chain_s == c.chain(s) and q.chain_t == c.chain(t)
            assert q.meet in c.chain(s) and q.meet in c.chain(t)
            d = cc.cch_query_dijkstra(c, met, s, t)
            assert d.steps <= q.steps + 2 and d.dist == pytest.approx(q.dist)
            assert q.relaxed >= 0 and q.shortcuts_used >= 0


def test_customization_is_independent_of_the_costs_for_the_structure():
    g = _random_graph(20, 40, 9)
    c = cc.build_cch(g, _order(g, "nd", 9))
    a, b = cc.customize(c, g), cc.customize(c, g, g.weight * 3.0)
    assert a.fw != b.fw and cc.build_cch(g, _order(g, "nd", 9)).m == c.m
    ref3 = _apsp(g, g.weight * 3.0)
    assert all(cc.cch_query(c, b, s, t).dist == pytest.approx(ref3[s][t]) for s in range(20) for t in range(20))
