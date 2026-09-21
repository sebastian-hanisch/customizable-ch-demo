"""Netze mit Verkehr (kleines Netz, Stadtnetz, Zufallsnetz), Kennzahlen, Bildfolge des Anpassens, Abbildungen und Messreihen."""

import networkx as nx
import numpy as np
import pytest

import cc_cch as cc
import cc_constants as C
import cc_evaluation as ev
import cc_scenario as sc
import cc_visualization as viz


def _connected(g):
    G = nx.Graph()
    G.add_nodes_from(range(g.n))
    G.add_edges_from((int(u), int(v)) for u, v in zip(np.repeat(np.arange(g.n), g.degree()), g.indices))
    return nx.is_connected(G)


# --- Netze und Verkehr --------------------------------------------------------------------------------------------------------------------------------

def test_small_network_has_two_districts_two_bridges_and_a_jam_on_the_north_bridge():
    net = sc.small_network()
    g = net.graph
    assert g.n == 10 and g.m == 28 and _connected(g) and [g.names[v] for v in net.fixed_pair] == ["Altstadt", "Fabrik"] and net.unit == "min"
    w = sc.traffic_weights(net, 0.5)
    changed = {(g.names[u], g.names[v]) for u, v, a, b in zip(np.repeat(np.arange(g.n), g.degree()), g.indices, g.weight, w) if a != b}
    assert len(changed) == 6 and all("Brücke Nord" in pair for pair in changed) and np.array_equal(w[w != g.weight], g.weight[w != g.weight] * sc.SMALL_JAM_FACTOR)


@pytest.mark.parametrize("seed", range(3))
def test_city_network_is_a_connected_grid_with_whole_number_costs(seed):
    net = sc.city_network(8, seed)
    g = net.graph
    assert g.n == 64 and g.m == 2 * 2 * 8 * 7 and _connected(g) and (g.weight >= 1).all() and np.array_equal(g.weight, np.rint(g.weight)) and not net.jam


def test_random_network_is_connected_with_the_requested_degree():
    net = sc.random_network(150, 3.0, 4)
    assert _connected(net.graph) and abs(net.graph.m / net.graph.n - 3.0) < 0.1 and not net.geometric


def test_traffic_makes_a_share_of_the_streets_dearer_in_both_directions():
    net = sc.city_network(10, 3)
    g = net.graph
    for frac in (0.0, 0.1, 0.5):
        w = sc.traffic_weights(net, frac, 3)
        assert (w >= g.weight).all() and np.array_equal(w, sc.traffic_weights(net, frac, 3))
        src = np.repeat(np.arange(g.n), g.degree())
        by = {(int(u), int(v)): x for u, v, x in zip(src, g.indices, w)}
        assert all(by[(v, u)] == x for (u, v), x in by.items())
        assert abs((w != g.weight).mean() - frac) < 0.15
    assert np.array_equal(sc.traffic_weights(net, 0.0), g.weight) and (sc.traffic_weights(net, 1.0) == np.maximum(1.0, np.rint(g.weight * C.TRAFFIC_FACTOR))).all()


def test_make_network_rejects_unknown_nets_and_is_reproducible():
    with pytest.raises(ValueError):
        sc.make_network("ring")
    a, b = sc.make_network("city", side=6, seed=3).graph, sc.make_network("city", side=6, seed=3).graph
    assert np.array_equal(a.weight, b.weight) and np.array_equal(a.xy, b.xy)


# --- Kennzahlen -----------------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", C.NETS)
@pytest.mark.parametrize("order", tuple(C.ORDER_LABELS))
def test_analysis_invariants_for_every_net_and_order(key, order):
    net = sc.make_network(key, side=6, nodes=40)
    a = ev.analyse(net, order, 20, 60)
    m, ps = a.metrics, a.metrics["pairs"]
    assert m["reachable"] and m["exact"] and ps["wrong_cch"] == 0 and ps["wrong_fresh"] == 0 and 0 <= ps["wrong_stale"] <= 1
    assert m["edges"] == m["original_edges"] + m["fill_edges"] == a.cch.m and m["triangles"] == a.cch.triangles() and m["tri_full"] == m["triangles"] and m["tri_part"] <= m["tri_full"]
    assert m["edges_part"] <= m["edges"] and m["touched"] <= m["edges_part"] and m["changed_edges"] == sum(a.changed) and len(a.changed) == a.cch.m and len(a.m1.trace) == a.cch.m
    assert a.q.route[0] == a.s and a.q.route[-1] == a.t and m["hops"] == len(a.q.route) - 1 and a.q.dist == m["dist"] and m["dist"] >= m["dist_old"] - 1e-9        # nur teurer geworden
    assert m["steps_tree"] == len(a.q.chain_s) + len(a.q.chain_t) and m["steps_dijkstra"] >= 1 and m["witness_settled"] > 0 and m["ch_edges"] >= m["arcs"]
    assert ev.verdict(a) in ("changed", "same", "nochange")


def test_verdict_covers_the_cases():
    small = ev.analyse(sc.small_network(), "nd")
    assert ev.verdict(small) == "changed" and small.metrics["stale_wrong"] and small.metrics["route_changed"]
    net = sc.city_network(8, 3)
    none = ev.analyse(net, "nd", 0, 60)
    assert ev.verdict(none) == "nochange" and none.metrics["changed_arcs"] == 0 and none.metrics["tri_part"] == 0 and not any(none.changed) and none.metrics["pairs"]["wrong_stale"] == 0
    verdicts = {ev.verdict(ev.analyse(net, "nd", 20, pct)) for pct in (0, 20, 40, 60, 80, 100)}
    assert verdicts <= {"changed", "same"} and "changed" in verdicts


def test_pick_pair_follows_the_distance_percentile():
    net = sc.city_network(8, 3)
    d = {}
    for pct in (0, 50, 100):
        s, t = ev.pick_pair(net, pct, 3)
        d[pct] = ev.dijkstra(net.graph, s)[0][t]
        assert s != t
    assert d[0] <= d[50] <= d[100] and ev.pick_pair(net, 50, 3) == ev.pick_pair(net, 50, 3)
    assert ev.pick_pair(sc.small_network()) == sc.small_network().fixed_pair


def test_dijkstra_matches_networkx():
    g = sc.city_network(6, 2).graph
    G = nx.Graph()
    for u, v, w in zip(np.repeat(np.arange(g.n), g.degree()), g.indices, g.weight):
        G.add_edge(int(u), int(v), weight=float(w))
    ref = nx.single_source_dijkstra_path_length(G, 0)
    d, order, relax = ev.dijkstra(g, 0)
    assert list(d) == pytest.approx([ref[v] for v in range(g.n)]) and order[0] == 0 and relax >= g.n - 1
    d2, order2, _ = ev.dijkstra(g, 0, 9)
    assert d2[9] == d[9] and order2[-1] == 9 and len(order2) <= len(order)


# --- Bildfolge und Abbildungen -------------------------------------------------------------------------------------------------------------------------------

def test_frames_cover_the_customization():
    a = ev.analyse(sc.small_network())
    n = len(a.m1.trace)
    assert ev.frames(a) == list(range(n + 1)) and n == 22
    big = ev.analyse(sc.city_network(10, 3), "nd", 10, 60)
    fb = ev.frames(big)
    assert fb[0] == 0 and fb[-1] == len(big.m1.trace) and len(fb) <= 61 and fb == sorted(set(fb))


def test_tables_list_the_steps_and_all_edges_of_the_small_network():
    a = ev.analyse(sc.small_network())
    rows = viz.edge_table(a)
    assert len(rows) == 22 and sum(r["Art"] == "Straße" for r in rows) == 14 and sum("Verkehr" in r["Gewicht alt → neu"] for r in rows) == 7
    steps = viz.step_table(a, 22)
    assert [r["Schritt"] for r in steps] == list(range(17, 23)) and all("→" in r["Gewicht"] for r in steps)
    assert len(viz.step_table(a, 3)) == 3 and viz.step_table(a, 0) == []


def test_charts_render_for_every_net_and_frame():
    for key in C.NETS:
        a = ev.analyse(sc.make_network(key, side=6, nodes=40), "nd" if key != "random" else "mindeg")
        frames = ev.frames(a)
        for k in (frames[0], frames[len(frames) // 2], frames[-1]):
            viz.build_network(a, k)
    viz.build_edges([{"key": "city", "n": 36, "ch_hier": 95.0, "nd": 153.0, "mindeg": 132.0, "ch": 172.0, "random": 202.0}, {"key": "city", "n": 400, "ch_hier": 1766.0, "nd": 4358.0, "mindeg": 3530.0, "ch": 11646.0, "random": float("nan")}])
    viz.build_change([{"fraction": 1, "tri_full": 100.0, "tri_part": 40.0, "witness": 200.0}])
    viz.build_wrong([{"fraction": 1, "wrong": 0.07}])
    viz.build_query([{"key": "city", "n": 36, "tree": 20.0, "cch_dijkstra": 14.0, "ch": 12.0, "dijkstra": 20.0}, {"key": "random", "n": 100, "tree": 30.0, "cch_dijkstra": 16.0, "ch": 15.0, "dijkstra": 48.0}])


# --- Messreihen ---------------------------------------------------------------------------------------------------------------------------------------

def test_edge_rows_show_that_the_cch_needs_more_edges_than_the_ch_and_that_the_order_matters():
    rows = ev.edge_rows(cases=[("city", 6), ("city", 10), ("random", 60)], seeds=C.SWEEP_SEEDS[:2])
    assert all(r["nd"] > r["ch_hier"] and r["mindeg"] > r["ch_hier"] and r["nd_height"] > 1 for r in rows)
    assert rows[1]["nd"] < rows[1]["ch"] and rows[1]["nd"] < rows[1]["random"] and rows[2]["mindeg"] < rows[2]["nd"]
    assert np.isnan(ev.edge_rows(cases=[("city", 18)], seeds=C.SWEEP_SEEDS[:1])[0]["random"])              # Zufallsordnung nur bis 200 Knoten


def test_change_rows_partial_customization_saves_only_for_small_changes():
    rows = ev.change_rows(fractions=(1, 20), side=8, seeds=C.SWEEP_SEEDS[:2], pairs=40)
    assert rows[0]["tri_part"] < rows[1]["tri_part"] <= rows[0]["tri_full"] == rows[1]["tri_full"] and rows[0]["wrong"] < rows[1]["wrong"] and rows[0]["witness"] > 0


def test_query_rows_are_exact_and_the_cch_beats_dijkstra():
    rows = ev.query_rows(cases=[("city", 10), ("random", 60)], seeds=C.SWEEP_SEEDS[:2])
    assert all(r["cch_dijkstra"] < r["dijkstra"] and r["ch"] <= r["cch_dijkstra"] * 1.5 and r["tree"] > r["cch_dijkstra"] for r in rows)
