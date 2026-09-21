"""Customizable CH nach einer Kostenänderung im Vergleich mit Dijkstra und der Contraction Hierarchy (alt und neu aufgebaut): Kennzahlen, Bildfolge des Anpassens, Verteilung über Zufallspaare und die Messreihen der Experimente.
Aufwand als Zähler (Kanten, Dreiecke, festgelegte Knoten); Laufzeiten stehen nur als Messwerte in der App."""

import dataclasses
import heapq
import time
from dataclasses import dataclass

import numpy as np

import cc_cch as cch_mod
import cc_ch as ch
import cc_constants as C
from cc_scenario import make_network, traffic_weights

INF = float("inf")


@dataclass(frozen=True)
class Analysis:
    net: object
    order_kind: str
    fraction: int
    h: ch.Hierarchy                 # Contraction Hierarchy mit den alten Kosten
    h2: ch.Hierarchy                # neu aufgebaut mit den neuen Kosten
    cch: cch_mod.CCH
    m0: cch_mod.Metric              # angepasst an die alten Kosten
    m1: cch_mod.Metric              # angepasst an die neuen Kosten (vollständig, mit Protokoll)
    weights: np.ndarray             # neue Kosten je Kante des Originalgraphen
    s: int
    t: int
    q: cch_mod.CchQuery             # Abfrage mit den neuen Kosten (Eliminationsbaum)
    q_old: cch_mod.CchQuery         # Abfrage mit den alten Kosten
    route_stale: list               # Route der alten Contraction Hierarchy
    changed: list                   # je CCH-Kante: hat sich ihr Gewicht geändert?
    metrics: dict
    seconds: dict


def with_weights(g, w):
    return dataclasses.replace(g, weight=np.asarray(w, dtype=float))


def dijkstra(g, s, t=None):
    """Einfaches Dijkstra: Entfernungen, Reihenfolge der festgelegten Knoten (bei gegebenem Ziel: Abbruch, sobald es festliegt), Kantenprüfungen."""
    n = g.n
    ip, ix, w = g.indptr.tolist(), g.indices.tolist(), g.weight.tolist()
    dist = [INF] * n
    dist[s] = 0.0
    done = [False] * n
    heap = [(0.0, s)]
    order, relax = [], 0
    while heap:
        d, u = heapq.heappop(heap)
        if done[u]:
            continue
        done[u] = True
        order.append(u)
        if t is not None and u == t:
            break
        for k in range(ip[u], ip[u + 1]):
            relax += 1
            nd = d + w[k]
            v = ix[k]
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return np.array(dist), order, relax


def pick_pair(net, distance_pct=C.DEFAULT_DISTANCE, seed=C.DEFAULT_SEED):
    """Start und Ziel: bei Netzen mit fester Aufgabe diese; sonst ein Start (Netze mit Karte: nahe bei 30 % Breite und 50 % Höhe; sonst zufällig) und als Ziel der Knoten, dessen Entfernung vom Start in der Rangfolge
    aller erreichbaren Knoten bei `distance_pct` Prozent liegt."""
    if net.fixed_pair:
        return net.fixed_pair
    g = net.graph
    rng = np.random.default_rng([int(seed), 808])
    if net.geometric:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        s = int(np.argmin(np.hypot(*(g.xy - (lo + (hi - lo) * np.array([0.3, 0.5]))).T)))
    else:
        s = int(rng.integers(0, g.n))
    d = dijkstra(g, s)[0]
    reachable = np.where(np.isfinite(d))[0]
    reachable = reachable[reachable != s]
    order = reachable[np.argsort(d[reachable], kind="stable")]
    t = int(order[min(len(order) - 1, int(round(distance_pct / 100.0 * (len(order) - 1))))])
    return s, t


def build(net, order_kind=C.DEFAULT_ORDER, seed=C.DEFAULT_SEED):
    """Alles, was nicht vom Verkehr abhängt: die Contraction Hierarchy mit den alten Kosten (für den Vergleich), die Ordnung, das Zusammenziehen ohne Kosten und das Anpassen an die alten Kosten. Rückgabe: (h, cch, m0, Sekunden)."""
    g = net.graph
    t0 = time.perf_counter()
    h = ch.build_hierarchy(g, "lazy_edge_difference")
    t_ch = time.perf_counter() - t0
    t0 = time.perf_counter()
    order = cch_mod.make_order(g, order_kind, seed, h)
    t_order = time.perf_counter() - t0
    t0 = time.perf_counter()
    cch = cch_mod.build_cch(g, order)
    t_contract = time.perf_counter() - t0
    t0 = time.perf_counter()
    m0 = cch_mod.customize(cch, g)
    t_custom0 = time.perf_counter() - t0
    return h, cch, m0, {"ch": t_ch, "order": t_order, "contract": t_contract, "customize0": t_custom0}


def pair_samples(g1, cch, m1, h, h2, n_pairs=C.PAIR_SAMPLES, seed=0):
    """Zufallspaare mit den neuen Kosten: Anteil falscher Antworten der alten Hierarchie, ob die angepasste CCH (beide Abfragen) und die neu gebaute CH immer stimmen, mittlere Schritte je Verfahren."""
    rng = np.random.default_rng([int(seed), 909])
    su, st, sd, sc = [], [], [], []
    wrong_stale = wrong_cch = wrong_fresh = n = 0
    for _ in range(n_pairs):
        s, t = (int(x) for x in rng.integers(0, g1.n, 2))
        if s == t:
            continue
        dist, order, _ = dijkstra(g1, s, t)
        true = dist[t]
        q1, q2 = cch_mod.cch_query(cch, m1, s, t), cch_mod.cch_query_dijkstra(cch, m1, s, t)
        cq = ch.ch_query(h2, s, t)
        n += 1
        wrong_stale += abs(ch.ch_query(h, s, t).cost - true) > 1e-9
        wrong_cch += abs(q1.dist - true) > 1e-9 or abs(q2.dist - true) > 1e-9
        wrong_fresh += abs(cq.cost - true) > 1e-9
        su.append(len(order))
        st.append(q1.steps)
        sd.append(q2.steps)
        sc.append(cq.settled)
    return {"n": n, "wrong_stale": wrong_stale / n, "wrong_cch": int(wrong_cch), "wrong_fresh": int(wrong_fresh), "dijkstra": float(np.mean(su)), "tree": float(np.mean(st)), "cch_dijkstra": float(np.mean(sd)), "ch": float(np.mean(sc))}


def analyse(net, order_kind=C.DEFAULT_ORDER, fraction=C.DEFAULT_TRAFFIC, distance_pct=C.DEFAULT_DISTANCE, seed=C.DEFAULT_SEED, built=None):
    g = net.graph
    h, cch, m0, seconds = built if built is not None else build(net, order_kind, seed)
    w1 = traffic_weights(net, fraction / 100.0, seed)
    g1 = with_weights(g, w1)
    t0 = time.perf_counter()
    m1 = cch_mod.customize(cch, g, w1, trace=True)
    t_full = time.perf_counter() - t0
    t0 = time.perf_counter()
    m1p, pc = cch_mod.recustomize(cch, g, m0, w1)
    t_part = time.perf_counter() - t0
    if m1p.fw != m1.fw or m1p.bw != m1.bw:
        raise AssertionError("teilweises und vollständiges Anpassen stimmen nicht überein")
    t0 = time.perf_counter()
    h2 = ch.build_hierarchy(g1, "lazy_edge_difference")
    t_rebuild = time.perf_counter() - t0
    s, t = pick_pair(net, distance_pct, seed)
    dist, order, relax = dijkstra(g1, s, t)
    t0 = time.perf_counter()
    q = cch_mod.cch_query(cch, m1, s, t)
    t_query = time.perf_counter() - t0
    qd = cch_mod.cch_query_dijkstra(cch, m1, s, t)
    q_old = cch_mod.cch_query(cch, m0, s, t)
    stale = ch.ch_query(h, s, t)
    fresh = ch.ch_query(h2, s, t)
    changed = [(m0.fw[e], m0.bw[e]) != (m1.fw[e], m1.bw[e]) for e in range(cch.m)]
    reachable = bool(np.isfinite(dist[t]))
    ps = pair_samples(g1, cch, m1, h, h2, seed=seed)
    n_changed_arcs = int((np.asarray(w1) != g.weight).sum())
    m = {"n": g.n, "arcs": g.m // 2, "reachable": reachable, "dist": float(dist[t]), "dist_old": float(q_old.dist), "dist_stale": float(stale.cost), "exact": bool(reachable and abs(q.dist - dist[t]) < 1e-9 and abs(qd.dist - dist[t]) < 1e-9 and abs(fresh.cost - dist[t]) < 1e-9),
         "stale_wrong": bool(reachable and abs(stale.cost - dist[t]) > 1e-9), "route_changed": q.route != q_old.route,
         "edges": cch.m, "original_edges": cch.counters["original_edges"], "fill_edges": cch.counters["fill_edges"], "triangles": cch.counters["triangles"], "height": cch.height(),
         "ch_edges": (h.n_original + h.n_shortcuts) // (1 if g.directed else 2), "ch_shortcuts": h.n_shortcuts // (1 if g.directed else 2), "ch_shortcuts_new": h2.n_shortcuts // (1 if g.directed else 2),
         "changed_arcs": n_changed_arcs // (1 if g.directed else 2), "changed_edges": int(sum(changed)),
         "tri_full": m1.counters["triangles"], "tri_part": pc["triangles"], "edges_part": pc["edges"], "touched": pc["touched"],
         "witness_settled": h2.counters["witness_settled"], "witness_settled_old": h.counters["witness_settled"],
         "steps_tree": q.steps, "steps_cchdij": qd.steps, "steps_ch": fresh.settled, "steps_dijkstra": len(order), "relaxed_tree": q.relaxed, "shortcuts_used": q.shortcuts_used, "hops": len(q.route) - 1 if q.route else 0,
         "pairs": ps}
    seconds = {**seconds, "customize": t_full, "recustomize": t_part, "rebuild": t_rebuild, "query": t_query}
    return Analysis(net, order_kind, fraction, h, h2, cch, m0, m1, w1, s, t, q, q_old, stale.route, changed, m, seconds)


def verdict(a):
    """Code für die App: unreachable / nochange (keine Straße im Stau) / same (Route bleibt) / changed (Route ändert sich)."""
    m = a.metrics
    if not m["reachable"]:
        return "unreachable"
    if m["changed_arcs"] == 0:
        return "nochange"
    return "changed" if m["route_changed"] else "same"


def frames(a, max_frames=60):
    """Welche Zahlen bearbeiteter Kanten beim Anpassen gezeigt werden (0 bis Ende, höchstens `max_frames` + 1 Bilder)."""
    n = len(a.m1.trace)
    if n <= max_frames:
        return list(range(n + 1))
    return sorted({round(i * n / max_frames) for i in range(max_frames + 1)} | {0, n})


# --- Messreihen ---------------------------------------------------------------------------------------------------------------------------------------------

def _mean(rows):
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


EDGE_CASES = [("city", 6), ("city", 10), ("city", 14), ("city", 20), ("random", 100), ("random", 200)]


def edge_rows(cases=EDGE_CASES, seeds=C.SWEEP_SEEDS):
    """Kanten der CCH je Ordnung (Mittel über die Sweep-Netze) gegen Originalkanten plus Abkürzungen der Contraction Hierarchy (`ch_hier`, ungerichtet gezählt); Zufallsordnung nur bis 200 Knoten (darüber zu groß)."""
    rows = []
    for key, size in cases:
        acc = []
        for sd in seeds:
            g = make_network(key, side=size, nodes=size, seed=sd).graph
            h = ch.build_hierarchy(g, "lazy_edge_difference")
            row = {"n": g.n, "arcs": g.m // 2, "ch_hier": (h.n_original + h.n_shortcuts) / 2}
            for kind in cch_mod.ORDERS:
                if kind == "random" and g.n > 200:
                    row[kind] = float("nan")
                    row[f"{kind}_tri"] = float("nan")
                    continue
                c = cch_mod.build_cch(g, cch_mod.make_order(g, kind, sd, h))
                row[kind], row[f"{kind}_tri"] = c.m, c.triangles()
            row["nd_height"] = cch_mod.build_cch(g, cch_mod.make_order(g, "nd")).height()
            acc.append(row)
        rows.append({"key": key, "size": size, **{k: (float(np.nanmean([r[k] for r in acc])) if not np.all(np.isnan([r[k] for r in acc])) else float("nan")) for k in acc[0]}})
    return rows


CHANGE_FRACTIONS = (1, 2, 5, 10, 20, 50)


def change_rows(fractions=CHANGE_FRACTIONS, side=20, order_kind="nd", seeds=C.SWEEP_SEEDS, pairs=100):
    """Nach einer Kostenänderung (Stadtnetz, ein Anteil der Straßen dreimal so teuer): Dreiecke beim vollständigen und beim teilweisen Anpassen, festgelegte Knoten der Zeugensuchen beim Neuaufbau der CH, Anteil falscher
    Antworten der alten CH. Mittel über die Sweep-Netze."""
    rows = []
    for frac in fractions:
        acc = []
        for sd in seeds:
            net = make_network("city", side=side, seed=sd)
            g = net.graph
            built = build(net, order_kind, sd)
            h, cch, m0, _ = built
            w1 = traffic_weights(net, frac / 100.0, sd)
            g1 = with_weights(g, w1)
            m1 = cch_mod.customize(cch, g, w1)
            m1p, pc = cch_mod.recustomize(cch, g, m0, w1)
            assert m1p.fw == m1.fw and m1p.bw == m1.bw
            h2 = ch.build_hierarchy(g1, "lazy_edge_difference")
            ps = pair_samples(g1, cch, m1, h, h2, n_pairs=pairs, seed=sd)
            assert ps["wrong_cch"] == 0 and ps["wrong_fresh"] == 0
            acc.append({"tri_full": m1.counters["triangles"], "tri_part": pc["triangles"], "edges_part": pc["edges"], "edges": cch.m, "witness": h2.counters["witness_settled"], "wrong": ps["wrong_stale"]})
        rows.append({"fraction": frac, **_mean(acc)})
    return rows


def query_rows(cases=EDGE_CASES, seeds=C.SWEEP_SEEDS[:3], fraction=10):
    """Schritte einer Abfrage nach dem Anpassen (Mittel über 100 Zufallspaare je Netz): Eliminationsbaum, bidirektionale Aufwärtssuche auf der CCH, neu gebaute CH und Dijkstra."""
    rows = []
    for key, size in cases:
        acc = []
        for sd in seeds:
            net = make_network(key, side=size, nodes=size, seed=sd)
            g = net.graph
            order_kind = "nd" if key == "city" else "mindeg"
            h, cch, m0, _ = build(net, order_kind, sd)
            w1 = traffic_weights(net, fraction / 100.0, sd)
            g1 = with_weights(g, w1)
            m1 = cch_mod.customize(cch, g, w1)
            h2 = ch.build_hierarchy(g1, "lazy_edge_difference")
            ps = pair_samples(g1, cch, m1, h, h2, seed=sd)
            assert ps["wrong_cch"] == 0
            acc.append({"n": g.n, "tree": ps["tree"], "cch_dijkstra": ps["cch_dijkstra"], "ch": ps["ch"], "dijkstra": ps["dijkstra"]})
        rows.append({"key": key, "size": size, **_mean(acc)})
    return rows
