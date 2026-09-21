"""Customizable Contraction Hierarchies (Dibbelt, Strasser, Wagner 2016): drei Phasen.

1. **Ordnung ohne Kosten:** eine Knotenordnung, die nur von der Form des Netzes abhängt - hier verschachtelte Zerlegung nach Koordinaten (nested dissection): das Netz wird an einer kleinen Trennmenge geteilt, beide Hälften werden zuerst
   nummeriert, die Trennmenge zuletzt (sie sind die wichtigsten Knoten). Zum Vergleich: kleinster Grad, die Ordnung der (kostenabhängigen) Contraction Hierarchy und Zufall.
2. **Zusammenziehen ohne Kosten:** die Knoten werden in dieser Ordnung zusammengezogen; beim Zusammenziehen von v werden alle höher nummerierten Nachbarn paarweise verbunden - **ohne Zeugensuche**, also unabhängig von den Kosten.
   Ergebnis: ein Graph mit Kanten zwischen unteren und oberen Knoten und für jede Kante ihre **unteren Dreiecke** (x, u, v): x unter u und v, beide über Kanten mit x verbunden.
3. **Anpassen:** die Kosten der Kanten werden von unten nach oben eingetragen: Originalkosten, dann `w(u,v) = min(w(u,v), w(u,x) + w(x,v))` über alle unteren Dreiecke. Nach einer Kostenänderung genügt es, die Kanten neu zu berechnen, deren Dreiecke betroffen sind.

Die Abfrage läuft im Eliminationsbaum (Elternknoten = niedrigster höherer Nachbar): von s und t aus je die Kette der Vorfahren entlang, ohne Warteschlange.

Alles ist eigene Umsetzung auf dem CSR-Graphen aus cc_graph.py; networkx kommt nur in den Tests vor (Kreuzprobe)."""

import heapq
from dataclasses import dataclass, field

import numpy as np

INF = float("inf")
ORDERS = ("nd", "mindeg", "ch", "random")


# --- Phase 1: Ordnungen ohne Kosten -------------------------------------------------------------------------------------------------------------------------------

def _neighbours(g):
    """Nachbarmengen (ohne Richtung, ohne Schleifen)."""
    adj = [set() for _ in range(g.n)]
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    for u in range(g.n):
        for k in range(ip[u], ip[u + 1]):
            v = ix[k]
            if v != u:
                adj[u].add(v)
                adj[v].add(u)
    return adj


def order_nd(g, leaf=3):
    """Verschachtelte Zerlegung nach Koordinaten: Knotenliste von unten (zuerst zusammengezogen) nach oben. Die Knoten werden an der Mitte der längeren Ausdehnung geteilt; als Trennmenge dient die kleinere der beiden
    Randmengen (Knoten einer Hälfte mit Nachbarn in der anderen). Beide Hälften ohne Trennmenge werden rekursiv geordnet, die Trennmenge steht zuletzt."""
    adj = _neighbours(g)
    xy = g.xy

    def rec(nodes):
        if len(nodes) <= leaf:
            return sorted(nodes)
        pts = xy[nodes]
        axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
        srt = sorted(nodes, key=lambda v: (xy[v, axis], v))
        half = len(srt) // 2
        a, b = srt[:half], srt[half:]
        sa, sb = set(a), set(b)
        border_a = [v for v in a if adj[v] & sb]
        border_b = [v for v in b if adj[v] & sa]
        if len(border_a) <= len(border_b):
            sep, a, b = border_a, [v for v in a if v not in set(border_a)], b
        else:
            sep, a, b = border_b, a, [v for v in b if v not in set(border_b)]
        return rec(a) + rec(b) + sorted(sep, key=lambda v: (xy[v, axis], v))
    return rec(list(range(g.n)))


def order_mindeg(g, seed=0):
    """Kleinster Grad: wiederholt den Knoten mit den wenigsten Nachbarn im Restgraphen entfernen und seine Nachbarn paarweise verbinden (Gleichstand: zufällig)."""
    adj = _neighbours(g)
    tie = np.random.default_rng([int(seed), 4713]).random(g.n)
    heap = [(len(adj[v]), tie[v], v) for v in range(g.n)]
    heapq.heapify(heap)
    done = [False] * g.n
    out = []
    while heap:
        d, _, v = heapq.heappop(heap)
        if done[v] or d != len(adj[v]):
            continue
        done[v] = True
        out.append(v)
        nb = list(adj[v])
        for a in nb:
            adj[a].discard(v)
        for i, a in enumerate(nb):
            for b in nb[i + 1:]:
                adj[a].add(b)
                adj[b].add(a)
        for a in nb:
            heapq.heappush(heap, (len(adj[a]), tie[a], a))
    return out


def make_order(g, kind, seed=0, h=None):
    """Knoten von unten (zuerst zusammengezogen) nach oben. "ch" braucht die Hierarchie `h` (ihre Zusammenziehreihenfolge hängt von den Kosten ab)."""
    if kind == "nd":
        return order_nd(g)
    if kind == "mindeg":
        return order_mindeg(g, seed)
    if kind == "ch":
        return [int(v) for v in h.order]
    if kind == "random":
        return [int(v) for v in np.random.default_rng([int(seed), 4711]).permutation(g.n)]
    raise ValueError(kind)


# --- Phase 2: Zusammenziehen ohne Kosten ---------------------------------------------------------------------------------------------------------------------------

@dataclass
class CCH:
    n: int
    order: list                     # Knoten von unten nach oben
    rank: np.ndarray                # Rang je Knoten (0 = unten)
    lo: list                        # je Kante der untere Endknoten
    hi: list                        # je Kante der obere Endknoten
    eid: dict                       # (unten, oben) -> Kantennummer; Kanten sind nach (Rang unten, Rang oben) sortiert
    up: list                        # up[v] = Kantennummern zu höheren Nachbarn, nach Rang des oberen Endes sortiert
    tri: list                       # tri[e] = [(x, Kante (x, lo), Kante (x, hi))] : untere Dreiecke der Kante e
    dep: list                       # dep[e] = Kanten, deren Dreiecke die Kante e benutzen (für das teilweise Anpassen)
    parent: list                    # Elternknoten im Eliminationsbaum (-1: Wurzel)
    arc_edge: list                  # je Kante k des Originalgraphen: (Kantennummer, 0 = unten -> oben / 1 = oben -> unten)
    edge_arcs: list                 # je CCH-Kante: [(k, Richtung)] der Originalkanten, die auf sie abgebildet werden
    counters: dict = field(default_factory=dict)

    @property
    def m(self):
        return len(self.lo)

    def triangles(self):
        return sum(len(t) for t in self.tri)

    def depth(self, v):
        """Zahl der Knoten auf der Kette von v bis zur Wurzel des Eliminationsbaums (v eingeschlossen)."""
        d = 0
        while v >= 0:
            d += 1
            v = self.parent[v]
        return d

    def height(self):
        return max(self.depth(v) for v in range(self.n))

    def chain(self, v):
        out = []
        while v >= 0:
            out.append(v)
            v = self.parent[v]
        return out


def build_cch(g, order):
    """Zusammenziehen in der Reihenfolge `order` ohne Kosten und ohne Zeugensuche: die höher nummerierten Nachbarn eines Knotens werden paarweise verbunden. Rückgabe: Struktur mit Kanten, Dreiecken und Eliminationsbaum."""
    n = g.n
    rank = np.empty(n, dtype=np.int64)
    rank[np.asarray(order)] = np.arange(n)
    rk = rank.tolist()
    adj = _neighbours(g)
    original = sum(len(a) for a in adj) // 2
    ups = [None] * n
    for v in order:
        higher = sorted((u for u in adj[v] if rk[u] > rk[v]), key=lambda u: rk[u])
        ups[v] = higher
        for i, a in enumerate(higher):
            for b in higher[i + 1:]:
                adj[a].add(b)
                adj[b].add(a)
    lo, hi, eid = [], [], {}
    for v in order:
        for u in ups[v]:
            eid[(v, u)] = len(lo)
            lo.append(v)
            hi.append(u)
    up = [[eid[(v, u)] for u in ups[v]] for v in range(n)]
    tri = [[] for _ in lo]
    dep = [[] for _ in lo]
    for x in order:
        us = ups[x]
        for i, u in enumerate(us):
            e_xu = eid[(x, u)]
            for v in us[i + 1:]:
                e = eid[(u, v)]
                e_xv = eid[(x, v)]
                tri[e].append((x, e_xu, e_xv))
                dep[e_xu].append(e)
                dep[e_xv].append(e)
    parent = [ups[v][0] if ups[v] else -1 for v in range(n)]
    ip, ix = g.indptr.tolist(), g.indices.tolist()
    arc_edge = [None] * g.m
    edge_arcs = [[] for _ in lo]
    for u in range(n):
        for k in range(ip[u], ip[u + 1]):
            v = ix[k]
            if v == u:
                continue
            a, b = (u, v) if rk[u] < rk[v] else (v, u)
            e = eid[(a, b)]
            direction = 0 if u == a else 1
            arc_edge[k] = (e, direction)
            edge_arcs[e].append((k, direction))
    cch = CCH(n, list(order), rank, lo, hi, eid, up, tri, dep, parent, arc_edge, edge_arcs)
    cch.counters = {"edges": len(lo), "original_edges": original, "fill_edges": len(lo) - original, "triangles": cch.triangles()}
    return cch


# --- Phase 3: Anpassen -----------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class Metric:
    """Kosten einer CCH: je Kante e das Gewicht von unten nach oben (fw) und von oben nach unten (bw), und der Knoten x des Dreiecks, über den das Gewicht kommt (-1: Originalkante)."""
    fw: list
    bw: list
    mid_fw: list
    mid_bw: list
    orig_fw: list
    orig_bw: list
    weights: list = field(default_factory=list)      # Kosten je Kante des Originalgraphen, mit denen angepasst wurde
    counters: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)      # nur mit trace=True: je Kante (Nummer, Ausgang fw, Ausgang bw, neu fw, neu bw, Dreiecke)

    def copy(self):
        return Metric(list(self.fw), list(self.bw), list(self.mid_fw), list(self.mid_bw), list(self.orig_fw), list(self.orig_bw), list(self.weights), dict(self.counters))


def _original(cch, weights):
    """Kleinstes Originalgewicht je Kante und Richtung."""
    ofw, obw = [INF] * cch.m, [INF] * cch.m
    for e, arcs in enumerate(cch.edge_arcs):
        for k, d in arcs:
            if d == 0:
                ofw[e] = min(ofw[e], weights[k])
            else:
                obw[e] = min(obw[e], weights[k])
    return ofw, obw


def _relax_edge(cch, m, e):
    """Gewicht der Kante e aus den Originalgewichten und allen unteren Dreiecken (deren Kanten müssen schon angepasst sein). Rückgabe: (fw, bw, mid_fw, mid_bw)."""
    f, b, mf, mb = m.orig_fw[e], m.orig_bw[e], -1, -1
    fw, bw = m.fw, m.bw
    for x, e_xu, e_xv in cch.tri[e]:
        cf = bw[e_xu] + fw[e_xv]                    # u -> x -> v
        if cf < f:
            f, mf = cf, x
        cb = bw[e_xv] + fw[e_xu]                    # v -> x -> u
        if cb < b:
            b, mb = cb, x
    return f, b, mf, mb


def customize(cch, g, weights=None, trace=False):
    """Alle Kantengewichte von unten nach oben eintragen. `weights`: Kosten je Kante des Originalgraphen (Standard: die des Graphen). Zähler: bearbeitete Dreiecke."""
    w = g.weight.tolist() if weights is None else np.asarray(weights, dtype=float).tolist()
    ofw, obw = _original(cch, w)
    m = Metric([INF] * cch.m, [INF] * cch.m, [-1] * cch.m, [-1] * cch.m, ofw, obw, w)
    triangles = 0
    for e in range(cch.m):                            # Kanten sind nach dem Rang ihres unteren Endes sortiert: die Kanten der Dreiecke kommen vorher
        f, b, mf, mb = _relax_edge(cch, m, e)
        m.fw[e], m.bw[e], m.mid_fw[e], m.mid_bw[e] = f, b, mf, mb
        triangles += len(cch.tri[e])
        if trace:
            m.trace.append((e, ofw[e], obw[e], f, b, len(cch.tri[e])))
    m.counters = {"triangles": triangles, "edges": cch.m}
    return m


def recustomize(cch, g, old, new_weights):
    """Teilweises Anpassen: nur die Kanten neu berechnen, deren Originalgewicht sich geändert hat oder deren Dreiecke sich geändert haben (von unten nach oben mit einer Warteschlange). Rückgabe: (neue Kosten, Zähler);
    das Ergebnis ist genau das des vollständigen Anpassens mit `new_weights`."""
    nw = np.asarray(new_weights, dtype=float).tolist()
    ow = old.weights
    m = old.copy()
    changed = [k for k in range(len(nw)) if nw[k] != ow[k] and cch.arc_edge[k] is not None]
    touched = {cch.arc_edge[k][0] for k in changed}
    for e in touched:
        f, b = INF, INF
        for k, d in cch.edge_arcs[e]:
            if d == 0:
                f = min(f, nw[k])
            else:
                b = min(b, nw[k])
        m.orig_fw[e], m.orig_bw[e] = f, b
    heap = list(touched)
    heapq.heapify(heap)                                # Kantennummern sind nach dem Rang des unteren Endes sortiert: kleinste zuerst
    queued = set(touched)
    recomputed = triangles = 0
    while heap:
        e = heapq.heappop(heap)
        queued.discard(e)
        f, b, mf, mb = _relax_edge(cch, m, e)
        recomputed += 1
        triangles += len(cch.tri[e])
        if (f, b) != (m.fw[e], m.bw[e]):
            m.fw[e], m.bw[e] = f, b
            for d in cch.dep[e]:
                if d not in queued:
                    queued.add(d)
                    heapq.heappush(heap, d)
        m.mid_fw[e], m.mid_bw[e] = mf, mb
    m.counters = {"triangles": triangles, "edges": recomputed, "changed_arcs": len(changed), "touched": len(touched)}
    m.weights = nw
    return m, m.counters


# --- Abfrage -----------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class CchQuery:
    dist: float
    route: list                     # Knotenfolge s ... t (leer, wenn unerreichbar)
    meet: int                       # Knoten, an dem sich beide Seiten treffen (-1: unerreichbar)
    chain_s: list                   # Knoten der Kette von s (im Eliminationsbaum aufwärts)
    chain_t: list
    steps: int                      # bearbeitete Knoten beider Ketten
    relaxed: int                    # geprüfte Kanten
    shortcuts_used: int = 0


def _unpack(cch, m, e, forward, out):
    """Hängt die Originalknoten der Kante e (in Richtung `forward`: unten -> oben, sonst oben -> unten) an `out` an (ohne den Anfangsknoten)."""
    x = m.mid_fw[e] if forward else m.mid_bw[e]
    u, v = cch.lo[e], cch.hi[e]
    if x < 0:
        out.append(v if forward else u)
        return 0
    e_xu, e_xv = cch.eid[(x, u)], cch.eid[(x, v)]
    if forward:                                                    # u -> x -> v
        return 1 + _unpack(cch, m, e_xu, False, out) + _unpack(cch, m, e_xv, True, out)
    return 1 + _unpack(cch, m, e_xv, False, out) + _unpack(cch, m, e_xu, True, out)     # v -> x -> u


def cch_query(cch, m, s, t):
    """Abfrage im Eliminationsbaum: von s aus die Kette der Vorfahren entlang (nur Kanten nach oben, ihre Gewichte von unten nach oben), von t aus ebenso (Gewichte von oben nach unten); jede Kette wird von unten nach oben abgearbeitet,
    dabei ist jeder Knoten endgültig, wenn er drankommt. Die Entfernung ist das Minimum von d(s, x) + d(x, t) über die Knoten x beider Ketten."""
    s, t = int(s), int(t)
    if s == t:
        return CchQuery(0.0, [s], s, [s], [t], 2, 0)
    chain_s, chain_t = cch.chain(s), cch.chain(t)
    df, pf = {s: 0.0}, {}
    db, pb = {t: 0.0}, {}
    relaxed = 0
    for v in chain_s:
        d = df.get(v)
        if d is None or d == INF:
            continue
        for e in cch.up[v]:
            relaxed += 1
            w = cch.hi[e]
            nd = d + m.fw[e]
            if nd < df.get(w, INF):
                df[w], pf[w] = nd, e
    for v in chain_t:
        d = db.get(v)
        if d is None or d == INF:
            continue
        for e in cch.up[v]:
            relaxed += 1
            w = cch.hi[e]
            nd = d + m.bw[e]
            if nd < db.get(w, INF):
                db[w], pb[w] = nd, e
    best, meet = INF, -1
    for x in df:
        if x in db and df[x] + db[x] < best:
            best, meet = df[x] + db[x], x
    steps = len(chain_s) + len(chain_t)
    if meet < 0:
        return CchQuery(INF, [], -1, chain_s, chain_t, steps, relaxed)
    left, cur = [], meet
    while cur != s:
        e = pf[cur]
        left.append(e)
        cur = cch.lo[e]
    right, cur = [], meet
    while cur != t:
        e = pb[cur]
        right.append(e)
        cur = cch.lo[e]
    route, used = [s], 0
    for e in reversed(left):                                        # s -> ... -> meet: jede Kante von unten nach oben
        used += _unpack(cch, m, e, True, route)
    for e in right:                                                 # meet -> ... -> t: jede Kante von oben nach unten
        used += _unpack(cch, m, e, False, route)
    return CchQuery(best, route, meet, chain_s, chain_t, steps, relaxed, used)


def cch_query_dijkstra(cch, m, s, t):
    """Abfrage mit zwei Warteschlangen: bidirektionales Dijkstra nur auf Kanten nach oben (vorwärts mit den Gewichten von unten nach oben, rückwärts mit denen von oben nach unten), jede Seite hört auf, wenn ihr kleinster Schlüssel
    die beste bisher gefundene Route erreicht. Festgelegte Knoten sind hier meist weniger als die ganzen Ketten des Eliminationsbaums, dafür braucht die Suche Warteschlangen."""
    s, t = int(s), int(t)
    if s == t:
        return CchQuery(0.0, [s], s, [s], [t], 2, 0)
    dist = [{s: 0.0}, {t: 0.0}]
    parent = [{}, {}]
    heaps = [[(0.0, s)], [(0.0, t)]]
    done = [set(), set()]
    weights = (m.fw, m.bw)
    mu, meet, relaxed, turn = INF, -1, 0, 0
    while True:
        alive = [side for side in (0, 1) if heaps[side] and heaps[side][0][0] < mu]
        if not alive:
            break
        side = turn if turn in alive else alive[0]
        turn = 1 - turn
        d, x = heapq.heappop(heaps[side])
        if x in done[side]:
            continue
        done[side].add(x)
        other = dist[1 - side].get(x)
        if other is not None and d + other < mu:
            mu, meet = d + other, x
        for e in cch.up[x]:
            relaxed += 1
            y = cch.hi[e]
            nd = d + weights[side][e]
            if nd < dist[side].get(y, INF):
                dist[side][y] = nd
                parent[side][y] = e
                heapq.heappush(heaps[side], (nd, y))
                other = dist[1 - side].get(y)
                if other is not None and nd + other < mu:
                    mu, meet = nd + other, y
    steps = len(done[0]) + len(done[1])
    if meet < 0:
        return CchQuery(INF, [], -1, sorted(done[0], key=lambda v: cch.rank[v]), sorted(done[1], key=lambda v: cch.rank[v]), steps, relaxed)
    left, cur = [], meet
    while cur != s:
        e = parent[0][cur]
        left.append(e)
        cur = cch.lo[e]
    right, cur = [], meet
    while cur != t:
        e = parent[1][cur]
        right.append(e)
        cur = cch.lo[e]
    route, used = [s], 0
    for e in reversed(left):
        used += _unpack(cch, m, e, True, route)
    for e in right:
        used += _unpack(cch, m, e, False, route)
    return CchQuery(mu, route, meet, sorted(done[0], key=lambda v: cch.rank[v]), sorted(done[1], key=lambda v: cch.rank[v]), steps, relaxed, used)
