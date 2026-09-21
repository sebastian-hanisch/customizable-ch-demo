"""Contraction Hierarchies als Vergleich: Vorberechnung (Knoten nach Wichtigkeit zusammenziehen, Abkürzungen mit Zeugensuche) und die CH-Abfrage als Vergleich. Kopie aus contraction-hierarchies-demo (Stück 4 der Linie,
Repos sind unabhängig); die Customizable CH steht in cc_cch.py.

Eigene Umsetzung auf dem CSR-Graphen aus cc_graph.py; networkx kommt nur in den Tests vor (Kreuzprobe)."""

import heapq
import time
from dataclasses import dataclass, field

import numpy as np

INF = float("inf")
ORDERS = ("edge_difference", "lazy_edge_difference", "degree", "random")
DEFAULT_WITNESS_LIMIT = 200       # festgelegte Knoten je Zeugensuche beim eigentlichen Zusammenziehen
DEFAULT_SIM_LIMIT = 50            # dasselbe beim Abschätzen der Wichtigkeit (Probelauf ohne Einfügen)


@dataclass
class Hierarchy:
    n: int
    rank: np.ndarray                            # Rang je Knoten: 0 = als erster zusammengezogen (unwichtigster), n - 1 = wichtigster
    order: list                                 # Knoten in der Reihenfolge des Zusammenziehens
    up: list                                    # up[v] = [(x, Kosten, Mittelknoten)]: Kanten von v zu höher gerankten Knoten (Vorwärtssuche)
    down_in: list                               # down_in[v] = [(u, Kosten, Mittelknoten)]: Kanten u -> v von höher gerankten u (Rückwärtssuche steigt von v nach u)
    arcs: dict                                  # (u, v) -> (Kosten, Mittelknoten): alle Kanten der Hierarchie (Original und Abkürzungen); Mittelknoten -1 = Originalkante
    counters: dict = field(default_factory=dict)
    log: list = field(default_factory=list)     # nur mit trace=True: je Zusammenziehung (Knoten, Rang, [(u, w, Kosten über v, Zeugenkosten oder None, eingefügt?)])
    seconds: float = 0.0
    params: dict = field(default_factory=dict)
    added: list = field(default_factory=list)   # eingefügte Abkürzungen je Zusammenziehung (in der Reihenfolge des Zusammenziehens)

    @property
    def n_original(self):
        return sum(1 for _, (_, mid) in self.arcs.items() if mid < 0)

    @property
    def n_shortcuts(self):
        return sum(1 for _, (_, mid) in self.arcs.items() if mid >= 0)


def _witness(out_adj, source, skip, limit_cost, limit_settled):
    """Begrenztes Dijkstra ab `source` im Restgraphen ohne `skip`: liefert die gefundenen Entfernungen und die Zahl festgelegter Knoten. Bricht ab, wenn der kleinste Schlüssel über `limit_cost` liegt
    (weiter weg wird nicht gebraucht) oder `limit_settled` Knoten festgelegt sind (dann fehlen Zeugen - das kostet höchstens Abkürzungen, nie Richtigkeit)."""
    dist = {source: 0.0}
    heap = [(0.0, source)]
    settled = 0
    while heap:
        d, x = heapq.heappop(heap)
        if d > dist[x]:
            continue
        if d > limit_cost:
            break
        settled += 1
        if settled > limit_settled:
            break
        for y, (w, _) in out_adj[x].items():
            if y == skip:
                continue
            nd = d + w
            if nd < dist.get(y, INF):
                dist[y] = nd
                heapq.heappush(heap, (nd, y))
    return dist, settled


def _plan_contraction(out_adj, in_adj, v, limit_settled):
    """Was passiert beim Zusammenziehen von v: je Paar u -> v -> w (u != w) die Abkürzung, die nötig wäre (Zeuge gefunden = keine). Rückgabe: (Liste (u, w, Kosten über v, Zeugenkosten oder None), festgelegte Knoten in Zeugensuchen, Zahl Suchen)."""
    plan, settled_total, searches = [], 0, 0
    outs = out_adj[v]
    if not outs:
        return plan, 0, 0
    max_out = max(w for w, _ in outs.values())
    for u, (cu, _) in in_adj[v].items():
        targets = [(w, cw) for w, (cw, _) in outs.items() if w != u]
        if not targets:
            continue
        dist, settled = _witness(out_adj, u, v, cu + max_out, limit_settled)
        settled_total += settled
        searches += 1
        for w, cw in targets:
            via = cu + cw
            wit = dist.get(w, INF)
            plan.append((u, w, via, wit if wit <= via else None))
    return plan, settled_total, searches


def _priority(order, out_adj, in_adj, v, contracted_neighbors, sim_limit):
    if order == "degree":
        return len(out_adj[v]) + len(in_adj[v]), 0
    plan, settled, _ = _plan_contraction(out_adj, in_adj, v, sim_limit)
    shortcuts = sum(1 for _, _, _, wit in plan if wit is None)
    return shortcuts - (len(in_adj[v]) + len(out_adj[v])) + contracted_neighbors[v], settled


def build_hierarchy(g, order="lazy_edge_difference", witness_limit=DEFAULT_WITNESS_LIMIT, sim_limit=DEFAULT_SIM_LIMIT, seed=0, trace=False):
    """Vorberechnung: die Knoten werden nacheinander zusammengezogen (aus dem Restgraphen entfernt); für jedes Paar Eingangs- und Ausgangsnachbar u -> v -> w, dessen kürzeste Route über v
    keinen Zeugen im Restgraphen hat, kommt eine Abkürzung u -> w (Kosten c(u,v) + c(v,w), Mittelknoten v) hinzu. Ein Zeuge ist ein Weg u -> w ohne v mit Kosten <= c(u,v) + c(v,w).
    Ordnungen: "edge_difference" (Wichtigkeit = Abkürzungen - Kanten + schon zusammengezogene Nachbarn; nach jeder Zusammenziehung neu für die Nachbarn), "lazy_edge_difference" (Wichtigkeit einmal am Anfang, beim Entnehmen
    neu geprüft: nur zusammenziehen, wenn sie nicht schlechter als die des nächsten ist), "degree" (Grad im Restgraphen), "random" (zufällige Reihenfolge).
    Die Richtigkeit hängt nicht von Ordnung und Zeugengrenze ab - nur Zahl der Abkürzungen, Vorberechnungszeit und Abfrageaufwand."""
    t0 = time.perf_counter()
    n = g.n
    out_adj = [dict() for _ in range(n)]
    in_adj = [dict() for _ in range(n)]
    for u in range(n):
        for v, w in zip(g.out(u).tolist(), g.out_weights(u).tolist()):
            if u != v and (v not in out_adj[u] or w < out_adj[u][v][0]):
                out_adj[u][v] = (w, -1)
                in_adj[v][u] = (w, -1)
    contracted_neighbors = [0] * n
    rank = np.full(n, -1, dtype=np.int64)
    up, down_in = [[] for _ in range(n)], [[] for _ in range(n)]
    arcs, log, added = {}, [], []
    counters = {"witness_searches": 0, "witness_settled": 0, "sim_settled": 0, "priority_evaluations": 0, "shortcuts_created": 0}
    rng = np.random.default_rng([int(seed), 1212])

    if order == "random":
        sequence = iter(int(x) for x in rng.permutation(n))
        heap = None
    else:
        heap, version = [], [0] * n
        for v in range(n):
            p, sset = _priority(order, out_adj, in_adj, v, contracted_neighbors, sim_limit)
            counters["priority_evaluations"] += 1
            counters["sim_settled"] += sset
            heap.append((p, v, 0))
        heapq.heapify(heap)

    def next_node():
        if heap is None:
            return next(sequence)
        while True:
            p, v, ver = heapq.heappop(heap)
            if rank[v] >= 0 or ver != version[v]:
                continue
            if order == "lazy_edge_difference":
                fresh, sset = _priority(order, out_adj, in_adj, v, contracted_neighbors, sim_limit)
                counters["priority_evaluations"] += 1
                counters["sim_settled"] += sset
                if heap and fresh > heap[0][0]:
                    version[v] += 1
                    heapq.heappush(heap, (fresh, v, version[v]))
                    continue
            return v

    for r in range(n):
        v = next_node()
        plan, settled, searches = _plan_contraction(out_adj, in_adj, v, witness_limit)
        counters["witness_searches"] += searches
        counters["witness_settled"] += settled
        entries = []
        # Kanten von und zu v festhalten: sie gehören zur Hierarchie (nach oben bzw. von oben nach unten)
        for x, (w, mid) in out_adj[v].items():
            up[v].append((x, w, mid))
            arcs[(v, x)] = (w, mid)
        for u, (w, mid) in in_adj[v].items():
            down_in[v].append((u, w, mid))
            arcs[(u, v)] = (w, mid)
        rank[v] = r
        neighbors = set(out_adj[v]) | set(in_adj[v])
        # Abkürzungen einfügen (bei mehreren Wegen zum selben Paar bleibt die billigste)
        before = counters["shortcuts_created"]
        for u, w, via, wit in plan:
            inserted = wit is None
            if inserted:
                cur = out_adj[u].get(w)
                if cur is None or via < cur[0]:
                    out_adj[u][w] = (via, v)
                    in_adj[w][u] = (via, v)
                    counters["shortcuts_created"] += 1
            if trace:
                entries.append((u, w, via, wit, inserted))
        added.append(counters["shortcuts_created"] - before)
        # v aus dem Restgraphen entfernen
        for x in list(out_adj[v]):
            del in_adj[x][v]
        for u in list(in_adj[v]):
            del out_adj[u][v]
        out_adj[v], in_adj[v] = {}, {}
        for x in neighbors:
            contracted_neighbors[x] += 1
        if trace:
            log.append((v, r, entries))
        if order == "edge_difference":
            for x in neighbors:
                p, sset = _priority(order, out_adj, in_adj, x, contracted_neighbors, sim_limit)
                counters["priority_evaluations"] += 1
                counters["sim_settled"] += sset
                version[x] += 1
                heapq.heappush(heap, (p, x, version[x]))
        elif order == "degree":
            for x in neighbors:
                p, _ = _priority(order, out_adj, in_adj, x, contracted_neighbors, sim_limit)
                version[x] += 1
                heapq.heappush(heap, (p, x, version[x]))
    order_list = np.argsort(rank).tolist()
    h = Hierarchy(n, rank, order_list, up, down_in, arcs, counters, log, time.perf_counter() - t0,
                  {"order": order, "witness_limit": witness_limit, "sim_limit": sim_limit, "seed": int(seed)}, added)
    h.counters["shortcuts"] = h.n_shortcuts
    h.counters["original_arcs"] = h.n_original
    return h


# --- Abfrage -----------------------------------------------------------------------------------------------------------------------------

@dataclass
class ChQuery:
    cost: float
    route: list                                 # ausgepackte Route (nur Originalkanten); leer, wenn keine
    packed: list                                # Route mit Abkürzungen: Knotenfolge s ... Spitze ... t
    peak: int                                   # Treffknoten (höchster Rang auf der Route)
    order_f: list = field(default_factory=list)        # vorwärts festgelegte Knoten
    order_b: list = field(default_factory=list)        # rückwärts festgelegte Knoten
    shortcuts_used: int = 0
    counters: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)          # je Festlegung (Seite "f"/"b", Knoten) in der Reihenfolge der Abfrage

    @property
    def settled(self):
        return len(self.order_f) + len(self.order_b)


def unpack_arc(h, a, b):
    """Kante (a, b) der Hierarchie als Folge von Originalkanten (ohne den Anfangsknoten a)."""
    out, stack = [], [(a, b)]
    while stack:
        x, y = stack.pop()
        _, mid = h.arcs[(x, y)]
        if mid < 0:
            out.append(y)
        else:
            stack.append((mid, y))
            stack.append((x, mid))
    return out


def ch_query(h, s, t):
    """Abfrage: bidirektionale Suche, vorwärts nur auf Kanten zu höherem Rang ab s, rückwärts nur auf Kanten von höherem Rang nach unten (also von t aus ebenfalls aufwärts). Jede Seite hört auf, wenn ihre Warteschlange leer
    ist oder ihr kleinster Schlüssel die beste bisher gefundene Route mu erreicht; mu = kleinste Summe d_f(x) + d_b(x) über Knoten x, die beide Seiten beschriftet haben. Danach wird die Route ausgepackt."""
    s, t = int(s), int(t)
    if s == t:
        return ChQuery(0.0, [s], [s], s, [], [], 0, {"pushes": 0, "relaxations": 0})
    dist = [{s: 0.0}, {t: 0.0}]
    parent = [{}, {}]
    heaps = [[(0.0, s)], [(0.0, t)]]
    done = [set(), set()]
    orders = [[], []]
    steps = []
    adj = (h.up, h.down_in)
    mu, peak = INF, -1
    pushes, relax, turn = 0, 0, 0
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
        orders[side].append(x)
        steps.append(("f" if side == 0 else "b", x))
        other = dist[1 - side].get(x)
        if other is not None and d + other < mu:
            mu, peak = d + other, x
        for y, w, mid in adj[side][x]:
            relax += 1
            nd = d + w
            if nd < dist[side].get(y, INF):
                dist[side][y] = nd
                parent[side][y] = (x, mid)
                heapq.heappush(heaps[side], (nd, y))
                pushes += 1
                other = dist[1 - side].get(y)
                if other is not None and nd + other < mu:
                    mu, peak = nd + other, y
    counters = {"pushes": pushes, "relaxations": relax}
    if peak < 0:
        return ChQuery(INF, [], [], -1, orders[0], orders[1], 0, counters, steps)
    left = [peak]
    while left[-1] in parent[0]:
        left.append(parent[0][left[-1]][0])
    left.reverse()
    right = [peak]
    while right[-1] in parent[1]:
        right.append(parent[1][right[-1]][0])
    packed = left + right[1:]
    route, used = [packed[0]], 0
    for a, b in zip(packed[:-1], packed[1:]):
        if h.arcs[(a, b)][1] >= 0:
            used += 1
        route.extend(unpack_arc(h, a, b))
    return ChQuery(mu, route, packed, peak, orders[0], orders[1], used, counters, steps)
