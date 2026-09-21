"""Gerichteter Graph in CSR-Form (Kompaktzeilen): Nachbarn eines Knotens u stehen in indices[indptr[u]:indptr[u+1]], die Kosten der Kanten in weight."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Graph:
    n: int
    indptr: np.ndarray            # (n + 1,) int
    indices: np.ndarray           # (m,) int   Zielknoten je gerichteter Kante, je Knoten aufsteigend sortiert (bei Parallelkanten: billigste zuerst)
    weight: np.ndarray            # (m,) float Kosten je gerichteter Kante (negativ erlaubt, wenn der Graph so gebaut wurde)
    xy: np.ndarray                # (n, 2) Lage für die Zeichnung
    names: tuple = ()             # optionale Beschriftungen
    directed: bool = False

    @property
    def m(self):
        return len(self.indices)

    def out(self, u):
        return self.indices[self.indptr[u]:self.indptr[u + 1]]

    def out_weights(self, u):
        return self.weight[self.indptr[u]:self.indptr[u + 1]]

    def degree(self):
        return np.diff(self.indptr)

    def arc(self, u, v):
        """Index der (billigsten) Kante u -> v in indices (oder -1)."""
        lo, hi = self.indptr[u], self.indptr[u + 1]
        k = lo + int(np.searchsorted(self.indices[lo:hi], v))
        return k if k < hi and self.indices[k] == v else -1

    def has_negative(self):
        return bool((self.weight < 0).any())


def from_arcs(n, arcs, xy, names=(), directed=False, clean=True):
    """Baut den Graphen aus (u, v, Kosten)-Tupeln. Ungerichtet: jede Kante einmal angeben, die Rückrichtung entsteht hier.
    `clean=True`: Selbstschleifen fallen weg, von mehreren parallelen Kanten bleibt die billigste (so empfiehlt es das Buch Optimization Algorithms für OSM-Netze:
    die Länge einer Route hinge sonst davon ab, welche Parallelkante gewählt wird). `clean=False` behält Rohdaten unverändert (Mehrfachkanten und Schleifen)."""
    rows = []
    for u, v, w in arcs:
        u, v, w = int(u), int(v), float(w)
        if clean and u == v:
            continue
        rows.append((u, v, w))
        if not directed and u != v:
            rows.append((v, u, w))
    if clean:
        best = {}
        for u, v, w in rows:
            if (u, v) not in best or w < best[(u, v)]:
                best[(u, v)] = w
        rows = [(u, v, w) for (u, v), w in best.items()]
    rows.sort()
    src = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
    dst = np.fromiter((r[1] for r in rows), dtype=np.int64, count=len(rows))
    wts = np.fromiter((r[2] for r in rows), dtype=float, count=len(rows))
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return Graph(n, indptr, dst, wts, np.asarray(xy, dtype=float), tuple(names), directed)


def route_cost(g, route):
    """Summe der Kantenkosten entlang einer Knotenfolge (bei Parallelkanten die billigste); Kanten, die es nicht gibt, sind ein Fehler."""
    total = 0.0
    for u, v in zip(route[:-1], route[1:]):
        k = g.arc(u, v)
        if k < 0:
            raise ValueError(f"keine Kante {u} -> {v}")
        total += float(g.weight[k])
    return total


def reverse_graph(g):
    """Der Graph mit umgedrehten Kanten (von v nach u statt von u nach v, gleiche Kosten): die Rückwärtssuche vom Ziel läuft auf ihm wie eine Vorwärtssuche.
    Bei ungerichteten Graphen ist er gleich dem Original."""
    src = np.repeat(np.arange(g.n), g.degree())
    new_src, new_dst = g.indices, src
    order = np.lexsort((g.weight, new_dst, new_src))                                     # nach (Start, Ziel, Kosten): bei Parallelkanten steht die billigste vorn
    new_src, new_dst, w = new_src[order], new_dst[order], g.weight[order]
    indptr = np.zeros(g.n + 1, dtype=np.int64)
    np.cumsum(np.bincount(new_src, minlength=g.n), out=indptr[1:])
    return Graph(g.n, indptr, new_dst.astype(np.int64), w, g.xy, g.names, g.directed)
