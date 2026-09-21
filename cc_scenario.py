"""Die Netze der Demo: kleines Netz (acht Orte, aus der CH-Demo übernommen), erzeugtes Stadtnetz und erzeugtes Zufallsnetz. Alle Kosten sind ganze Zahlen (Meter, Minuten, Einheiten): Gleichstände sind exakt, Zähler und Kosten plattformfest."""

from dataclasses import dataclass

import numpy as np

import cc_constants as C
from cc_graph import Graph, from_arcs, reverse_graph


@dataclass(frozen=True)
class Network:
    key: str
    graph: Graph
    title: str
    note: str = ""
    side: int = 0
    fixed_pair: tuple = ()         # (Start, Ziel) bei Netzen mit fester Aufgabe, sonst leer
    jam: frozenset = frozenset()   # feste Netze: Straßen (unten, oben) mit Stau
    geometric: bool = True         # Lage der Knoten ist eine Karte (Kanten zeichnen), sonst nur Punkte
    unit: str = ""


# --- Stadtnetz -------------------------------------------------------------------------------------------------------------------------------------

def build_city(side, seed):
    """Gestörtes Raster: Kreuzungen im Abstand SPACING mit Lageabweichung, jede mit ihren Nachbarn rechts und darüber verbunden. Kosten einer Straße = ihre Länge in Metern mal (1 + SPREAD * Zufall), ganzzahlig."""
    rng = np.random.default_rng([int(seed), 101])
    n = side * side
    ij = np.stack(np.divmod(np.arange(n), side), axis=1)
    xy = np.stack([ij[:, 1], ij[:, 0]], axis=1) * C.SPACING + rng.uniform(-C.JITTER, C.JITTER, (n, 2)) * C.SPACING
    arcs = []
    for i in range(side):
        for j in range(side):
            for di, dj in ((0, 1), (1, 0)):
                i2, j2 = i + di, j + dj
                if i2 < side and j2 < side:
                    u, v = i * side + j, i2 * side + j2
                    length = float(np.hypot(*(xy[u] - xy[v])))
                    arcs.append((u, v, max(1.0, float(np.rint(length * (1.0 + C.SPREAD * rng.random()))))))
    return from_arcs(n, arcs, xy)


def city_network(side, seed):
    return Network("city", build_city(int(side), int(seed)), "Stadtnetz", "Erzeugtes Stadtnetz: Kreuzungen auf einem gestörten Raster, jede mit den Nachbarn rechts und darüber verbunden; Kosten = Länge in Metern mit Zufallszuschlag. Der Verkehr macht einen Anteil der Straßen dreimal so teuer.", int(side), unit="m")


# --- Zufallsnetz -----------------------------------------------------------------------------------------------------------------------------------

def build_random(n, degree, seed):
    """Zusammenhängender Zufallsgraph: jeder Knoten hängt an einem früheren, dann kommen zufällige Kanten bis zum mittleren Grad `degree`. Ganzzahlige Kosten 1 bis 9. Die Kugeln um einen Knoten wachsen hier exponentiell (wenige Schritte bis überall hin)."""
    rng = np.random.default_rng([int(seed), 707])
    arcs = {(int(rng.integers(0, i)), i): float(rng.integers(1, 10)) for i in range(1, n)}
    target_m = int(round(n * degree / 2))
    while len(arcs) < target_m:
        u, v = (int(x) for x in rng.integers(0, n, 2))
        if u != v and (u, v) not in arcs and (v, u) not in arcs:
            arcs[(u, v)] = float(rng.integers(1, 10))
    xy = rng.random((n, 2)) * 1000.0
    return from_arcs(n, [(u, v, w) for (u, v), w in arcs.items()], xy)


def random_network(n, degree, seed):
    return Network("random", build_random(int(n), float(degree), int(seed)), "Zufallsnetz",
                   "Erzeugter Zufallsgraph: jeder Knoten hat im Mittel dieselbe Zahl Nachbarn, es gibt keine Karte - die Punkte sind zufällig verteilt und die Kanten nicht gezeichnet.", geometric=False, unit="Einheiten")


# --- Kleines Netz ----------------------------------------------------------------------------------------------------------------------------------

# zwei Stadtteile (West und Ost) und zwei Brücken dazwischen; Fahrminuten an den Strecken
SMALL_STOPS = [("Altstadt", 0.0, 1.5), ("Markt", 1.2, 2.7), ("Bahnhof", 1.2, 0.3), ("Campus", 2.4, 1.5), ("Brücke Nord", 4.2, 2.7), ("Brücke Süd", 4.2, 0.3),
               ("Hafen", 6.0, 1.5), ("Werft", 7.0, 2.7), ("Dorf", 7.0, 0.3), ("Fabrik", 8.2, 1.5)]
SMALL_LINKS = [("Altstadt", "Markt", 3), ("Altstadt", "Bahnhof", 3), ("Markt", "Campus", 2), ("Bahnhof", "Campus", 2), ("Markt", "Brücke Nord", 3), ("Campus", "Brücke Nord", 4), ("Campus", "Brücke Süd", 4),
               ("Bahnhof", "Brücke Süd", 4), ("Brücke Nord", "Hafen", 3), ("Brücke Süd", "Hafen", 4), ("Hafen", "Werft", 2), ("Hafen", "Dorf", 2), ("Werft", "Fabrik", 3), ("Dorf", "Fabrik", 3)]
SMALL_JAM = (("Markt", "Brücke Nord"), ("Campus", "Brücke Nord"), ("Brücke Nord", "Hafen"))       # Stau auf der nördlichen Brücke
SMALL_JAM_FACTOR = 4.0


def small_network():
    names = [s[0] for s in SMALL_STOPS]
    idx = {n: i for i, n in enumerate(names)}
    g = from_arcs(len(names), [(idx[a], idx[b], w) for a, b, w in SMALL_LINKS], [(s[1], s[2]) for s in SMALL_STOPS], names)
    jam = frozenset((min(idx[a], idx[b]), max(idx[a], idx[b])) for a, b in SMALL_JAM)
    return Network("small", g, "Kleines Netz", "Ein eigenes kleines Netz: zwei Stadtteile, dazwischen zwei Brücken (Fahrminuten an den Strecken). Im Stau auf der nördlichen Brücke werden die drei Strecken dort viermal so langsam - "
                   "die Ordnung und die Struktur der Hierarchie bleiben, nur die Gewichte ändern sich.", fixed_pair=(idx["Altstadt"], idx["Fabrik"]), unit="min", jam=jam)


def traffic_weights(net, fraction, seed=C.DEFAULT_SEED, factor=C.TRAFFIC_FACTOR):
    """Kosten je Kante nach dem Verkehr: bei festen Netzen die Strecken der Staustelle (mal `SMALL_JAM_FACTOR`), sonst ein zufälliger Anteil der Straßen (mal `factor`, gerundet); beide Richtungen einer Straße gleich."""
    g = net.graph
    src = np.repeat(np.arange(g.n), g.degree())
    lo, hi = np.minimum(src, g.indices), np.maximum(src, g.indices)
    w = g.weight.copy()
    if net.jam:
        mask = np.array([(int(a), int(b)) in net.jam for a, b in zip(lo, hi)], dtype=bool)
        w[mask] = np.maximum(1.0, np.rint(w[mask] * SMALL_JAM_FACTOR))
        return w
    rng = np.random.default_rng([int(seed), 1313])
    key = lo * g.n + hi
    uniq = np.unique(key)
    hit = set(uniq[rng.random(len(uniq)) < fraction].tolist())
    mask = np.array([k in hit for k in key.tolist()], dtype=bool)
    w[mask] = np.maximum(1.0, np.rint(w[mask] * factor))
    return w


def make_network(net, side=C.DEFAULT_SIDE, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, seed=C.DEFAULT_SEED):
    if net == "small":
        return small_network()
    if net == "city":
        return city_network(side, seed)
    if net == "random":
        return random_network(nodes, degree, seed)
    raise ValueError(net)
