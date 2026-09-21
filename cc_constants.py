"""Konstanten und Grenzen der Regler. Die Zahlen in Hilfetexten und Tabellen der App sind in tests/test_claims.py belegt."""

SPACING = 100.0                    # Meter zwischen benachbarten Kreuzungen im erzeugten Stadtnetz
JITTER = 0.25                      # Lageabweichung der Kreuzungen in Blocklängen
SPREAD = 0.3                       # Kosten einer Straße = Länge * (1 + SPREAD * Zufall): Ampeln, Steigung, Belag
TRAFFIC_FACTOR = 3.0               # Verkehr: betroffene Straßen werden so viel teurer

NETS = ("small", "city", "random")
NET_LABELS = {
    "small": "🔀 Kleines Netz (zwei Stadtteile, zwei Brücken)",
    "city": "🏙️ Stadtnetz (erzeugt)",
    "random": "🕸️ Zufallsnetz (erzeugt)",
}
FIXED_NETS = ("small",)

SIDE_MIN, SIDE_MAX, DEFAULT_SIDE = 4, 20, 10               # n = Seite², höchstens 400 Knoten
NODES_MIN, NODES_MAX, DEFAULT_NODES = 20, 200, 100
DEGREE_MIN, DEGREE_MAX, DEFAULT_DEGREE = 2.0, 6.0, 3.0
TRAFFIC_MIN, TRAFFIC_MAX, DEFAULT_TRAFFIC = 0, 50, 10      # Anteil der Straßen im Stau in Prozent
DISTANCE_MIN, DISTANCE_MAX, DEFAULT_DISTANCE = 0, 100, 60  # Ziel bei diesem Prozentrang der Entfernungen vom Start
DEFAULT_SEED = 7
DEFAULT_NET = "small"

ORDER_LABELS = {"nd": "Verschachtelte Zerlegung (Koordinaten)", "mindeg": "Kleinster Grad", "ch": "Ordnung der Contraction Hierarchy", "random": "Zufall"}
DEFAULT_ORDER = "nd"

SWEEP_SEEDS = tuple(range(100000, 100005))
PAIR_SAMPLES = 100

COLORS = {"same": "#1f77b4", "changed": "#d62728", "todo": "#cccccc", "fill": "#ff7f0e", "route": "#1f77b4", "stale": "#7f7f7f", "start": "#111111", "goal": "#ff7f0e"}


_BASE = dict(side=DEFAULT_SIDE, nodes=DEFAULT_NODES, degree=DEFAULT_DEGREE, traffic=DEFAULT_TRAFFIC, distance=DEFAULT_DISTANCE, order=DEFAULT_ORDER, seed=DEFAULT_SEED)
PRESETS = {
    "🔀 Kleines Netz": {**_BASE, "net": "small"},
    "🏙️ Stadtnetz": {**_BASE, "net": "city", "seed": 3, "distance": 80},
    "🕸️ Zufallsnetz": {**_BASE, "net": "random", "seed": 2, "distance": 100, "order": "mindeg"},
    "🎲 Schlechte Ordnung": {**_BASE, "net": "city", "side": 14, "order": "random"},
}
PRESET_HELP = {
    "🔀 Kleines Netz": "Kleines Netz (10 Orte, 14 Straßen, Verschachtelte Zerlegung): die Hierarchie hat 22 Kanten (14 Straßen und 8 Abkürzungen) und 21 untere Dreiecke. Der Stau auf der nördlichen Brücke verändert 7 der 22 Kanten. Altstadt → Fabrik: vorher 14 min über die nördliche Brücke, jetzt 16 min über die südliche - die alte Contraction Hierarchy meldet weiter 14 min. Das teilweise Anpassen bearbeitet 17 statt 21 Dreiecke.",
    "🏙️ Stadtnetz": "Stadtnetz (10 × 10, Seed 3, 10 % der Straßen dreimal so teuer): 657 Kanten (180 Straßen), 2 348 Dreiecke; das Anpassen verändert 168 Kanten. Für das gezeigte Paar ändert sich die Route: vorher 797 m, jetzt 831 m - die alte CH meldet weiter 797 und liegt bei 49.5 % aller Paare falsch. Teilweises Anpassen: 1 839 statt 2 348 Dreiecke; die CH neu zu bauen kostet 4 682 festgelegte Knoten in Zeugensuchen.",
    "🕸️ Zufallsnetz": "Zufallsnetz (100 Knoten, Grad 3, Ordnung nach kleinstem Grad, Seed 2, 10 % der Straßen dreimal so teuer): 318 Kanten (150 Straßen), 626 Dreiecke. Das gezeigte Paar braucht jetzt 63 statt 36; die alte CH meldet weiter 36 und liegt bei 18.2 % aller Paare falsch. Teilweises Anpassen: 413 statt 626 Dreiecke.",
    "🎲 Schlechte Ordnung": "Stadtnetz 14 × 14 mit zufälliger Ordnung: 3 712 Kanten statt 1 670 mit der Verschachtelten Zerlegung und 62 450 statt 8 588 Dreiecke. Die Antworten sind dieselben - aber die Abfrage im Eliminationsbaum braucht für das gezeigte Paar 126 statt 38 Schritte (Dijkstra: 110), im Mittel über 100 Paare 124.4 statt 56.2 (Dijkstra: 97.3).",
}
