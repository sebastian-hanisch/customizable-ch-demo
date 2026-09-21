"""Jede Zahl aus Texten, Hilfen und README ist hier belegt (gemessen am 2026-09-21, Toleranzen fangen Rundung ab). Kanten, Dreiecke, festgelegte Knoten, Schritte und ganzzahlige Kosten sind plattformfest
(reine Python-Rechnung mit festen Seeds); Laufzeiten stehen in der App nur als Messwerte und werden hier nie geprüft."""

import pytest

import cc_constants as C
import cc_evaluation as ev
import cc_scenario as sc

PRESET = {"small": "🔀 Kleines Netz", "city": "🏙️ Stadtnetz", "random": "🕸️ Zufallsnetz", "bad": "🎲 Schlechte Ordnung"}


def _preset(key):
    p = C.PRESETS[PRESET[key]]
    net = sc.make_network(p["net"], p["side"], p["nodes"], p["degree"], p["seed"])
    return p, ev.analyse(net, p["order"], p["traffic"], p["distance"], p["seed"])


def _has(key, *needles):
    help_ = C.PRESET_HELP[PRESET[key]]
    for n in needles:
        assert n in help_, (key, n)


# --- Preset-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_small_preset_numbers():
    _, a = _preset("small")
    m, g = a.metrics, a.net.graph
    assert (m["n"], m["arcs"], m["edges"], m["original_edges"], m["fill_edges"], m["triangles"], m["changed_edges"], m["tri_full"], m["tri_part"]) == (10, 14, 22, 14, 8, 21, 7, 21, 17)
    assert (m["dist_old"], m["dist_stale"], m["dist"]) == (14.0, 14.0, 16.0) and [g.names[v] for v in a.q_old.route] == ["Altstadt", "Markt", "Brücke Nord", "Hafen", "Werft", "Fabrik"]
    assert [g.names[v] for v in a.q.route] == ["Altstadt", "Bahnhof", "Brücke Süd", "Hafen", "Werft", "Fabrik"] and a.metrics["stale_wrong"]
    _has("small", "10 Orte, 14 Straßen", "22 Kanten (14 Straßen und 8 Abkürzungen)", "21 untere Dreiecke", "7 der 22 Kanten", "vorher 14 min", "jetzt 16 min", "meldet weiter 14 min", "17 statt 21 Dreiecke")


def test_city_preset_numbers():
    p, a = _preset("city")
    m = a.metrics
    assert (p["seed"], p["distance"], p["traffic"], m["n"], m["arcs"], m["edges"], m["triangles"], m["changed_edges"], m["tri_part"], m["witness_settled"]) == (3, 80, 10, 100, 180, 657, 2348, 168, 1839, 4682)
    assert (m["dist_old"], m["dist_stale"], m["dist"]) == (797.0, 797.0, 831.0) and m["pairs"]["wrong_stale"] == pytest.approx(0.495, abs=0.0005) and m["pairs"]["wrong_cch"] == 0
    _has("city", "10 × 10, Seed 3", "10 % der Straßen dreimal so teuer", "657 Kanten (180 Straßen)", "2 348 Dreiecke", "168 Kanten", "vorher 797 m", "jetzt 831 m", "meldet weiter 797", "49.5 %", "1 839 statt 2 348", "4 682")


def test_random_preset_numbers():
    p, a = _preset("random")
    m = a.metrics
    assert (p["order"], p["seed"], p["distance"], m["n"], m["arcs"], m["edges"], m["triangles"], m["tri_part"]) == ("mindeg", 2, 100, 100, 150, 318, 626, 413)
    assert (m["dist_old"], m["dist_stale"], m["dist"]) == (36.0, 36.0, 63.0) and m["pairs"]["wrong_stale"] == pytest.approx(0.182, abs=0.0005)
    _has("random", "100 Knoten, Grad 3", "kleinstem Grad", "Seed 2", "318 Kanten (150 Straßen)", "626 Dreiecke", "63 statt 36", "meldet weiter 36", "18.2 %", "413 statt 626")


def test_bad_order_preset_numbers():
    p, a = _preset("bad")
    good = ev.analyse(sc.city_network(14, 7), "nd", 10, 60, 7)
    m, mg = a.metrics, good.metrics
    assert p["order"] == "random" and m["n"] == 196 and (m["edges"], mg["edges"], m["triangles"], mg["triangles"]) == (3712, 1670, 62450, 8588)
    assert (m["steps_tree"], mg["steps_tree"], m["steps_dijkstra"]) == (126, 38, 110) and m["dist"] == mg["dist"]                                              # dieselbe Antwort
    assert (m["pairs"]["tree"], mg["pairs"]["tree"], m["pairs"]["dijkstra"]) == pytest.approx((124.4, 56.2, 97.3), abs=0.05) and m["pairs"]["tree"] > m["pairs"]["dijkstra"]
    _has("bad", "14 × 14", "3 712 Kanten statt 1 670", "62 450 statt 8 588", "126 statt 38", "Dijkstra: 110", "124.4 statt 56.2", "97.3")


# --- Sidebar-Hilfen -----------------------------------------------------------------------------------------------------------------------------

def test_order_and_size_help_numbers():
    rows = {(r["key"], r["size"]): r for r in ev.edge_rows()}
    big, rnd = rows[("city", 20)], rows[("random", 200)]
    assert (big["nd"], big["mindeg"], big["ch"]) == pytest.approx((4358.0, 3530.4, 11646.2), abs=0.5) and (rnd["nd"], rnd["mindeg"], rnd["ch"]) == pytest.approx((3389.6, 904.2, 1225.0), abs=0.5)
    assert [rows[("city", s)]["nd"] for s in (6, 10, 14, 20)] == pytest.approx([152.8, 663.0, 1673.8, 4358.0], abs=0.5)
    assert [rows[("city", s)]["ch_hier"] for s in (6, 10, 14, 20)] == pytest.approx([95.4, 335.4, 776.4, 1765.8], abs=0.5)
    assert all(rows[("city", s)]["nd"] > rows[("city", s)]["ch_hier"] for s in (6, 10, 14, 20))                  # die CCH hat immer mehr Kanten als die CH


def test_traffic_help_numbers():
    rows = {r["fraction"]: r for r in ev.change_rows()}
    assert [rows[f]["wrong"] for f in (1, 2, 5, 10, 20, 50)] == pytest.approx([0.072, 0.216, 0.451, 0.689, 0.862, 0.984], abs=0.0005)          # Hilfe: 7 / 22 / 45 / 69 / 86 / 98 %


# --- Experimente und die Tabelle "Wo die Annahmen enden" ------------------------------------------------------------------------------------------

def test_edge_experiment_numbers():
    rows = {(r["key"], r["size"]): r for r in ev.edge_rows()}
    big = rows[("city", 20)]
    assert big["nd"] / big["ch_hier"] == pytest.approx(2.47, abs=0.005) and big["mindeg"] / big["ch_hier"] == pytest.approx(2.0, abs=0.005)               # Tabelle: 2.5-mal und 2.0-mal so viele
    assert big["ch"] / big["ch_hier"] == pytest.approx(6.6, abs=0.05) and big["nd_tri"] == pytest.approx(32051.0, abs=0.5) and big["mindeg_tri"] == pytest.approx(22373.0, abs=0.5)
    assert rows[("random", 200)]["nd"] > 3.5 * rows[("random", 200)]["mindeg"] and rows[("random", 100)]["nd"] > 2 * rows[("random", 100)]["mindeg"]      # Koordinaten taugen im Zufallsnetz nichts
    assert rows[("city", 20)]["nd"] < rows[("city", 20)]["ch"] and rows[("city", 14)]["nd"] < rows[("city", 14)]["random"] and rows[("city", 20)]["nd_height"] == pytest.approx(57.2, abs=0.05)
    bad = ev.analyse(sc.city_network(14, 7), "random", 10, 60, 7).metrics["edges"] / ev.analyse(sc.city_network(14, 7), "nd", 10, 60, 7).metrics["edges"]
    assert bad == pytest.approx(2.22, abs=0.005)                                                                                                     # Tabelle: 2.2-mal so groß


def test_change_experiment_numbers():
    rows = {r["fraction"]: r for r in ev.change_rows()}
    assert all(r["tri_full"] == pytest.approx(32051.0, abs=0.5) for r in rows.values())
    assert [rows[f]["tri_part"] for f in (1, 2, 5, 10, 20, 50)] == pytest.approx([11242.8, 19846.0, 26830.6, 29833.6, 31192.0, 31932.0], abs=0.5)
    assert [1 - rows[f]["tri_part"] / rows[f]["tri_full"] for f in (1, 5, 20)] == pytest.approx([0.649, 0.163, 0.027], abs=0.0005)                         # Tabelle: 65 / 16 / 3 %
    assert rows[1]["witness"] == pytest.approx(41547.0, abs=0.5) and all(37000 < r["witness"] < 43000 for r in rows.values())                             # Neuaufbau: fast unabhängig vom Anteil
    assert [rows[f]["edges_part"] for f in (1, 50)] == pytest.approx([719.0, 3946.4], abs=0.5) and rows[1]["edges"] == 4358


def test_query_experiment_numbers():
    rows = {(r["key"], r["size"]): r for r in ev.query_rows()}
    big = rows[("city", 20)]
    assert (big["dijkstra"], big["ch"], big["cch_dijkstra"], big["tree"]) == pytest.approx((210.9, 53.1, 66.5, 88.0), abs=0.05)                            # Tabelle: 67 gegen 53, Baum 88
    assert all(r["ch"] < r["cch_dijkstra"] < r["dijkstra"] and r["cch_dijkstra"] < r["tree"] for r in rows.values())
    assert rows[("city", 6)]["dijkstra"] == pytest.approx(20.3, abs=0.05) and rows[("city", 6)]["tree"] > rows[("city", 6)]["dijkstra"] * 0.99          # im kleinsten Netz lohnt der Baum nicht


def test_after_customization_every_answer_is_right_and_the_stale_ch_is_not():
    for key in PRESET:
        _, a = _preset(key)
        m = a.metrics
        assert m["exact"] and m["pairs"]["wrong_cch"] == 0 and m["pairs"]["wrong_fresh"] == 0 and m["pairs"]["wrong_stale"] > 0.1, key
