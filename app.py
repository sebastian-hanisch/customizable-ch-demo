"""Customizable Contraction Hierarchies - Ordnung einmal, Kosten je Verkehrslage - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo EIN Verfahren - die Customizable Contraction Hierarchy - und lässt stattdessen das Beispiel wachsen.
Elftes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe, Kind von Contraction Hierarchies: die CH wird ungültig, wenn sich die Kosten ändern - die CCH trennt die Form des Netzes (einmal) von den Kosten (je Verkehrslage neu eintragen).
Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time

import numpy as np
import pandas as pd
import streamlit as st

import cc_cch as cch_mod
import cc_ch as ch
import cc_constants as C
import cc_evaluation as ev
from cc_presets import (
    KEPT,
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from cc_scenario import make_network
from cc_visualization import (
    build_change,
    build_edges,
    build_network,
    build_query,
    build_wrong,
    edge_table,
    name_of,
    step_table,
)

st.set_page_config(page_title="Customizable CH – Sebastian Hanisch", layout="wide")


def _num(x):
    return f"{x:,.0f}".replace(",", ".")


@st.cache_resource(show_spinner=False, max_entries=8)
def _network(params):
    return make_network(*params)


@st.cache_resource(show_spinner=False, max_entries=8)
def _built(params, order):
    return ev.build(_network(params), order, params[-1])


@st.cache_resource(show_spinner=False, max_entries=16)
def _analysis(params, order, traffic, distance):
    return ev.analyse(_network(params), order, traffic, distance, params[-1], built=_built(params, order))


@st.cache_data(show_spinner=False)
def _edges():
    return ev.edge_rows()


@st.cache_data(show_spinner=False)
def _changes():
    return ev.change_rows()


@st.cache_data(show_spinner=False)
def _queries():
    return ev.query_rows()


st.title("🔧 Customizable Contraction Hierarchies – Ordnung einmal, Kosten je Verkehrslage")
st.markdown(
    """
Eine Contraction Hierarchy ist schnell, aber **an ihre Kosten gebunden**: welche Abkürzungen entstehen, entscheidet die Zeugensuche - und die hängt von den Kosten ab. Ändert sich der Verkehr, ist die Hierarchie **veraltet** und muss neu gebaut werden.
Die **Customizable CH** trennt beides: die Knotenordnung und das **Zusammenziehen** hängen nur von der **Form des Netzes** ab (ohne Zeugensuche, mit mehr Kanten), die Kosten werden in einer schnellen dritten Phase, dem **Anpassen**, von unten nach oben eingetragen -
über die **unteren Dreiecke** jeder Kante: $w(u,v) = \\min\\bigl(w(u,v),\\ w(u,x) + w(x,v)\\bigr)$. Nach einer Kostenänderung wird nur diese Phase wiederholt.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren vergleichen, zeigt diese Demo - elftes Stück der Kürzeste-Wege-Linie der \"Konzepte\"-Reihe, Kind von Contraction Hierarchies - **ein** Verfahren an einem wachsenden Beispiel. "
    "Das Verfahren geht auf Dibbelt, Strasser und Wagner (2016) zurück, die verschachtelte Zerlegung auf George (1973); alle Netze, Verkehrslagen und Zahlen dieser Demo sind eigene Konstruktionen und Messungen."
)

with st.expander("So funktioniert die Customizable CH", expanded=True):
    st.markdown(
        """
1. **Ordnung ohne Kosten:** die Knoten werden nach Wichtigkeit geordnet, und zwar nur aus der Form des Netzes. Hier: **verschachtelte Zerlegung** - das Netz wird an der Mitte geteilt, die Knoten an der Trennlinie (kleinere Randmenge) sind die wichtigsten; beide Hälften werden rekursiv ebenso geordnet. Zum Vergleich: kleinster Grad, die Ordnung einer Contraction Hierarchy und Zufall.
2. **Zusammenziehen ohne Kosten:** die Knoten werden von unten nach oben zusammengezogen; die höher geordneten Nachbarn eines Knotens werden **alle paarweise verbunden** - ohne Zeugensuche. Ergebnis: Kanten (Straßen und Abkürzungen) und für jede Kante ihre **unteren Dreiecke** $(x, u, v)$: $x$ liegt unter $u$ und $v$ und ist mit beiden verbunden.
3. **Anpassen:** die Gewichte werden von unten nach oben eingetragen: erst das Gewicht der Straße, dann das Minimum mit $w(u,x) + w(x,v)$ über alle unteren Dreiecke. Die Kanten der Dreiecke liegen tiefer und sind schon angepasst.
4. **Teilweises Anpassen:** ändern sich nur einige Straßen, werden nur die Kanten neu berechnet, deren Gewicht oder deren Dreiecke sich ändern - von unten nach oben, solange sich Gewichte ändern.
5. **Abfrage:** von $s$ und $t$ aus je die Kette der Vorfahren im **Eliminationsbaum** (Elternknoten = niedrigster höherer Nachbar) entlang, nur Kanten nach oben; oder eine bidirektionale Aufwärtssuche mit Warteschlange. Die Route wird über die Dreiecke ausgepackt.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielnetz laden:")
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    net_key = st.selectbox(
        "Netz", C.NETS, key="net_select", format_func=lambda k: C.NET_LABELS[k],
        help="Klein und fest (zwei Stadtteile, zwei Brücken, Stau auf der nördlichen) oder erzeugt (Stadtnetz auf einem gestörten Raster, Zufallsnetz). Alle Kosten sind ganze Zahlen; höchstens 400 Knoten.",
    )
    order = st.selectbox("Ordnung ohne Kosten", tuple(C.ORDER_LABELS), key="order_select", format_func=lambda k: C.ORDER_LABELS[k],
                         help="Welche Knoten oben stehen. Die Antworten hängen nicht davon ab - nur die Zahl der Kanten und der Dreiecke. Kanten im 20 × 20-Stadtnetz (Mittel über fünf Netze): 4 358 mit der Verschachtelten Zerlegung, 3 530 nach kleinstem Grad, 11 646 mit der Ordnung der Contraction Hierarchy. "
                              "Im Zufallsnetz mit 200 Knoten: 3 390 / 904 / 1 225 - dort taugt eine Zerlegung nach Koordinaten nichts, weil die Lage der Knoten nichts über das Netz sagt.")
    if net_key == "city":
        side = st.slider("Kreuzungen je Seite", *bounds("side_slider"), key="side_slider",
                         help="Größe des Rasters: n = Seite² Knoten. Kanten der CCH (Verschachtelte Zerlegung, Mittel über fünf Netze) bei 6 / 10 / 14 / 20 Kreuzungen je Seite: 153 / 663 / 1 674 / 4 358; die Contraction Hierarchy braucht dort 95 / 335 / 776 / 1 766.")
        st.session_state[KEPT["side_slider"]] = side
    else:
        side = int(st.session_state.get(KEPT["side_slider"], C.DEFAULT_SIDE))
    if net_key == "random":
        nodes = st.slider("Knoten", *bounds("nodes_slider"), key="nodes_slider", step=10, help="Anzahl der Knoten n; höchstens 200, weil eine schlechte Ordnung im Zufallsnetz schnell zu sehr vielen Kanten führt.")
        st.session_state[KEPT["nodes_slider"]] = nodes
        degree = st.slider("Mittlerer Grad", *bounds("degree_slider"), key="degree_slider", step=0.5, help="Kanten je Knoten.")
        st.session_state[KEPT["degree_slider"]] = degree
    else:
        nodes = int(st.session_state.get(KEPT["nodes_slider"], C.DEFAULT_NODES))
        degree = float(st.session_state.get(KEPT["degree_slider"], C.DEFAULT_DEGREE))
    if net_key in ("city", "random"):
        traffic = st.slider("Straßen im Stau [%]", *bounds("traffic_slider"), key="traffic_slider", step=5,
                            help="Dieser Anteil der Straßen wird dreimal so teuer (beide Richtungen). Anteil falscher Antworten einer alten Contraction Hierarchy im 20 × 20-Stadtnetz (Mittel über fünf Netze) bei 1 / 2 / 5 / 10 / 20 / 50 %: 7 / 22 / 45 / 69 / 86 / 98 %. Mit 0 ändert sich nichts.")
        st.session_state[KEPT["traffic_slider"]] = traffic
        distance = st.slider("Entfernung des Paares [Perzentil]", *bounds("distance_slider"), key="distance_slider", step=5,
                             help="Das Ziel liegt so weit vom Start entfernt, wie es dem Perzentil aller Entfernungen von diesem Start entspricht: 0 = der nächste Knoten, 100 = der am weitesten entfernte.")
        st.session_state[KEPT["distance_slider"]] = distance
        seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)
        st.session_state[KEPT["seed_input"]] = seed
        st.button("🎲 Neues Netz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Zufalls-Seed für das Netz.")
    else:
        traffic = int(st.session_state.get(KEPT["traffic_slider"], C.DEFAULT_TRAFFIC))
        distance = int(st.session_state.get(KEPT["distance_slider"], C.DEFAULT_DISTANCE))
        seed = int(st.session_state.get(KEPT["seed_input"], C.DEFAULT_SEED))
        st.caption("Dieses Netz ist fest - Start Altstadt, Ziel Fabrik, Stau auf der nördlichen Brücke.")

# nicht zum Netz gehörende Regler ändern das Netz nicht: sonst würden gleiche Netze unter verschiedenen Schlüsseln mehrfach berechnet
d = dict(side=C.DEFAULT_SIDE, nodes=C.DEFAULT_NODES, degree=C.DEFAULT_DEGREE, seed=C.DEFAULT_SEED)
if net_key == "city":
    d.update(side=int(side), seed=int(seed))
elif net_key == "random":
    d.update(nodes=int(nodes), degree=round(float(degree), 1), seed=int(seed))
params = (net_key, d["side"], d["nodes"], d["degree"], d["seed"])
gen = net_key in ("city", "random")
traffic_pct = int(traffic) if gen else C.DEFAULT_TRAFFIC
dist_pct = int(distance) if gen else C.DEFAULT_DISTANCE
sync_query_params({"net_select": net_key, "order_select": order, "side_slider": int(side), "nodes_slider": int(nodes), "degree_slider": round(float(degree), 1), "traffic_slider": int(traffic), "distance_slider": int(distance), "seed_input": int(seed)})
with st.spinner("Rechne ..."):
    a = _analysis(params, order, traffic_pct, dist_pct)
net, m, g, cch = a.net, a.metrics, a.net.graph, a.cch
small = bool(g.names)
frame_list = ev.frames(a)
last_step = len(frame_list) - 1
view_key = (params, order, traffic_pct, dist_pct)
if st.session_state.get("cc_step_owner") != view_key:
    st.session_state["cc_step_owner"] = view_key
    st.session_state["cc_step"] = last_step

# --- Customizable CH in Aktion ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Customizable CH in Aktion")
st.caption("Nach dem Verkehr: die Struktur (Ordnung, Kanten, Dreiecke) bleibt, die Gewichte werden neu eingetragen - Kante für Kante von unten nach oben. Rot: der Verkehr hat das Gewicht dieser Kante verändert.")
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.slider("Bearbeitete Kanten", 0, last_step, key="cc_step",
                     help="Wie viele Kanten das Anpassen schon bearbeitet hat (von den unteren zu den oberen): 0 = noch keine, ganz rechts = alle, die Routen stehen. Bei vielen Kanten zeigt die Ansicht etwa 60 Zwischenstände.")
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")
view_slot = st.empty()


def _render(current):
    k = frame_list[current]
    final = k >= len(a.m1.trace)
    with view_slot.container():
        st.plotly_chart(build_network(a, k, height=420 if small else 480), width="stretch", key=f"net_chart_{current}")
        if k > 0:
            st.dataframe(pd.DataFrame(step_table(a, k)), hide_index=True, width="stretch")
        if final and m["reachable"]:
            unit = net.unit
            route = " → ".join(name_of(g, v) for v in a.q.route) if small else f"{m['hops']} Kanten"
            st.markdown(f"**Beste Route jetzt:** {m['dist']:g} {unit} ({route}); mit den alten Kosten waren es {m['dist_old']:g} {unit}.")


if auto_play:
    n_frames = min(last_step + 1, 40)
    for kk in sorted({int(round(x)) for x in np.linspace(0, last_step, n_frames)}):
        _render(kk)
        time.sleep(min(0.7, 6.0 / n_frames))
    step = last_step
else:
    _render(step)
if small:
    st.markdown("**Alle Kanten der Hierarchie** (Gewicht alt → neu, mit dem Ort des Dreiecks, über den es kommt):")
    st.dataframe(pd.DataFrame(edge_table(a)), hide_index=True, width="stretch")
st.caption(net.note)

st.markdown("---")

# --- Nach dem Verkehr ---------------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Nach dem Verkehr")
st.caption("**Zähler** = Schritte und Kanten: Kanten der Hierarchie, bearbeitete Dreiecke, festgelegte Knoten in Zeugensuchen. Sie sind plattformfest. Laufzeiten stehen nur als Messwerte im Vergleich unten.")
code = ev.verdict(a)
ps = m["pairs"]
if code == "unreachable":
    st.warning("⚠️ Das Ziel ist vom Start aus nicht erreichbar.")
else:
    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Kanten der CCH", _num(m["edges"]), delta=f"CH: {_num(m['ch_edges'])}", delta_color="off",
              help=f"Kanten ohne Richtung: {_num(m['original_edges'])} Straßen und {_num(m['fill_edges'])} Abkürzungen. Sie hängen nur von der Form des Netzes ab. Die Contraction Hierarchy braucht Straßen plus ihre Abkürzungen.")
    p2.metric("Anpassen", f"{_num(m['tri_full'])} Dreiecke", delta=f"teilweise: {_num(m['tri_part'])} ({_num(m['edges_part'])} Kanten)", delta_color="off",
              help=f"Alle unteren Dreiecke bei vollständigem Anpassen; beim teilweisen Anpassen nur die der Kanten, die sich ändern (hier {_num(m['touched'])} Straßen direkt, {_num(m['edges_part'])} Kanten insgesamt neu berechnet). Das Ergebnis ist genau dasselbe.")
    p3.metric("CH neu bauen", f"{_num(m['witness_settled'])} Knoten", delta=f"in Zeugensuchen, {_num(m['ch_shortcuts_new'])} Abkürzungen", delta_color="off", help="Festgelegte Knoten der Zeugensuchen beim Neuaufbau der Contraction Hierarchy mit den neuen Kosten.")
    p4.metric("Alte CH nach dem Verkehr", f"{ps['wrong_stale']:.0%} falsch", delta=f"CCH: {ps['wrong_cch']} falsch", delta_color="off", help=f"Anteil der Zufallspaare ({ps['n']}), für die die veraltete Contraction Hierarchy eine falsche Entfernung liefert; die angepasste CCH stimmt in jedem Fall.")
    if code == "nochange":
        st.info("ℹ️ Es sind keine Straßen im Stau: die Kosten haben sich nicht geändert, das Anpassen liefert dieselben Gewichte. Erhöhen Sie den Anteil der Straßen im Stau.")
    elif code == "same":
        st.info(f"ℹ️ Für dieses Paar bleibt die beste Route gleich ({m['dist']:g} {net.unit}); die alte Contraction Hierarchy hätte hier zufällig recht. Über alle {ps['n']} Zufallspaare liegt sie in {ps['wrong_stale']:.0%} falsch - wählen Sie ein anderes Paar (Entfernung, Seed).")
    else:
        st.success(f"✅ **Der Verkehr ändert die beste Route:** {m['dist_old']:g} → {m['dist']:g} {net.unit}. Die alte Contraction Hierarchy meldet weiter {m['dist_stale']:g} {net.unit} und liegt bei {ps['wrong_stale']:.0%} der Zufallspaare falsch. "
                   f"Nach dem Anpassen ({_num(m['tri_full'])} Dreiecke, teilweise {_num(m['tri_part'])}) ist die CCH für alle {ps['n']} Paare richtig; die CH müsste mit {_num(m['witness_settled'])} festgelegten Knoten in Zeugensuchen neu gebaut werden.")

st.markdown("---")

# --- Vergleich -------------------------------------------------------------------------------------------------------------------------------------

with st.expander("🔧 Wie wir das erreichen – Anpassen gegen Neuaufbau, Abfrage im Vergleich"):
    if m["reachable"]:
        t0 = time.perf_counter()
        ev.dijkstra(ev.with_weights(g, a.weights), a.s, a.t)
        t_dij = time.perf_counter() - t0
        t0 = time.perf_counter()
        ch.ch_query(a.h2, a.s, a.t)
        t_ch = time.perf_counter() - t0
        t0 = time.perf_counter()
        cch_mod.cch_query_dijkstra(a.cch, a.m1, a.s, a.t)
        t_cd = time.perf_counter() - t0
        st.table({"Nach einer Kostenänderung": ["CH neu bauen", "CCH vollständig anpassen", "CCH teilweise anpassen"],
                  "Zähler": [f"{_num(m['witness_settled'])} festgelegte Knoten in Zeugensuchen", f"{_num(m['tri_full'])} Dreiecke", f"{_num(m['tri_part'])} Dreiecke, {_num(m['edges_part'])} Kanten"],
                  "Laufzeit [ms]": [f"{a.seconds['rebuild'] * 1000:.1f}", f"{a.seconds['customize'] * 1000:.1f}", f"{a.seconds['recustomize'] * 1000:.1f}"]})
        st.table({"Abfrage": ["Dijkstra", "CH (neu gebaut)", "CCH: Eliminationsbaum", "CCH: Aufwärtssuche"],
                  "Dieses Paar": [f"{m['steps_dijkstra']}", f"{m['steps_ch']}", f"{m['steps_tree']}", f"{m['steps_cchdij']}"],
                  "Mittel (100 Paare) / Laufzeit [ms]": [f"{ps['dijkstra']:.1f} / {t_dij * 1000:.3f}", f"{ps['ch']:.1f} / {t_ch * 1000:.3f}", f"{ps['tree']:.1f} / {a.seconds['query'] * 1000:.3f}", f"{ps['cch_dijkstra']:.1f} / {t_cd * 1000:.3f}"]})
        st.caption(f"Einmalig (ohne Kosten): Ordnung {a.seconds['order'] * 1000:.1f} ms, Zusammenziehen {a.seconds['contract'] * 1000:.1f} ms ({_num(m['edges'])} Kanten, {_num(m['triangles'])} Dreiecke, Eliminationsbaum der Höhe {m['height']}). "
                   "Die Laufzeiten sind Messwerte dieses Laufs (reines Python, ein Lauf, schwankend); verglichen werden die Zähler. Schritte der Abfrage: festgelegte Knoten bei Dijkstra, CH und Aufwärtssuche, bearbeitete Knoten beider Ketten beim Eliminationsbaum. "
                   f"Alle Antworten stimmen mit Dijkstra überein (Zufallspaare: {ps['wrong_cch']} Abweichungen der CCH, {ps['wrong_fresh']} der neu gebauten CH).")

st.markdown("---")

# --- Experimente -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie viele Kanten braucht die CCH?")
if st.button("Kanten gegen Netzgröße und Ordnung messen (dauert einen Moment)", key="edges_start"):
    st.session_state["edges_on"] = True
if st.session_state.get("edges_on"):
    with st.spinner("Rechne 6 Netze × 5 Seeds × 4 Ordnungen ..."):
        erows = _edges()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_edges(erows), width="stretch", key="edges_chart")
    c2.table({"Netz": [f"{'Stadt' if r['key'] == 'city' else 'Zufall'} {r['n']:.0f}" for r in erows], "CH": [_num(r["ch_hier"]) for r in erows], "CCH (Zerlegung)": [_num(r["nd"]) for r in erows]})
    big, rnd = erows[3], erows[5]
    st.caption(f"Mittel über 5 Netze, Kanten ohne Richtung (logarithmische Achse). Die CCH braucht mehr Kanten als die CH: im 20 × 20-Stadtnetz {_num(big['nd'])} mit der Verschachtelten Zerlegung ({big['nd'] / big['ch_hier']:.1f}-mal so viele) und {_num(big['mindeg'])} nach kleinstem Grad, die CH nur {_num(big['ch_hier'])} - "
               f"das ist der Preis dafür, dass sie ohne Kosten gebaut wird. Die Ordnung der Contraction Hierarchy taugt hier nicht ({_num(big['ch'])} Kanten): sie ist auf Zeugensuchen zugeschnitten. Im Zufallsnetz mit 200 Knoten ist die Zerlegung nach Koordinaten sogar schlecht ({_num(rnd['nd'])} Kanten gegen {_num(rnd['mindeg'])} nach kleinstem Grad): "
               "die Lage der Knoten sagt dort nichts über das Netz.")

st.markdown("---")

st.subheader("🔬 Anpassen gegen Neuaufbau")
if st.button("Nach einer Kostenänderung messen (dauert einen Moment)", key="change_start"):
    st.session_state["change_on"] = True
if st.session_state.get("change_on"):
    with st.spinner("Rechne 6 Anteile × 5 Netze ..."):
        crows = _changes()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_change(crows), width="stretch", key="change_chart")
    c2.plotly_chart(build_wrong(crows), width="stretch", key="wrong_chart")
    lo, hi = crows[0], crows[-1]
    st.caption(f"20 × 20-Stadtnetz, Verschachtelte Zerlegung, ein Anteil der Straßen wird dreimal so teuer, Mittel über 5 Netze. Das vollständige Anpassen bearbeitet immer {_num(lo['tri_full'])} Dreiecke; das teilweise Anpassen bearbeitet bei {lo['fraction']} % veränderten Straßen nur {_num(lo['tri_part'])} Dreiecke ({1 - lo['tri_part'] / lo['tri_full']:.0%} gespart), "
               f"bei {hi['fraction']} % {_num(hi['tri_part'])} ({1 - hi['tri_part'] / hi['tri_full']:.1%} gespart): die Änderungen laufen durch die Hierarchie nach oben. Die CH neu zu bauen kostet etwa {_num(lo['witness'])} festgelegte Knoten in Zeugensuchen - unabhängig vom Anteil (eine andere Einheit als Dreiecke: jeder Knoten braucht eine Warteschlange und Kantenprüfungen). "
               f"Die alte CH liegt bei {lo['fraction']} % veränderten Straßen für {lo['wrong']:.0%} der Paare falsch, bei {hi['fraction']} % für {hi['wrong']:.0%}; die angepasste CCH und die neu gebaute CH stimmen in jedem Fall.")

st.markdown("---")

st.subheader("🔬 Wie schnell ist die Abfrage?")
if st.button("Abfrage gegen Netzgröße messen (dauert einen Moment)", key="query_start"):
    st.session_state["query_on"] = True
if st.session_state.get("query_on"):
    with st.spinner("Rechne 6 Netze × 3 Seeds × 100 Paare ..."):
        qrows = _queries()
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(build_query(qrows), width="stretch", key="query_chart")
    cq = [r for r in qrows if r["key"] == "city"]
    c2.table({"Knoten": [f"{r['n']:.0f}" for r in cq], "Dijkstra": [f"{r['dijkstra']:.0f}" for r in cq], "CH / CCH": [f"{r['ch']:.0f} / {r['cch_dijkstra']:.0f}" for r in cq]})
    st.caption(f"Nach dem Anpassen (10 % der Straßen im Stau), Mittel über 100 Zufallspaare in 3 Netzen (alle Antworten mit Dijkstra verglichen). Im 20 × 20-Stadtnetz legt Dijkstra {cq[-1]['dijkstra']:.0f} Knoten fest, die neu gebaute CH {cq[-1]['ch']:.0f}, die Aufwärtssuche auf der CCH {cq[-1]['cch_dijkstra']:.0f}; "
               f"die Kette im Eliminationsbaum braucht {cq[-1]['tree']:.0f} Knoten, keine Warteschlange, aber auch kein Abbruch. Die CCH ist bei der Abfrage etwas langsamer als die CH - dafür reicht nach einer Kostenänderung das Anpassen.")

st.markdown("---")

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Ordnung passt zur Form des Netzes** | Die Zerlegung nach Koordinaten braucht ein Netz mit Lage: im Zufallsnetz (200 Knoten) entstehen 3 390 Kanten statt 904 nach kleinstem Grad. Eine zufällige Ordnung macht die Hierarchie im 14 × 14-Stadtnetz 2.2-mal so groß (3 712 Kanten statt 1 670) und die Abfrage im Eliminationsbaum langsamer als Dijkstra. | Zerlegung nach dem Netz (Literatur: Inertial Flow), kleinster Grad |
| **Mehr Kanten sind egal** | Ohne Zeugensuche entstehen mehr Kanten als bei der CH: im 20 × 20-Stadtnetz 4 358 gegen 1 766 (2.5-mal so viele), nach kleinstem Grad 3 530 (2.0-mal). Speicher und Anpassen wachsen mit. | Zeugensuchen in der Anpassung (Literatur) |
| **Änderungen sind klein** | Das teilweise Anpassen spart bei 1 % veränderten Straßen 65 % der Dreiecke, bei 5 % noch 16 %, bei 20 % nur 3 %. Wer viele Kosten ändert, passt alles an. | Anpassen mit Parallelisierung (Literatur) |
| **Nur die Kosten ändern sich** | Neue oder gesperrte Straßen ändern die Form: eine gesperrte Straße lässt sich als unendliche Kosten anpassen, eine neue Straße nicht - dann muss das Zusammenziehen neu laufen. | (nicht gebaut) |
| **Die Abfrage muss so schnell sein wie bei der CH** | Die Aufwärtssuche auf der CCH legt im 20 × 20-Stadtnetz 67 Knoten fest gegen 53 bei der neu gebauten CH, der Eliminationsbaum braucht 88 Knoten. | Stalling und Pruning der Aufwärtssuche (Literatur) |
"""
)
st.caption("Der Contraction-Hierarchies-Ast der Kürzeste-Wege-Linie: Contraction Hierarchies (Stück 4), Hub Labeling (Stück 10, feste Kosten), Customizable Contraction Hierarchies (Stück 11, wechselnde Kosten).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Modell.** Graph $G=(V,E)$ mit nichtnegativen Kosten $c$. Eine Ordnung $\pi$ (Rang je Knoten) legt fest, welcher Knoten "höher" ist. Für jede Kante $\{u,v\}$ mit $\pi(u)<\pi(v)$ heißt $u$ der untere, $v$ der obere Endknoten.

**Zusammenziehen (Eliminationsspiel).** In der Reihenfolge von $\pi$ wird jeder Knoten $x$ entfernt und seine noch vorhandenen Nachbarn $N^+(x)$ (alle mit höherem Rang) werden paarweise verbunden. Die entstehende Kantenmenge $E^+$ ist die des **chordalen Abschlusses**; sie hängt nur von $G$ und $\pi$ ab, nicht von den Kosten.
Für jede Kante $\{u,v\}\in E^+$ mit $\pi(u)<\pi(v)$ sind die **unteren Dreiecke** die Knoten $x$ mit $\pi(x)<\pi(u)$ und $\{x,u\},\{x,v\}\in E^+$.

**Anpassen.** Setze $w(u,v)=c(u,v)$ (oder $\infty$) und bearbeite die Kanten nach dem Rang ihres unteren Endknotens aufsteigend: $w(u,v)\leftarrow\min\bigl(w(u,v),\,w(u,x)+w(x,v)\bigr)$ über alle unteren Dreiecke $x$. Die Kanten $\{x,u\},\{x,v\}$ haben einen tieferen unteren Endknoten $x$ und sind schon endgültig. Bei gerichteten Kosten gilt dasselbe getrennt für "aufwärts" und "abwärts".

**Korrektheit.** Behauptung: nach dem Anpassen gibt es zu jedem Knotenpaar einen kürzesten Weg, der erst nur aufwärts und dann nur abwärts in $E^+$ verläuft, mit dem Gewicht $d(s,t)$. Beweis durch Induktion über die Zahl der Knoten: der Knoten $x$ mit dem niedrigsten Rang auf einem kürzesten Weg hat zwei Nachbarn $u,v$ auf dem Weg, beide mit höherem Rang - also sind sie in $E^+$ verbunden, und $w(u,v)\le w(u,x)+w(x,v)$. Ersetzt man $u\,x\,v$ durch die Kante $\{u,v\}$, wird der Weg kürzer oder gleich und hat einen Knoten weniger.
Die Abfrage sucht in beiden Richtungen nur aufwärts; für jeden Knoten sind alle höheren Nachbarn Vorfahren im Eliminationsbaum (Elternknoten = niedrigster höherer Nachbar), weil $N^+(x)$ eine Clique bildet.

**Teilweises Anpassen.** Eine Kante muss neu berechnet werden, wenn sich ihr Originalgewicht geändert hat oder eine Kante eines ihrer unteren Dreiecke; ändert sich das neu berechnete Gewicht nicht, hören die Änderungen dort auf. Das Ergebnis ist das des vollständigen Anpassens.

**Aufwand.** Zusammenziehen und Dreiecke: $\sum_x \binom{|N^+(x)|}{2}$; bei einer Ordnung mit kleinen Separatoren (verschachtelte Zerlegung) in ebenen Netzen $O(n\log n)$ Kanten und $O(n^{1.5})$ Dreiecke (Lehrbuchwert, George 1973; gemessen: siehe Experimente). Anpassen: eine Addition und ein Vergleich je Dreieck. Abfrage im Eliminationsbaum: höchstens so viele Knoten wie die Höhe des Baums (in ebenen Netzen mit guter Zerlegung $O(\sqrt n)$).

Implementiert in `cc_graph.py` (CSR-Graph), `cc_ch.py` (Contraction Hierarchy als Vergleich), `cc_cch.py` (Ordnungen, Zusammenziehen, Anpassen, teilweises Anpassen, Abfragen), `cc_scenario.py` (Netze und Verkehr), `cc_evaluation.py` (Kennzahlen, Bildfolge, Experimente).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
