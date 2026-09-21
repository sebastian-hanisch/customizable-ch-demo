"""Plotly-Abbildungen: Karte mit dem Anpassen der Kanten (welche Straßen der Verkehr verändert hat) und den Routen vor und nach dem Verkehr, Tabellen für kleine Netze, Experimente. Achsen sind gesperrt (fixedrange),
damit Touch-Geräte beim Scrollen nicht zoomen."""

import numpy as np
import plotly.graph_objects as go

import cc_constants as C

INF = float("inf")


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.16), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def name_of(g, v):
    return g.names[v] if g.names else f"{v}"


def _lines(g, pairs):
    """Linienspur für eine Liste von Knotenpaaren (None trennt die Segmente)."""
    x = np.full(3 * len(pairs), None, dtype=object)
    y = np.full(3 * len(pairs), None, dtype=object)
    for i, (u, v) in enumerate(pairs):
        x[3 * i], x[3 * i + 1] = g.xy[u, 0], g.xy[v, 0]
        y[3 * i], y[3 * i + 1] = g.xy[u, 1], g.xy[v, 1]
    return x, y


def _route_xy(g, route):
    return g.xy[route]


def build_network(a, k, height=480):
    """Die Karte, nachdem das Anpassen `k` Kanten bearbeitet hat (in der Reihenfolge von unten nach oben): Straßen grau (noch nicht), blau (bearbeitet, Gewicht unverändert) oder rot (bearbeitet, vom Verkehr verändert); bei kleinen Netzen
    auch die Abkürzungen (orange, gestrichelt). Am Ende die Routen: alte Route (grau gestrichelt, auch die der veralteten CH) und neue Route (blau)."""
    net, g, cch = a.net, a.net.graph, a.cch
    small = bool(g.names)
    final = k >= len(a.m1.trace)
    fig = go.Figure()
    todo, same, changed, fill_todo, fill_done = [], [], [], [], []
    for e in range(cch.m):
        pair = (cch.lo[e], cch.hi[e])
        is_original = bool(cch.edge_arcs[e])
        done = e < k
        if is_original:
            (changed if done and a.changed[e] else same if done else todo).append(pair)
        elif small:
            (fill_done if done else fill_todo).append(pair)
    if net.geometric or small:
        for pairs, color, width, name, dash in ((todo, C.COLORS["todo"], 1.5, "noch nicht angepasst", "solid"), (same, C.COLORS["same"], 2.5, "angepasst, unverändert", "solid"), (changed, C.COLORS["changed"], 3.5, "angepasst, vom Verkehr verändert", "solid")):
            if pairs:
                x, y = _lines(g, pairs)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, line=dict(color=color, width=width, dash=dash), hoverinfo="skip"))
        for pairs, alpha, name in ((fill_todo, 0.25, None), (fill_done, 0.9, "Abkürzung (Kante ohne Straße)")):
            if pairs:
                x, y = _lines(g, pairs)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name or "Abkürzung (noch nicht angepasst)", line=dict(color=f"rgba(255,127,14,{alpha})", width=1.5, dash="dash"), hoverinfo="skip"))
    size = 15 if small else (5 if g.n > 300 else 8)
    rank = np.asarray(cch.rank, dtype=float)
    hover = [f"{name_of(g, v)}<br>Rang {int(rank[v])}" for v in range(g.n)]
    if small:
        fig.add_trace(go.Scatter(x=g.xy[:, 0], y=g.xy[:, 1], mode="markers+text", text=[f"{g.names[v]} ({int(rank[v])})" for v in range(g.n)], textposition="top center", showlegend=False, hovertext=hover, hoverinfo="text",
                                 marker=dict(size=size, color=rank, colorscale="Viridis", cmin=0, cmax=g.n - 1, line=dict(color="gray", width=1.5))))
    else:
        fig.add_trace(go.Scatter(x=g.xy[:, 0], y=g.xy[:, 1], mode="markers", showlegend=False, hovertext=hover, hoverinfo="text",
                                 marker=dict(size=size, color=rank, colorscale="Viridis", cmin=0, cmax=g.n - 1, colorbar=dict(title="Rang", thickness=12, len=0.6))))
    if 0 < k <= len(a.m1.trace):                                             # die zuletzt bearbeitete Kante hervorheben
        e = a.m1.trace[k - 1][0]
        pts = g.xy[[cch.lo[e], cch.hi[e]]]
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", name="zuletzt bearbeitet", line=dict(color="black", width=5), hoverinfo="skip"))
    if final and a.metrics["reachable"]:
        if a.q_old.route and a.q_old.route != a.q.route:
            pts = _route_xy(g, a.q_old.route)
            fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", name="alte Route", line=dict(color=C.COLORS["stale"], width=4, dash="dot"), hoverinfo="skip"))
        pts = _route_xy(g, a.q.route)
        fig.add_trace(go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines", name="beste Route jetzt", line=dict(color=C.COLORS["route"], width=6), hoverinfo="skip", opacity=0.55))
    for node, name, color, symbol in ((a.s, "Start", C.COLORS["start"], "square"), (a.t, "Ziel", C.COLORS["goal"], "x")):
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", name=f"{name}: {name_of(g, node)}", hoverinfo="skip", marker=dict(size=size + 4, color=color, symbol=symbol, line=dict(color="white", width=1.5))))
    fig.update_xaxes(visible=False)
    if net.key == "city":
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(visible=False)
    if small:
        lo, hi = g.xy.min(axis=0), g.xy.max(axis=0)
        pad = 0.16 * (hi - lo)
        fig.update_xaxes(range=[lo[0] - pad[0], hi[0] + 1.2 * pad[0]])
        fig.update_yaxes(range=[lo[1] - 1.8 * pad[1], hi[1] + 1.8 * pad[1]])
    return _base(fig, height)


def _edge_name(g, cch, e):
    return f"{name_of(g, cch.lo[e])} – {name_of(g, cch.hi[e])}"


def step_table(a, k, last=6):
    """Die letzten `last` Schritte des Anpassens bis Schritt k: (Schritt, Kante, Originalgewicht, Ergebnis mit Dreiecken)."""
    g, cch, m0, m1 = a.net.graph, a.cch, a.m0, a.m1
    rows = []
    for idx in range(max(0, k - last), k):
        e, ofw, _, nfw, _, ntri = m1.trace[idx]
        via = m1.mid_fw[e]
        kind = "Straße" if cch.edge_arcs[e] else "Abkürzung"
        rows.append({"Schritt": idx + 1, "Kante": f"{_edge_name(g, cch, e)} ({kind})",
                     "Gewicht": (f"{ofw:g}" if np.isfinite(ofw) else "–") + f" → {nfw:g}" + (f" über {name_of(g, via)}" if via >= 0 else "") + (f" [{ntri} Dreiecke]" if ntri else "")})
    return rows


def edge_table(a):
    """Alle CCH-Kanten des kleinen Netzes: Kante, Art, altes und neues Gewicht, Dreieck."""
    g, cch, m0, m1 = a.net.graph, a.cch, a.m0, a.m1
    rows = []
    for e in range(cch.m):
        via = m1.mid_fw[e]
        rows.append({"Kante": _edge_name(g, cch, e), "Art": "Straße" if cch.edge_arcs[e] else "Abkürzung",
                     "Gewicht alt → neu": f"{m0.fw[e]:g} → {m1.fw[e]:g}" + (f" (über {name_of(g, via)})" if via >= 0 else "") + ("  ← Verkehr" if a.changed[e] else "")})
    return rows


def build_edges(rows):
    """Kanten der CCH je Ordnung gegen Originalkanten plus Abkürzungen der Contraction Hierarchy (logarithmisch)."""
    fig = go.Figure()
    x = [f"{'Stadt' if r['key'] == 'city' else 'Zufall'} {r['n']:.0f}" for r in rows]
    for col, name in (("ch_hier", "Contraction Hierarchy"), ("nd", "CCH: Verschachtelte Zerlegung"), ("mindeg", "CCH: kleinster Grad"), ("ch", "CCH: Ordnung der CH"), ("random", "CCH: Zufall")):
        fig.add_trace(go.Bar(x=x, y=[None if np.isnan(r[col]) else r[col] for r in rows], name=name))
    fig.update_layout(barmode="group", yaxis_title="Kanten (ohne Richtung)")
    fig.update_yaxes(type="log")
    return _base(fig, 340)


def build_change(rows):
    """Aufwand nach einer Kostenänderung gegen den Anteil der veränderten Straßen: Dreiecke (vollständig, teilweise) und festgelegte Knoten der Zeugensuchen beim Neuaufbau der CH."""
    fig = go.Figure()
    x = [r["fraction"] for r in rows]
    fig.add_trace(go.Scatter(x=x, y=[r["tri_full"] for r in rows], mode="lines+markers", name="Anpassen: alle Dreiecke"))
    fig.add_trace(go.Scatter(x=x, y=[r["tri_part"] for r in rows], mode="lines+markers", name="Anpassen: nur betroffene Dreiecke"))
    fig.add_trace(go.Scatter(x=x, y=[r["witness"] for r in rows], mode="lines+markers", name="CH neu bauen: Knoten der Zeugensuchen", line=dict(dash="dot")))
    fig.update_layout(xaxis_title="Straßen im Stau [%]", yaxis_title="Zähler")
    return _base(fig, 320)


def build_wrong(rows):
    fig = go.Figure(go.Bar(x=[f"{r['fraction']}%" for r in rows], y=[100 * r["wrong"] for r in rows], marker_color="#d62728"))
    fig.update_layout(xaxis_title="Straßen im Stau (dreifache Kosten)", yaxis_title="falsche Antworten der alten CH [%]")
    return _base(fig, 320)


def build_query(rows):
    """Schritte einer Abfrage nach dem Anpassen gegen die Knotenzahl (logarithmisch)."""
    fig = go.Figure()
    for col, name in (("dijkstra", "Dijkstra"), ("tree", "CCH: Eliminationsbaum"), ("cch_dijkstra", "CCH: Aufwärtssuche"), ("ch", "CH (neu gebaut)")):
        for key, label, dash in (("city", "Stadtnetz", "solid"), ("random", "Zufallsnetz", "dot")):
            r = [x for x in rows if x["key"] == key]
            fig.add_trace(go.Scatter(x=[x["n"] for x in r], y=[x[col] for x in r], mode="lines+markers", name=f"{name} ({label})", line=dict(dash=dash)))
    fig.update_layout(xaxis_title="Knoten", yaxis_title="Schritte einer Abfrage")
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    return _base(fig, 360)
