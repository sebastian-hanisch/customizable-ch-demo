# Customizable Contraction Hierarchies – Ordnung einmal, Kosten je Verkehrslage – Streamlit-Demo

Elftes Stück der Kürzeste-Wege-Linie der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations Research und Machine Learning", Kind von [Contraction Hierarchies](../contraction-hierarchies-demo):
anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im Vergleich) zeigt diese Demo **ein** Verfahren – die **Customizable Contraction Hierarchy (CCH)** – an einem wachsenden Beispiel.
Eine Contraction Hierarchy ist schnell, aber **an ihre Kosten gebunden**: welche Abkürzungen entstehen, entscheidet die Zeugensuche, und die hängt von den Kosten ab. Ändert sich der Verkehr, ist die Hierarchie veraltet. Die CCH trennt beides: **Ordnung** und **Zusammenziehen** hängen nur von der
**Form des Netzes** ab (ohne Zeugensuche, mit mehr Kanten), die Kosten werden in einer schnellen dritten Phase, dem **Anpassen**, von unten nach oben über die **unteren Dreiecke** jeder Kante eingetragen. Nach einer Kostenänderung wird nur diese Phase wiederholt.

**Einordnung in die Reihe (die Kanten des Graphen):**
```
bfs-demo (Wurzel: Kanten zählen, nicht Kosten)                                       [gebaut]
  └─ dijkstra-demo (Kosten korrekt, blind in alle Richtungen)                        [gebaut]
       ├─ bidirectional-demo → contraction-hierarchies-demo                          [gebaut]
       │                         ├─ hub-labeling-demo (die Suche vorwegnehmen)       [gebaut]
       │                         └─ customizable-ch-demo (Kosten wechseln)           [dieses Stück]
       ├─ bellman-ford-demo ─┐                                                       [gebaut]
       │   floyd-warshall-demo ─┴→ johnson-demo (Konvergenz: Umgewichtung)           [gebaut]
       ├─ multicriteria-demo (Zeit gegen CO₂, Pareto)                                [gebaut]
       └─ time-dependent-demo (Kosten hängen von der Uhrzeit ab)                     [gebaut]
```

## Quellen

| Bestandteil | Quelle |
|---|---|
| Verfahren (drei Phasen, untere Dreiecke, Eliminationsbaum-Abfrage) | Dibbelt, Strasser und Wagner (2016); die Bücher (*Grokking Algorithms*, *Optimization Algorithms*) behandeln CCH nicht, es gibt kein Buchbeispiel zu spiegeln |
| Verschachtelte Zerlegung (nested dissection), Eliminationsspiel | George (1973), Rose (1972) – als Verfahren; die Umsetzung nach Koordinaten ist eigen |
| Umsetzung, teilweises Anpassen, Abfragen, Contraction Hierarchy als Vergleich (Kopie aus `contraction-hierarchies-demo`), Messreihen | eigen |
| Alle Netze und Verkehrslagen | **eigene Graphen und Erzeuger**: kleines Netz mit zwei Stadtteilen und zwei Brücken, Stadtnetz auf einem gestörten Raster, Zufallsnetz; der Verkehr macht einen Anteil der Straßen dreimal so teuer |
| Zahlen | **eigene Messungen** an diesen Netzen |

Aus den Büchern stammt keine Zahl, kein Graph und kein Text. **Kein OpenStreetMap-Auszug**, also keine ODbL-Pflichten.

## Ergebnis (Zahlen aus den Tests)

| Frage | Ergebnis |
|---|---|
| Kleines Netz (10 Orte, 14 Straßen) | ✅ die Hierarchie hat **22 Kanten** (14 Straßen und 8 Abkürzungen) und 21 untere Dreiecke; der Stau auf der nördlichen Brücke verändert 7 der 22 Kanten. Altstadt → Fabrik: vorher **14 min** über die nördliche Brücke, jetzt **16 min** über die südliche – die alte Contraction Hierarchy meldet weiter 14 min. Teilweises Anpassen: 17 statt 21 Dreiecke |
| Stadtnetz (10 × 10, Seed 3, 10 % der Straßen dreimal so teuer) | ✅ 657 Kanten (180 Straßen), 2 348 Dreiecke, 168 Kanten ändern ihr Gewicht; gezeigtes Paar 797 m → 831 m, die alte CH meldet weiter 797 und liegt bei **49.5 % aller Paare** falsch; die angepasste CCH stimmt bei jedem |
| Alte CH nach dem Verkehr (20 × 20-Stadtnetz, Mittel über 5 Netze) | ❌ wenn 1 / 2 / 5 / 10 / 20 / 50 % der Straßen dreimal so teuer werden, liefert sie für **7 / 22 / 45 / 69 / 86 / 98 %** der Paare eine falsche Entfernung |
| Anpassen gegen Neuaufbau (20 × 20) | ⚠️ das **vollständige Anpassen** bearbeitet 32 051 Dreiecke; das **teilweise Anpassen** nur 11 243 bei 1 % veränderten Straßen (65 % gespart), 26 831 bei 5 % (16 %), 31 192 bei 20 % (**3 %**) – die Änderungen laufen durch die Hierarchie nach oben. Die CH neu zu bauen kostet etwa 41 500 festgelegte Knoten in Zeugensuchen (eine andere Einheit: jeder mit Warteschlange), unabhängig vom Anteil |
| Preis der CCH: Kanten (20 × 20, Mittel über 5 Netze) | ⚠️ **4 358 Kanten** mit der Verschachtelten Zerlegung (2.5-mal so viele wie die 1 766 der CH), 3 530 nach kleinstem Grad (2.0-mal); bei 6 / 10 / 14 / 20 Kreuzungen je Seite 153 / 663 / 1 674 / 4 358 gegen 95 / 335 / 776 / 1 766 |
| Ordnung | ⚠️ die Ordnung der Contraction Hierarchy taugt als CCH-Ordnung nicht (**11 646** Kanten im 20 × 20-Netz); eine zufällige Ordnung macht das 14 × 14-Netz 2.2-mal so groß (3 712 statt 1 670 Kanten, 62 450 statt 8 588 Dreiecke) und die Abfrage im Eliminationsbaum langsamer als Dijkstra (124.4 gegen 97.3 Schritte). Im Zufallsnetz (200 Knoten) ist die **Zerlegung nach Koordinaten schlecht** (3 390 Kanten gegen 904 nach kleinstem Grad): die Lage der Knoten sagt dort nichts über das Netz |
| Abfrage nach dem Anpassen (20 × 20, 100 Paare) | ⚠️ Dijkstra legt **211** Knoten fest, die neu gebaute CH **53**, die bidirektionale Aufwärtssuche auf der CCH **67**, die Kette im Eliminationsbaum braucht **88** Knoten: die CCH ist bei der Abfrage etwas langsamer als die CH, aber deutlich schneller als Dijkstra; im kleinsten Netz (36 Knoten) lohnt der Baum nicht |
| Korrektheit | ✅ beide Abfragen beantworten **jedes Paar** exakt wie networkx (gerichtete und ungerichtete Graphen, alle vier Ordnungen, Nullkosten, Parallelkanten, unerreichbare Ziele); die **Struktur** (chordaler Abschluss, untere Dreiecke, Eliminationsbaum) wird aus den Definitionen nachgerechnet; **jedes Kantengewicht** ist die kürzeste Entfernung über tiefer geordnete Knoten (unabhängiger Dijkstra); das **teilweise Anpassen ist nach beliebigen Änderungen (auch Senkungen und Nullkosten) identisch mit dem vollständigen** |

Die Zähler (Kanten, Dreiecke, festgelegte Knoten, Schritte) sind Schritte des Verfahrens und plattformfest. Laufzeiten stehen in der App nur als Messwerte (reines Python) und werden nirgends behauptet oder getestet.

## Was die Demo zeigt

1. **Customizable CH in Aktion:** ein Regler über die **bearbeiteten Kanten** (+ Abspielen): das Anpassen von unten nach oben auf der Karte – Straßen grau (noch nicht), blau (bearbeitet, unverändert) oder **rot (der Verkehr hat das Gewicht verändert)**, beim kleinen Netz auch die Abkürzungen; am Ende die alte und die neue beste Route. Dazu die letzten Schritte als Tabelle (Gewicht alt → neu, über welches Dreieck) und beim kleinen Netz **alle Kanten** der Hierarchie.
2. **Nach dem Verkehr:** Kanten der CCH gegen die der CH, Dreiecke beim vollständigen und beim teilweisen Anpassen, Neuaufbau der CH (festgelegte Knoten in Zeugensuchen), Anteil falscher Antworten der alten CH; das Urteil unterscheidet Route ändert sich, Route bleibt für dieses Paar gleich und keine Straße im Stau.
3. **Vergleich** (Expander: Anpassen gegen Neuaufbau, Abfrage bei Dijkstra, CH und beiden CCH-Abfragen); **Experimente auf Knopfdruck**: Kanten gegen Netzgröße und Ordnung; Anpassen gegen Neuaufbau gegen den Anteil veränderter Straßen; Abfrage gegen Netzgröße.
4. **Wo die Annahmen enden** (Tabelle; Ordnung passt zur Form, mehr Kanten sind egal, Änderungen sind klein, nur die Kosten ändern sich, Abfrage so schnell wie CH) und **Mathematische Formulierung** (Eliminationsspiel, Dreiecke, Anpassen, Korrektheit, teilweises Anpassen, Aufwand als Lehrbuchwert gekennzeichnet).

Bedienung: Beispielnetz per Schnellstart-Knopf laden oder in der Seitenleiste Netz, **Ordnung ohne Kosten** und – bei erzeugten Netzen – Größe, Dichte, **Anteil der Straßen im Stau**, Entfernung des Paares und Seed wählen. Die Adresszeile spiegelt die Konfiguration (Permalink). Höchstens 400 Knoten (Zufallsnetz 200).

## Dateien

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `cc_graph.py`, `cc_ch.py` | Graph in CSR-Form; Contraction Hierarchy (Kopie aus `contraction-hierarchies-demo`) als Vergleich |
| `cc_cch.py` | Ordnungen ohne Kosten, Zusammenziehen (Kanten, Dreiecke, Eliminationsbaum), Anpassen, teilweises Anpassen, beide Abfragen mit Auspacken |
| `cc_scenario.py` | Netze und Verkehr: kleines Netz, Stadtnetz, Zufallsnetz |
| `cc_evaluation.py` | Kennzahlen, Bildfolge, Messreihen |
| `cc_visualization.py`, `cc_presets.py`, `cc_constants.py` | Abbildungen, Presets und Permalink, Konstanten |

## Lokal starten

```bash
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/`. Jede Zahl in Hilfetexten, Presets und Tabellen ist in `tests/test_claims.py` belegt; die Kreuzprobe läuft gegen networkx für **alle Paare** und gegen Definitionen (Struktur, Gewichte über tiefere Knoten); ein Regressionstest klickt "▶️ Abspielen" auf Netzen mit mehreren Bildern.
