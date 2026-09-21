"""Presets, Permalink-Angaben und Regler-Grenzen sind untereinander stimmig."""

import pytest

import cc_constants as C
import cc_presets as P
from cc_scenario import make_network


def test_every_preset_sets_every_control_within_bounds_and_has_help():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 4
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and p["net"] in C.NETS and p["order"] in C.ORDER_LABELS
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            if spec.lo is not None:
                assert spec.lo <= p[key] <= spec.hi, (name, key)
        make_network(p["net"], p["side"], p["nodes"], p["degree"], p["seed"])
        assert C.PRESET_HELP[name]
    assert [p["net"] for p in C.PRESETS.values()][:3] == list(C.NETS)


def test_setting_specs_and_kept_keys_are_consistent():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS) and set(P.KEPT) <= set(P.SETTING_SPECS)
    for spec in P.SETTING_SPECS.values():
        assert spec.lo is None or spec.lo < spec.hi                                 # kein Regler mit gleichen Grenzen (Streamlit bricht ab)
    assert len({s.url_param for s in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)


def test_defaults_lie_inside_the_bounds_and_the_largest_net_has_at_most_400_nodes():
    for lo, hi, d in ((C.SIDE_MIN, C.SIDE_MAX, C.DEFAULT_SIDE), (C.NODES_MIN, C.NODES_MAX, C.DEFAULT_NODES), (C.DEGREE_MIN, C.DEGREE_MAX, C.DEFAULT_DEGREE), (C.TRAFFIC_MIN, C.TRAFFIC_MAX, C.DEFAULT_TRAFFIC),
                      (C.DISTANCE_MIN, C.DISTANCE_MAX, C.DEFAULT_DISTANCE)):
        assert lo < d < hi
    assert C.DEFAULT_NET == "small" and C.DEFAULT_ORDER in C.ORDER_LABELS and C.SIDE_MAX ** 2 <= 400 and C.NODES_MAX <= 200


def test_choice_casters_reject_unknown_values():
    for key, good, bad in (("net_select", "random", "ring"), ("order_select", "mindeg", "alphabet")):
        cast = P.SETTING_SPECS[key].caster
        assert cast(good) == good
        with pytest.raises(ValueError):
            cast(bad)
