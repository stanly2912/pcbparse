from __future__ import annotations

from copy import deepcopy

import numpy as np

from pcbparse.association.pin_net_ranker import (
    _select_target,
    apply_pin_net_ranker,
)
from pcbparse.ir import (
    BBox,
    CircuitIR,
    Component,
    ImageInfo,
    PhysicalNet,
    Pin,
    Point,
    WireSegment,
)


POLICY = {
    "min_confidence": 0.45,
    "selection_margin": 0.05,
    "accepted_margin": 0.15,
}


def _fixture() -> CircuitIR:
    ir = CircuitIR(
        image=ImageInfo(
            image_id="pin-net-ranker-fixture",
            path="fixture.png",
            width=160,
            height=100,
        )
    )
    ir.components = [
        Component(
            internal_id="c1",
            key="R1",
            category="R",
            bbox=BBox(x1=10, y1=35, x2=30, y2=65),
        ),
        Component(
            internal_id="c2",
            key="R2",
            category="R",
            bbox=BBox(x1=130, y1=35, x2=150, y2=65),
        ),
    ]
    ir.pins = [
        Pin(
            internal_id="p1",
            component_id="c1",
            number="1",
            point=Point(x=30, y=50),
            orientation="R",
            score=0.9,
        ),
        Pin(
            internal_id="p2",
            component_id="c2",
            number="1",
            point=Point(x=130, y=50),
            orientation="L",
            score=0.9,
        ),
    ]
    ir.segments = [
        WireSegment(
            internal_id="s1", start=Point(x=32, y=50), end=Point(x=75, y=50)
        ),
        WireSegment(
            internal_id="s2", start=Point(x=85, y=50), end=Point(x=128, y=50)
        ),
    ]
    ir.physical_nets = [
        PhysicalNet(internal_id="n1", pins=["p1"], segments=["s1"]),
        PhysicalNet(internal_id="n2", pins=["p2"], segments=["s2"]),
    ]
    ir.metadata["snap_decisions"] = [
        {"pin_id": "p1", "status": "accepted", "segment_id": "s1", "cost": 2.0, "margin": 10.0},
        {"pin_id": "p2", "status": "accepted", "segment_id": "s2", "cost": 2.0, "margin": 10.0},
    ]
    return ir


def _geometry(ir: CircuitIR):
    return {
        "pins": [item.model_dump(mode="json") for item in ir.pins],
        "segments": [item.model_dump(mode="json") for item in ir.segments],
        "physical": [
            {
                key: value
                for key, value in item.model_dump(mode="json").items()
                if key != "pins"
            }
            for item in ir.physical_nets
        ],
    }


def test_selection_policy_is_fail_closed_for_accepted_pins() -> None:
    target, reason = _select_target(
        ["current", "other"],
        np.asarray([0.45, 0.55]),
        status="accepted",
        current_id="current",
        policy=POLICY,
    )
    assert target == "current"
    assert reason == "stability_kept"

    target, reason = _select_target(
        ["current", "other"],
        np.asarray([0.35, 0.65]),
        status="accepted",
        current_id="current",
        policy=POLICY,
    )
    assert target == "other"
    assert reason == "reassigned"


def test_ranker_is_deterministic_and_preserves_pin_wire_entities() -> None:
    left = _fixture()
    right = left.model_copy(deep=True)
    before = deepcopy(_geometry(left))

    left_report = apply_pin_net_ranker(left)
    right_report = apply_pin_net_ranker(right)

    assert left_report == right_report
    assert _geometry(left) == before
    assert _geometry(right) == before
    assert [net.model_dump(mode="json") for net in left.physical_nets] == [
        net.model_dump(mode="json") for net in right.physical_nets
    ]
    memberships = [pin for net in left.physical_nets for pin in net.pins]
    assert sorted(memberships) == ["p1", "p2"]
    assert left.metadata["pin_net_ranker"]["resource_sha256"]
