from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pcbparse.association.geometry import (
    closest_point_on_segment,
    orientation_penalty,
    point_segment_distance,
)
from pcbparse.ir.models import CircuitIR


MAX_RADIUS_PX = 20.0
MAX_CANDIDATES = 8
CELL_SIZE = 32.0
ENDPOINT_TOLERANCE_PX = 3.0
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resources"


class PinNetRankerError(RuntimeError):
    """Raised when the frozen Pin-to-Net ranker cannot run safely."""


class _SegmentIndex:
    def __init__(self, segments: list[Any]) -> None:
        self.cells: dict[tuple[int, int], list[Any]] = defaultdict(list)
        for segment in segments:
            min_x = min(segment.start.x, segment.end.x) - MAX_RADIUS_PX
            max_x = max(segment.start.x, segment.end.x) + MAX_RADIUS_PX
            min_y = min(segment.start.y, segment.end.y) - MAX_RADIUS_PX
            max_y = max(segment.start.y, segment.end.y) + MAX_RADIUS_PX
            for cell_x in range(
                math.floor(min_x / CELL_SIZE), math.floor(max_x / CELL_SIZE) + 1
            ):
                for cell_y in range(
                    math.floor(min_y / CELL_SIZE), math.floor(max_y / CELL_SIZE) + 1
                ):
                    self.cells[(cell_x, cell_y)].append(segment)

    def query(self, point: Any) -> list[Any]:
        key = (math.floor(point.x / CELL_SIZE), math.floor(point.y / CELL_SIZE))
        unique = {
            segment.internal_id: segment for segment in self.cells.get(key, [])
        }
        return [unique[item] for item in sorted(unique)]


def _orientation_vector(value: str) -> tuple[float, float] | None:
    return {
        "L": (-1.0, 0.0),
        "R": (1.0, 0.0),
        "T": (0.0, -1.0),
        "B": (0.0, 1.0),
        "U": (0.0, -1.0),
        "D": (0.0, 1.0),
    }.get(str(value))


def _nearest_endpoint(segment: Any, projection: Any) -> tuple[float, Any]:
    start_gap = math.hypot(
        projection.x - segment.start.x, projection.y - segment.start.y
    )
    end_gap = math.hypot(
        projection.x - segment.end.x, projection.y - segment.end.y
    )
    return (start_gap, segment.start) if start_gap <= end_gap else (end_gap, segment.end)


def _endpoint_degree(
    endpoint: Any, net_segment_ids: set[str], segments: Mapping[str, Any]
) -> int:
    return sum(
        min(
            math.hypot(
                endpoint.x - segments[item].start.x,
                endpoint.y - segments[item].start.y,
            ),
            math.hypot(
                endpoint.x - segments[item].end.x,
                endpoint.y - segments[item].end.y,
            ),
        )
        <= ENDPOINT_TOLERANCE_PX
        for item in net_segment_ids
        if item in segments
    )


def _candidate_record(
    pin: Any,
    segment: Any,
    net_id: str,
    net_segment_ids: set[str],
    segments: Mapping[str, Any],
) -> dict[str, Any]:
    projection = closest_point_on_segment(pin.point, segment)
    distance = point_segment_distance(pin.point, segment)
    orient = orientation_penalty(pin.orientation, segment)
    endpoint_gap, endpoint = _nearest_endpoint(segment, projection)
    delta_x = projection.x - pin.point.x
    delta_y = projection.y - pin.point.y
    norm = math.hypot(delta_x, delta_y)
    direction = _orientation_vector(pin.orientation)
    alignment = (
        None
        if direction is None or norm <= 1e-6
        else (delta_x * direction[0] + delta_y * direction[1]) / norm
    )
    return {
        "physical_net_id": net_id,
        "segment_id": segment.internal_id,
        "distance_px": float(distance),
        "orientation_penalty": float(orient),
        "snap_cost": float(distance + 2.0 * orient),
        "projection": {"x": float(projection.x), "y": float(projection.y)},
        "projection_kind": (
            "endpoint" if endpoint_gap <= ENDPOINT_TOLERANCE_PX else "interior"
        ),
        "endpoint_distance_px": float(endpoint_gap),
        "endpoint_degree": (
            _endpoint_degree(endpoint, net_segment_ids, segments)
            if endpoint_gap <= ENDPOINT_TOLERANCE_PX
            else 0
        ),
        "outward_alignment": alignment,
    }


def _feature_vector(option: Mapping[str, Any], categories: list[str]) -> np.ndarray:
    distance = min(20.0, float(option.get("distance_px", 20.0))) / 20.0
    alignment = option.get("outward_alignment")
    status = str(option.get("baseline_status", "unknown"))
    numeric = [
        distance,
        distance * distance,
        float(option.get("orientation_penalty", 1.0)),
        0.0 if alignment is None else float(alignment),
        float(alignment is None),
        float(option.get("projection_kind") == "endpoint"),
        min(20.0, float(option.get("endpoint_distance_px", 20.0))) / 20.0,
        min(8.0, float(option.get("endpoint_degree", 0))) / 8.0,
        min(8.0, float(option.get("rank", 8))) / 8.0,
        min(24.0, float(option.get("snap_cost", 24.0))) / 24.0,
        min(20.0, max(0.0, float(option.get("cost_gap", 20.0)))) / 20.0,
        min(8.0, float(option.get("candidate_count", 0))) / 8.0,
        min(32.0, float(option.get("component_pin_count", 1))) / 32.0,
        min(64.0, float(option.get("net_segment_count", 0))) / 64.0,
        min(32.0, float(option.get("net_pin_count", 0))) / 32.0,
        min(16.0, float(option.get("peer_support", 0))) / 16.0,
        min(8.0, float(option.get("same_component_peer_support", 0))) / 8.0,
        float(option.get("is_current_net", False)),
        0.0,
        float(status == "accepted"),
        float(status == "ambiguous"),
        float(status == "unmatched"),
        float(option.get("pin_score", 0.0)),
        float(option.get("pin_inside_component", 0.0)),
        min(2.0, float(option.get("pin_boundary_distance", 2.0))) / 2.0,
        max(-2.0, min(2.0, float(option.get("pin_relative_x", 0.0)))) / 2.0,
        max(-2.0, min(2.0, float(option.get("pin_relative_y", 0.0)))) / 2.0,
        min(512.0, float(option.get("component_width", 1.0))) / 512.0,
        min(512.0, float(option.get("component_height", 1.0))) / 512.0,
    ]
    category = str(option.get("category", "unknown")).casefold()
    numeric.extend(float(category == item) for item in categories)
    return np.asarray(numeric, dtype=np.float32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=4)
def _load_artifact(resource_name: str) -> dict[str, Any]:
    if Path(resource_name).name != resource_name:
        raise PinNetRankerError("pin-net ranker resource must be a file name")
    path = (RESOURCE_ROOT / resource_name).resolve()
    if path.parent != RESOURCE_ROOT.resolve() or not path.is_file():
        raise PinNetRankerError(f"pin-net ranker resource not found: {resource_name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise PinNetRankerError("unsupported pin-net ranker schema")
    categories = payload.get("categories")
    layers = payload.get("layers")
    policy = payload.get("policy")
    if not isinstance(categories, list) or not isinstance(layers, list) or len(layers) != 3:
        raise PinNetRankerError("invalid pin-net ranker dimensions")
    if not isinstance(policy, dict):
        raise PinNetRankerError("missing pin-net ranker policy")
    payload["resource_sha256"] = _sha256(path)
    return payload


def _forward(features: np.ndarray, artifact: Mapping[str, Any]) -> np.ndarray:
    value = features.astype(np.float32, copy=False)
    mean = np.asarray(artifact["mean"], dtype=np.float32)
    std = np.asarray(artifact["std"], dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != len(mean) or len(std) != len(mean):
        raise PinNetRankerError("pin-net feature dimension mismatch")
    value = (value - mean) / std
    for index, layer in enumerate(artifact["layers"]):
        weight = np.asarray(layer["weight"], dtype=np.float32)
        bias = np.asarray(layer["bias"], dtype=np.float32)
        value = value @ weight.T + bias
        if index < len(artifact["layers"]) - 1:
            value = np.maximum(value, 0.0)
    return value.reshape(-1)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / np.sum(values)


def _select_target(
    option_ids: list[str],
    probabilities: np.ndarray,
    *,
    status: str,
    current_id: str | None,
    policy: Mapping[str, Any],
) -> tuple[str | None, str]:
    if not option_ids:
        return current_id, "no_candidate"
    order = np.argsort(-probabilities)
    top_index = int(order[0])
    top_score = float(probabilities[top_index])
    second_score = float(probabilities[order[1]]) if len(order) > 1 else 0.0
    target = option_ids[top_index]
    if target == current_id:
        return current_id, "kept"
    if top_score < float(policy["min_confidence"]):
        return current_id, "low_confidence"
    if top_score - second_score < float(policy["selection_margin"]):
        return current_id, "ambiguous_margin"
    if status == "accepted":
        if current_id not in option_ids:
            return current_id, "current_not_in_candidates"
        current_score = float(probabilities[option_ids.index(current_id)])
        if top_score - current_score < float(policy["accepted_margin"]):
            return current_id, "stability_kept"
    return target, "reassigned"


def _pin_candidates(ir: CircuitIR, artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    categories = list(map(str, artifact["categories"]))
    components = ir.component_index()
    pins = ir.pin_index()
    segments = ir.segment_index()
    nets = {net.internal_id: net for net in ir.physical_nets}
    segment_to_net = {
        segment_id: net.internal_id
        for net in ir.physical_nets
        for segment_id in net.segments
    }
    membership = {
        pin_id: net.internal_id for net in ir.physical_nets for pin_id in net.pins
    }
    component_pin_counts = Counter(str(pin.component_id) for pin in ir.pins)
    decisions = {
        str(item["pin_id"]): item for item in ir.metadata.get("snap_decisions", [])
    }
    spatial = _SegmentIndex(ir.segments)
    rows = []
    for pin in sorted(ir.pins, key=lambda item: item.internal_id):
        best_by_net: dict[str, tuple[tuple[Any, ...], Any]] = {}
        for segment in spatial.query(pin.point):
            net_id = segment_to_net.get(segment.internal_id)
            if net_id is None:
                continue
            distance = point_segment_distance(pin.point, segment)
            if distance > MAX_RADIUS_PX:
                continue
            cost = distance + 2.0 * orientation_penalty(pin.orientation, segment)
            key = (cost, distance, segment.internal_id)
            if net_id not in best_by_net or key < best_by_net[net_id][0]:
                best_by_net[net_id] = (key, segment)
        ordered = sorted(
            best_by_net.items(), key=lambda item: (item[1][0], item[0])
        )[:MAX_CANDIDATES]
        candidates = [
            _candidate_record(
                pin, segment, net_id, set(nets[net_id].segments), segments
            )
            for net_id, (_key, segment) in ordered
        ]
        component = components.get(str(pin.component_id))
        width = max(1.0, float(component.bbox.width)) if component else 1.0
        height = max(1.0, float(component.bbox.height)) if component else 1.0
        scale = max(width, height)
        center_x = (
            (component.bbox.x1 + component.bbox.x2) / 2.0
            if component
            else pin.point.x
        )
        center_y = (
            (component.bbox.y1 + component.bbox.y2) / 2.0
            if component
            else pin.point.y
        )
        inside = float(
            component is not None
            and component.bbox.x1 <= pin.point.x <= component.bbox.x2
            and component.bbox.y1 <= pin.point.y <= component.bbox.y2
        )
        boundary = (
            min(
                abs(pin.point.x - component.bbox.x1),
                abs(pin.point.x - component.bbox.x2),
                abs(pin.point.y - component.bbox.y1),
                abs(pin.point.y - component.bbox.y2),
            )
            / scale
            if component
            else 2.0
        )
        component_id = str(pin.component_id)
        current_id = membership.get(pin.internal_id)
        status = str(decisions.get(pin.internal_id, {}).get("status", "unknown"))
        best_cost = min(
            (float(item["snap_cost"]) for item in candidates), default=0.0
        )
        vectors = []
        option_ids = []
        for rank, candidate in enumerate(candidates, start=1):
            net_id = str(candidate["physical_net_id"])
            net = nets[net_id]
            peers = [peer for peer in net.pins if peer != pin.internal_id]
            record = {
                **candidate,
                "rank": rank,
                "candidate_count": len(candidates),
                "cost_gap": float(candidate["snap_cost"]) - best_cost,
                "component_pin_count": component_pin_counts.get(component_id, 1),
                "net_segment_count": len(net.segments),
                "net_pin_count": len(net.pins),
                "peer_support": len(peers),
                "same_component_peer_support": sum(
                    str(pins[peer].component_id) == component_id
                    for peer in peers
                    if peer in pins
                ),
                "is_current_net": current_id == net_id,
                "baseline_status": status,
                "category": (
                    str(component.category).casefold() if component else "unknown"
                ),
                "pin_score": float(pin.score),
                "pin_inside_component": inside,
                "pin_boundary_distance": boundary,
                "pin_relative_x": (pin.point.x - center_x) / scale,
                "pin_relative_y": (pin.point.y - center_y) / scale,
                "component_width": width,
                "component_height": height,
            }
            vectors.append(_feature_vector(record, categories))
            option_ids.append(net_id)
        rows.append(
            {
                "pin_id": pin.internal_id,
                "status": status,
                "current_id": current_id,
                "option_ids": option_ids,
                "features": vectors,
            }
        )
    return rows


def _apply_assignments(ir: CircuitIR, assignments: Mapping[str, str]) -> None:
    nets = {net.internal_id: net for net in ir.physical_nets}
    for target in assignments.values():
        if target not in nets:
            raise PinNetRankerError(f"unknown physical net target: {target}")
    for pin_id, target in sorted(assignments.items()):
        for net in ir.physical_nets:
            net.pins = [value for value in net.pins if value != pin_id]
        nets[target].pins.append(pin_id)
    for net in ir.physical_nets:
        net.pins = sorted(set(net.pins))


def apply_pin_net_ranker(
    ir: CircuitIR, *, resource_name: str = "pin_net_ranker_v1.json"
) -> dict[str, Any]:
    """Re-rank existing Pin-to-Physical-Net candidates without changing geometry."""
    artifact = _load_artifact(resource_name)
    policy = artifact["policy"]
    rows = _pin_candidates(ir, artifact)
    assignments: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for row in rows:
        if not row["features"]:
            counts[f"{row['status']}_no_candidate"] += 1
            continue
        features = np.stack(row["features"])
        probabilities = _softmax(_forward(features, artifact))
        target, reason = _select_target(
            row["option_ids"],
            probabilities,
            status=row["status"],
            current_id=row["current_id"],
            policy=policy,
        )
        counts[f"{row['status']}_{reason}"] += 1
        if target is not None and target != row["current_id"]:
            assignments[str(row["pin_id"])] = str(target)
    _apply_assignments(ir, assignments)
    report = {
        "enabled": True,
        "version": str(artifact.get("version", "unknown")),
        "resource": resource_name,
        "resource_sha256": artifact["resource_sha256"],
        "policy": dict(policy),
        "pins_considered": len(rows),
        "changed_pins": len(assignments),
        "counts": dict(sorted(counts.items())),
    }
    ir.metadata["pin_net_ranker"] = report
    return report


__all__ = [
    "PinNetRankerError",
    "apply_pin_net_ranker",
]
