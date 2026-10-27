from __future__ import annotations

import hashlib
import json
import os
import tempfile
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pcbparse.association import associate_ocr
from pcbparse.association.pin_net_ranker import apply_pin_net_ranker
from pcbparse.graph.palette_wire import apply_palette_wire_filter, prepare_palette_wire
from pcbparse.graph.internal_wire_filter import filter_internal_wire_segments
from pcbparse.graph.physical import (
    build_physical_graph,
    filter_body_outline_wire_segments,
    filter_decorative_wire_segments,
    filter_glyph_wire_segments,
)
from pcbparse.ir.models import CircuitIR, Diagnostic, IRState
from pcbparse.nets.hypergraph import build_electrical_nets
from pcbparse.output.official_schema import split_official_pin_numbers
from pcbparse.output.official_serializer import (
    atomic_write_official_json,
    official_json_hash,
    serialize_official_with_quality,
)
from pcbparse.pipeline.context import RunContext, stable_file_hash
from pcbparse.reasoning.orchestrator import run_selective_reasoning
from pcbparse.reasoning.qwen_client import (
    CandidateOnlyReasoner,
    DisabledReasoner,
    HTTPQwenReasoner,
)
from pcbparse.validation import (
    GeometryValidator,
    GraphValidator,
    ReferenceValidator,
    SchemaValidator,
)
from pcbparse.validation.common import merge_diagnostics, stable_diagnostics
from pcbparse.validation.repair import repair_ir_once


class PipelineError(RuntimeError):
    def __init__(self, message: str, diagnostics: list[Diagnostic] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or []


class IRProvider(Protocol):
    def build_ir(self, input_path: str | Path, ctx: RunContext | None = None) -> CircuitIR: ...


@dataclass(slots=True)
class InferenceResult:
    input_path: Path
    output_path: Path
    output: dict[str, Any]
    output_hash: str
    ir: CircuitIR
    diagnostics: list[Diagnostic]
    context: RunContext
    quality: dict[str, Any] = field(default_factory=dict)


def _submission_mode(config: dict[str, Any]) -> str:
    run = config.get("run", {})
    mode = str(run.get("submission_mode", "strict")).strip().lower()
    return mode if mode in {"strict", "degraded"} else "strict"


def _is_debug_execution(config: dict[str, Any]) -> bool:
    """True when the pipeline runs in debug/non-production mode."""
    run = config.get("run", {})
    profile = str(run.get("profile", "")).strip().lower()
    return profile == "debug"


def _prune_unrepresentable(ir: Any) -> None:
    """Drop objects the official schema cannot represent (never guess).

    Records deleted object IDs and affected nets so degraded output stays
    auditable; missing component keys / pin numbers are never fabricated.
    """
    valid_components = {
        item.internal_id
        for item in ir.components
        if isinstance(item.key, str) and item.key.strip()
    }
    total_components = len(ir.components)
    missing_number_pin_ids = sorted(
        item.internal_id
        for item in ir.pins
        if item.component_id in valid_components
        and not split_official_pin_numbers(item.number)
    )
    unowned_pin_ids = sorted(
        item.internal_id
        for item in ir.pins
        if item.component_id not in valid_components
    )
    valid_pins = {
        item.internal_id
        for item in ir.pins
        if item.internal_id not in set(missing_number_pin_ids)
        and item.internal_id not in set(unowned_pin_ids)
    }
    total_pins = len(ir.pins)
    removed_component_ids = sorted(
        item.internal_id for item in ir.components if item.internal_id not in valid_components
    )
    removed_pin_ids = sorted(
        item.internal_id for item in ir.pins if item.internal_id not in valid_pins
    )
    ir.components = [item for item in ir.components if item.internal_id in valid_components]
    ir.pins = [item for item in ir.pins if item.internal_id in valid_pins]
    nets_affected = 0
    affected_net_ids: list[str] = []
    for net in ir.physical_nets:
        kept = [pid for pid in net.pins if pid in valid_pins]
        if len(kept) != len(net.pins):
            nets_affected += 1
            affected_net_ids.append(net.internal_id)
        net.pins = kept
    for net in ir.electrical_nets:
        kept = [pid for pid in net.pins if pid in valid_pins]
        if len(kept) != len(net.pins):
            nets_affected += 1
            affected_net_ids.append(net.internal_id)
        net.pins = kept
    kept_electrical = [net for net in ir.electrical_nets if net.pins]
    dropped_net_ids = sorted(
        net.internal_id for net in ir.electrical_nets if not net.pins
    )
    nets_dropped = len(ir.electrical_nets) - len(kept_electrical)
    ir.electrical_nets = kept_electrical
    ir.metadata["degraded_submission_prune"] = {
        "components_dropped": total_components - len(ir.components),
        "component_ids_dropped": removed_component_ids,
        "pins_dropped": total_pins - len(ir.pins),
        "pin_ids_dropped": removed_pin_ids,
        "pins_dropped_missing_number": len(missing_number_pin_ids),
        "pin_ids_dropped_missing_number": missing_number_pin_ids,
        "pins_dropped_unowned": len(unowned_pin_ids),
        "pin_ids_dropped_unowned": unowned_pin_ids,
        "nets_affected": nets_affected,
        "affected_net_ids": sorted(affected_net_ids),
        "nets_dropped": nets_dropped,
        "dropped_net_ids": dropped_net_ids,
    }


def infer_one(
    input_path: str | Path,
    output_path: str | Path,
    *,
    provider: IRProvider,
    config: dict[str, Any],
    official_profile: str | Path | dict[str, Any] | None = None,
    evidence_sink: Any = None,
    deadline_at: float | None = None,
) -> InferenceResult:
    source = Path(input_path)
    ctx = RunContext.start(source, config, deadline_at=deadline_at)
    try:
        return _infer_one(
            input_path,
            output_path,
            provider=provider,
            config=config,
            official_profile=official_profile,
            evidence_sink=evidence_sink,
            _context=ctx,
        )
    except Exception as exc:
        setattr(exc, "context_manifest", ctx.manifest())
        # Attach the partial IR so failure artifacts can be saved for replay.
        # The PipelineError from _infer_one already carries partial_ir; don't
        # overwrite it with None.
        if not hasattr(exc, "partial_ir") or getattr(exc, "partial_ir", None) is None:
            partial_ir = getattr(exc, "_partial_ir", None)
            if partial_ir is None:
                for stage_data in reversed(ctx._stages if hasattr(ctx, '_stages') else []):
                    if isinstance(stage_data, dict) and "ir" in stage_data:
                        partial_ir = stage_data["ir"]
                        break
            setattr(exc, "partial_ir", partial_ir)
        raise


def _infer_one(
    input_path: str | Path,
    output_path: str | Path,
    *,
    provider: IRProvider,
    config: dict[str, Any],
    official_profile: str | Path | dict[str, Any] | None = None,
    evidence_sink: Any = None,
    _context: RunContext | None = None,
) -> InferenceResult:
    source = Path(input_path)
    destination = Path(output_path)
    ctx = _context or RunContext.start(source, config)
    graph_config = config.get("graph", {})
    palette_enabled = graph_config.get("palette_wire_filter_enabled", False)
    if type(palette_enabled) is not bool:
        raise ValueError("graph.palette_wire_filter_enabled must be a boolean")
    palette_context = None

    def _snapshot(stage: str) -> None:
        if evidence_sink is not None:
            evidence_sink(stage, ir)

    with ctx.stage("perception") as trace:
        ir = provider.build_ir(source, ctx)
        if ir.state == IRState.RAW:
            ir.state = IRState.PERCEIVED
        if ir.state != IRState.PERCEIVED:
            raise PipelineError(f"provider returned unexpected state {ir.state}; expected PERCEIVED")
        declared_versions = ir.metadata.get("model_versions", {})
        if isinstance(declared_versions, dict):
            ctx.model_versions.update(
                {
                    str(task): str(version)
                    for task, version in declared_versions.items()
                    if str(task) and str(version)
                }
            )
        provider_version = getattr(provider, "version", None)
        if provider_version:
            ctx.model_versions.setdefault("provider", str(provider_version))
        trace["object_count"] = len(ir.components) + len(ir.pins) + len(ir.segments)
    _snapshot("perception")

    if palette_enabled:
        with ctx.stage("wire_palette_prepare") as trace:
            palette_context = prepare_palette_wire(source, ir)
            trace["object_count"] = len(palette_context.palette["votes"])
            trace["message"] = palette_context.palette["reason"]

    with ctx.stage("association") as trace:
        associate_ocr(ir, config)
        ir.state = IRState.ASSOCIATED
        trace["object_count"] = len(ir.components) + len(ir.pins)
        report = ir.metadata.get("ocr_association", {})
        if isinstance(report, dict):
            diagnostics = report.get("diagnostics", {})
            if isinstance(diagnostics, dict):
                trace["message"] = f"ocr_association_diagnostics={diagnostics.get('total', 0)}"
    _snapshot("association")

    _snapshot("pre_prune")
    if _submission_mode(config) == "degraded":
        # Degraded output is explicit and local to this image. Drop objects
        # that cannot be represented safely (missing key / missing pin number /
        # unowned pin), recording deleted IDs and affected nets. Never invent
        # keys or pin numbers.
        _prune_unrepresentable(ir)

    # Junction decisions change physical connectivity, so selective reasoning
    # must finish before either graph is materialized.  Running it later would
    # leave physical/electrical nets stale even though the junction object had
    # changed kind.
    with ctx.stage("selective_reasoning") as trace:
        reasoner = _reasoner_from_config(config)
        run_selective_reasoning(ir, config, ctx, reasoner)
        report = ir.metadata.get("selective_reasoning", {})
        trace["object_count"] = (
            int(report.get("generated_tasks", 0)) if isinstance(report, dict) else 0
        )
        if isinstance(report, dict):
            trace["message"] = (
                f"routed={report.get('routed_tasks', 0)},"
                f"applied={report.get('applied_tasks', 0)},"
                f"cache_hits={report.get('cache_hits', 0)}"
            )

    graph_config = config.get("graph", {})
    association_config = config.get("association", {})
    pin_net_ranker_config = association_config.get("pin_net_ranker", {})
    if not isinstance(pin_net_ranker_config, dict):
        raise ValueError("association.pin_net_ranker must be an object")
    pin_net_ranker_enabled = pin_net_ranker_config.get("enabled", False)
    if type(pin_net_ranker_enabled) is not bool:
        raise ValueError("association.pin_net_ranker.enabled must be a boolean")
    pin_net_ranker_resource = str(
        pin_net_ranker_config.get("resource", "pin_net_ranker_v1.json")
    )
    internal_wire_filter_enabled = graph_config.get("internal_wire_filter_enabled", False)
    if type(internal_wire_filter_enabled) is not bool:
        raise ValueError("graph.internal_wire_filter_enabled must be a boolean")
    body_outline_filter_enabled = graph_config.get("body_outline_filter_enabled", False)
    if type(body_outline_filter_enabled) is not bool:
        raise ValueError("graph.body_outline_filter_enabled must be a boolean")
    glyph_wire_filter_enabled = graph_config.get("glyph_wire_filter_enabled", False)
    if type(glyph_wire_filter_enabled) is not bool:
        raise ValueError("graph.glyph_wire_filter_enabled must be a boolean")
    decorative_wire_filter_enabled = graph_config.get(
        "decorative_wire_filter_enabled", False
    )
    if type(decorative_wire_filter_enabled) is not bool:
        raise ValueError("graph.decorative_wire_filter_enabled must be a boolean")
    graph_kwargs = {
        "endpoint_tolerance_px": float(graph_config.get("endpoint_merge_tolerance_px", 3.0)),
        "collinear_merge_tolerance_px": float(graph_config.get("collinear_merge_tolerance_px", 0.0)),
        "collinear_gap_merge_tolerance_px": float(graph_config.get("collinear_gap_merge_tolerance_px", 0.0)),
        "pin_snap_min_px": float(association_config.get("pin_wire_snap_min_px", 2.0)),
        "pin_snap_tolerance_px": float(
            association_config.get(
                "pin_wire_snap_max_px",
                graph_config.get("pin_snap_tolerance_px", 6.0),
            )
        ),
        "pin_point_projection_max_px": float(graph_config.get("pin_point_projection_max_px", 0.0)),
        "unique_margin": float(association_config.get("unique_margin", 1.5)),
        "gap_repair_max_px": float(graph_config.get("gap_repair_max_px", 12.0)),
    }

    def rebuild_physical_graph():
        return build_physical_graph(ir, **graph_kwargs)

    with ctx.stage("physical_graph") as trace:
        if palette_context is not None:
            apply_palette_wire_filter(ir, palette_context)
        if body_outline_filter_enabled:
            outline_report = filter_body_outline_wire_segments(
                ir,
                tolerance_px=float(graph_config.get("body_outline_tolerance_px", 4.0)),
                coverage=float(graph_config.get("body_outline_coverage", 0.5)),
                min_edge_span_px=float(graph_config.get("body_outline_min_edge_span_px", 20.0)),
            )
            trace["message"] = (
                trace.get("message", "")
                + f"body_outline_suppressed={outline_report['suppressed']}"
            )
        if glyph_wire_filter_enabled:
            glyph_report = filter_glyph_wire_segments(
                ir,
                margin_px=float(graph_config.get("glyph_wire_margin_px", 2.0)),
            )
            trace["message"] = (
                trace.get("message", "")
                + f"glyph_wire_suppressed={glyph_report['suppressed']},"
                f"glyph_junctions_suppressed={glyph_report['junctions_suppressed']}"
            )
        ir.physical_nets, snap_decisions = rebuild_physical_graph()

        if internal_wire_filter_enabled:
            filter_report = filter_internal_wire_segments(
                ir,
                containment_threshold=float(
                    graph_config.get("internal_wire_containment_threshold", 0.80)
                ),
                pin_radius=float(
                    graph_config.get(
                        "internal_wire_pin_anchor_radius_px",
                        association_config.get(
                            "pin_wire_snap_max_px",
                            graph_config.get("pin_snap_tolerance_px", 6.0),
                        ),
                    )
                ),
                label_radius=float(
                    graph_config.get("internal_wire_label_anchor_radius_px", 8.0)
                ),
            )
            ir.metadata["internal_wire_filter"] = filter_report
            if filter_report["deleted_segments"]:
                # Rebuild derived graph evidence after changing perceived edges.
                ir.physical_nets = []
                ir.electrical_nets = []
                for pin in ir.pins:
                    pin.evidence = [
                        evidence
                        for evidence in pin.evidence
                        if not (
                            isinstance(evidence.metadata, dict)
                            and evidence.metadata.get("kind") == "pin_net_evidence"
                        )
                    ]
                for label in ir.net_labels:
                    label.attached_physical_net_id = None
                for key in (
                    "connection_evidence",
                    "connection_params",
                    "snap_decisions",
                    "pin_net_evidence",
                    "label_merge_events",
                ):
                    ir.metadata.pop(key, None)
                ir.physical_nets, snap_decisions = rebuild_physical_graph()
        if decorative_wire_filter_enabled:
            decorative_report = filter_decorative_wire_segments(
                ir,
                span_fraction=float(
                    graph_config.get("decorative_span_fraction", 0.8)
                ),
                rect_min_area_px=float(
                    graph_config.get("decorative_rect_min_area_px", 500.0)
                ),
                interior_inset_px=float(
                    graph_config.get("decorative_interior_inset_px", 5.0)
                ),
            )
            trace["message"] = (
                trace.get("message", "")
                + f"decorative_wire_suppressed={decorative_report['suppressed']}"
            )
            if decorative_report["suppressed"]:
                ir.physical_nets = []
                ir.electrical_nets = []
                for pin in ir.pins:
                    pin.evidence = [
                        evidence
                        for evidence in pin.evidence
                        if not (
                            isinstance(evidence.metadata, dict)
                            and evidence.metadata.get("kind") == "pin_net_evidence"
                        )
                    ]
                for label in ir.net_labels:
                    label.attached_physical_net_id = None
                for key in (
                    "connection_evidence",
                    "connection_params",
                    "snap_decisions",
                    "pin_net_evidence",
                    "label_merge_events",
                ):
                    ir.metadata.pop(key, None)
                ir.physical_nets, snap_decisions = rebuild_physical_graph()
        ir.metadata["snap_decisions"] = [
            {
                "pin_id": decision.pin_id,
                "status": decision.status,
                "segment_id": decision.segment_id,
                "cost": decision.cost,
                "margin": decision.margin,
            }
            for decision in snap_decisions
        ]
        if pin_net_ranker_enabled:
            ranker_report = apply_pin_net_ranker(
                ir, resource_name=pin_net_ranker_resource
            )
            trace["message"] = (
                trace.get("message", "")
                + f"pin_net_ranker_changed={ranker_report['changed_pins']}"
            )
        ir.state = IRState.GRAPHED
        trace["object_count"] = len(ir.physical_nets)
        if internal_wire_filter_enabled:
            trace["message"] = (
                f"internal_filter_deleted="
                f"{ir.metadata.get('internal_wire_filter', {}).get('deleted_segments', 0)}"
            )

    label_config = config.get("labels", {})
    with ctx.stage("electrical_graph") as trace:
        ir.electrical_nets = build_electrical_nets(
            ir,
            merge_signal_labels=bool(label_config.get("merge_signal_labels", True)),
            merge_power_labels=bool(label_config.get("merge_power_labels", False)),
            merge_ground_labels=bool(label_config.get("merge_ground_labels", False)),
            min_text_score=float(label_config.get("min_text_score", 0.75)),
            merge_threshold=float(label_config.get("merge_threshold", 0.85)),
        )
        ir.state = IRState.SEMANTIC
        trace["object_count"] = len(ir.electrical_nets)

    with ctx.stage("validation") as trace:
        validators = (
            SchemaValidator(),
            ReferenceValidator(),
            GeometryValidator(),
            GraphValidator(
                endpoint_tolerance_px=float(
                    graph_config.get("endpoint_merge_tolerance_px", 3.0)
                ),
                collinear_merge_tolerance_px=float(
                    graph_config.get("collinear_merge_tolerance_px", 0.0)
                ),
                collinear_gap_merge_tolerance_px=float(
                    graph_config.get("collinear_gap_merge_tolerance_px", 0.0)
                ),
                gap_repair_max_px=float(graph_config.get("gap_repair_max_px", 12.0)),
            ),
        )
        repaired = repair_ir_once(ir, config_hash=ctx.config_hash)
        validator_issues = stable_diagnostics(
            issue for validator in validators for issue in validator.validate(repaired, ctx)
        )
        diagnostics = merge_diagnostics(repaired.diagnostics, validator_issues)
        repaired.diagnostics = diagnostics
        ir = repaired
        errors = [item for item in diagnostics if item.severity == "ERROR"]
        trace["object_count"] = len(diagnostics)
        if errors:
            if _is_debug_execution(config):
                # Debug mode: log quality errors but don't block serialization.
                # Debug providers (e.g., dataset_debug) naturally produce IR
                # with quality gaps; blocking them prevents debug output.
                ir.state = IRState.VALIDATED
            else:
                err = PipelineError(f"validation failed with {len(errors)} error(s)", diagnostics)
                err.partial_ir = ir
                raise err
        elif ir.state in {IRState.SEMANTIC, IRState.VALIDATED}:
            ir.state = IRState.VALIDATED

    with ctx.stage("serialization") as trace:
        serialize_mode = (
            "degraded" if _is_debug_execution(config) else _submission_mode(config)
        )
        output, quality = serialize_official_with_quality(
            ir, official_profile, mode=serialize_mode
        )
        output_hash = official_json_hash(output, official_profile)
        atomic_write_official_json(output, destination, official_profile)
        trace["object_count"] = len(output["components"]) + len(output["pins"]) + len(output["nets"])
        ir.state = IRState.SERIALIZED
    _snapshot("final")

    return InferenceResult(
        input_path=source,
        output_path=destination,
        output=output,
        output_hash=output_hash,
        ir=ir,
        diagnostics=diagnostics,
        context=ctx,
        quality=quality,
    )


def _reasoner_from_config(config: dict[str, Any]):
    qwen = config.get("qwen", {})
    if not isinstance(qwen, dict) or not bool(qwen.get("enabled", False)):
        return DisabledReasoner()
    backend = str(qwen.get("backend", "disabled")).strip().lower()
    if backend == "candidate_only":
        return CandidateOnlyReasoner(
            minimum_score=float(qwen.get("candidate_min_score", 0.75)),
            minimum_margin=float(qwen.get("candidate_min_margin", 0.10)),
        )
    if backend == "http":
        endpoint = str(qwen.get("endpoint", "")).strip()
        model = str(qwen.get("model", "")).strip()
        if not endpoint or not model:
            raise PipelineError(
                "qwen backend=http requires non-empty qwen.endpoint and qwen.model"
            )
        return HTTPQwenReasoner(
            endpoint=endpoint,
            model=model,
            timeout_seconds=float(qwen.get("timeout_seconds", 12.0)),
            api_key=(str(qwen["api_key"]) if qwen.get("api_key") else None),
            enable_thinking=bool(qwen.get("enable_thinking", False)),
            send_images=bool(qwen.get("send_images", False)),
        )
    if backend == "disabled":
        raise PipelineError(
            "qwen.enabled=true requires an explicit qwen.backend of candidate_only or http"
        )
    raise PipelineError(f"unsupported qwen backend {backend!r}")


def infer_batch(
    inputs: list[str | Path],
    output_dir: str | Path,
    *,
    provider: IRProvider,
    config: dict[str, Any],
    official_profile: str | Path | dict[str, Any] | None = None,
    continue_on_error: bool = True,
    evidence_sink: Any = None,
) -> tuple[list[InferenceResult], list[dict[str, Any]]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    ordered_inputs = sorted(
        (Path(item) for item in inputs),
        key=lambda path: (path.stem.casefold(), path.as_posix().casefold(), path.as_posix()),
    )
    # Official output names are derived from the stem.  Windows filesystems
    # are case-insensitive, so a case-sensitive check would allow Board.png
    # and board.png to overwrite one another during serialization.
    duplicate_stems = sorted(
        stem
        for stem, count in Counter(path.stem.casefold() for path in ordered_inputs).items()
        if count > 1
    )
    if duplicate_stems:
        raise ValueError(
            "batch input stems would overwrite output files; duplicate stems="
            f"{duplicate_stems}"
        )
    results: list[InferenceResult] = []
    errors: list[dict[str, Any]] = []
    for input_item in ordered_inputs:
        output_path = destination / f"{input_item.stem}.json"
        try:
            results.append(
                infer_one(
                    input_item,
                    output_path,
                    provider=provider,
                    config=config,
                    official_profile=official_profile,
                    evidence_sink=evidence_sink,
                )
            )
        except Exception as exc:
            context_manifest = getattr(exc, "context_manifest", {})
            if not isinstance(context_manifest, dict):
                context_manifest = {}
            trace = context_manifest.get("trace", [])
            last_stage = None
            if isinstance(trace, list) and trace:
                last = trace[-1]
                if isinstance(last, dict):
                    last_stage = last.get("stage")
            record = {
                "error_id": _error_id(
                    input_item,
                    error_type=type(exc).__name__,
                    message=str(exc),
                    config_hash=context_manifest.get("config_hash"),
                ),
                "input": str(input_item),
                "output": str(output_path),
                "input_sha256": _safe_file_hash(input_item),
                "config_hash": context_manifest.get("config_hash"),
                "run_id": context_manifest.get("run_id"),
                "failed_stage": last_stage,
                "trace": trace if isinstance(trace, list) else [],
                "error_type": type(exc).__name__,
                "message": str(exc),
                "diagnostics": _diagnostic_payloads(getattr(exc, "diagnostics", [])),
            }
            errors.append(record)
            if not continue_on_error:
                raise
    if errors:
        error_path = destination / "error_manifest.json"
        _atomic_write_json(error_path, errors)
    return results, errors


def _error_id(
    input_path: Path,
    *,
    error_type: str,
    message: str,
    config_hash: Any,
) -> str:
    """Build a stable identifier for one failed image/config execution."""

    payload = json.dumps(
        {
            "config_hash": config_hash,
            "error_type": error_type,
            "input": str(input_path),
            "message": message,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_file_hash(path: Path) -> str | None:
    try:
        return stable_file_hash(path)
    except (OSError, ValueError):
        # The input may disappear between discovery and execution.  Preserve
        # the failure record without masking the original pipeline exception.
        return None


def _diagnostic_payloads(value: Any) -> list[dict[str, Any]]:
    """Serialize failure diagnostics without masking the original exception."""

    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        value = [value]
    payloads: list[dict[str, Any]] = []
    for item in value:
        model_dump = getattr(item, "model_dump", None)
        if callable(model_dump):
            try:
                rendered = model_dump(mode="json")
            except Exception:  # pragma: no cover - defensive failure path
                rendered = None
            if isinstance(rendered, dict):
                payloads.append(rendered)
                continue
        if isinstance(item, dict):
            payloads.append(dict(item))
        else:
            payloads.append({"message": str(item)})
    return payloads


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
