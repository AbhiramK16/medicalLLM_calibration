"""Experiment configuration loaded from ``configs/experiment.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from calibration_collapse.experiments.probe_isolation import PatientAgentConfig

DEFAULT_CONFIG_PATH = Path("configs/experiment.yaml")
DEFAULT_MANIFEST_PATH = "data/cases.json"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    experiment_name: str = "calibration-collapse"
    dataset: str = "agentclinic-medqa"
    backbones: tuple[str, ...] = ()
    seeds: tuple[int, ...] = (0,)
    case_ids: tuple[str, ...] = ()
    case_count: int | None = None
    output_dir: str = "results/raw_outputs"
    turn_file: str = "turns.jsonl"
    manifest_path: str = DEFAULT_MANIFEST_PATH
    patient_agent: PatientAgentConfig | None = None

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.turn_file

    def run_id(self, backbone: str, seed: int) -> str:
        return f"{self.experiment_name}:{backbone}:{seed}"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    output = raw.get("output", {}) or {}
    patient = raw.get("patient_agent", {}) or {}
    backbone_entries = raw.get("backbones", [])
    backbones = tuple(
        str(entry["model"]) if isinstance(entry, dict) else str(entry)
        for entry in backbone_entries
    )
    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "calibration-collapse"),
        dataset=(raw.get("dataset", {}) or {}).get("name", "agentclinic-medqa"),
        backbones=backbones,
        seeds=tuple(int(seed) for seed in raw.get("seeds", [0])),
        case_ids=tuple(str(case_id) for case_id in raw.get("case_ids", [])),
        case_count=raw.get("case_count"),
        output_dir=output.get("directory", "results/raw_outputs"),
        turn_file=output.get("turn_file", "turns.jsonl"),
        patient_agent=PatientAgentConfig(
            model=str(patient["model"]),
            temperature=float(patient["temperature"]),
            max_output_tokens=int(patient["max_output_tokens"]),
            prompt_version=str(patient["prompt_version"]),
            bias=str(patient.get("bias", "none")),
        ) if patient else None,
    )


def select_cases(loader, config: ExperimentConfig) -> list[tuple[int, dict]]:
    """Return (line_index, case) pairs selected by config.case_ids or case_count."""
    if config.case_ids:
        for raw_id in config.case_ids:
            if not raw_id.isdigit() or int(raw_id) >= len(loader.data):
                raise ValueError(
                    f"unknown case_id {raw_id!r} (have {len(loader.data)} cases)"
                )
        return [(int(raw_id), loader.getID(int(raw_id))) for raw_id in config.case_ids]
    count = config.case_count if config.case_count is not None else len(loader.data)
    if count < 1 or count > len(loader.data):
        raise ValueError(f"case_count {count} out of range (have {len(loader.data)} cases)")
    return [(line_index, case) for line_index, case in enumerate(loader.data[:count])]
