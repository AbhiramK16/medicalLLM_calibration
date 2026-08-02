# Medical LLM Calibration

Research code for measuring whether diagnostic confidence remains aligned with
correctness across turns in simulated clinical dialogue.

The experiment compares three pipelines on matched cases:

1. `sequential` — a doctor agent gathers evidence through dialogue.
2. `randomized_order` — the same evidence is revealed cumulatively in a
   reproducibly shuffled order.
3. `static` — the complete vignette is shown in one step.

## Frozen study scope

- **Cases:** 100 AgentClinic-MedQA cases.
- **Standard instruction backbone:** `openai/gpt-4o-mini`.
- **Reasoning backbone:** `openai/o4-mini`.
- **Dialogue length:** six prediction turns per case for `sequential` and
  `randomized_order`.
- **Static length:** one full-vignette prediction per case.
- **Turn bins:** for dialogue turn `t` out of `T`, `early` means
  `t/T <= 1/3`, `middle` means `1/3 < t/T <= 2/3`, and `late` means
  `t/T > 2/3`.

With six dialogue turns, the bins are balanced:

| Bin | Turns |
|---|---|
| Early | 1–2 |
| Middle | 3–4 |
| Late | 5–6 |

Static predictions use the separate `static` label and are not placed into a
dialogue bin.

All pipelines must write the same versioned turn-level JSONL record defined in
`src/calibration_collapse/evaluation/schema.py`. See
`docs/turn_logging_schema.md` for field meanings and examples.

## Repository structure

```text
calibration-collapse/
├── README.md
├── configs/
│   └── experiment.yaml
├── src/
│   └── calibration_collapse/
│       ├── datasets/
│       │   ├── agentclinic.py
│       │   └── mint.py
│       ├── models/
│       │   └── run_model.py
│       ├── evaluation/
│       │   ├── schema.py
│       │   ├── calibration.py
│       │   ├── metrics.py
│       │   └── confidence.py
│       ├── experiments/
│       │   ├── sequential.py
│       │   ├── randomized_order.py
│       │   └── static.py
│       └── analysis/
│           ├── plots.py
│           └── statistics.py
├── results/
│   ├── figures/
│   ├── tables/
│   └── raw_outputs/
└── paper/
```

- `configs` freezes experiment choices separately from code.
- `src/calibration_collapse/datasets` converts benchmarks into a common case
  representation.
- `src/calibration_collapse/models` contains the provider-independent model-call
  interface.
- `src/calibration_collapse/experiments` contains the three
  evidence-presentation pipelines.
- `src/calibration_collapse/evaluation` owns confidence parsing, metrics,
  calibration, and the shared logging contract.
- `src/calibration_collapse/analysis` creates summaries and figures.
- `results` contains generated outputs and is ignored except for its folders.
- `paper` contains manuscript material.

## Local setup

Python 3.11 or newer is required.

Copy `.env.example` to `.env` when API access is added.

## Collaboration

## Other
This repository is for benchmark research.

## License
To be filled.
