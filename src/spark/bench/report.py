"""Turn results/benchmark.json into the tables in docs/benchmark.md and a Pareto plot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPLIT_ORDER = ("standard", "unseen-cameras", "sun-to-cloud-and-rain")


def _pct(x: float | None, digits: int = 1) -> str:
    return "–" if x is None else f"{100 * x:.{digits}f}"


def _model_order(name: str) -> tuple[int, float, int]:
    if name == "edge-energy":
        return (0, 0, 0)
    if name == "hog-linear":
        return (1, 0, 0)
    # sparknet-w0.5-r48
    _, w, r = name.split("-")[:3]
    return (2, float(w[1:]), int(r[1:]))


def accuracy_table(res: dict[str, Any]) -> str:
    acc = res.get("accuracy", {})
    models = sorted({m for s in acc.values() for m in s}, key=_model_order)
    splits = [s for s in SPLIT_ORDER if s in acc]
    head = "| Model | " + " | ".join(splits) + " |\n|---|" + "---:|" * len(splits) + "\n"
    rows = []
    for m in models:
        cells = []
        for s in splits:
            e = acc[s].get(m)
            cells.append("–" if e is None else f"{_pct(e['accuracy'])} ({_pct(e['auc'], 2)})")
        rows.append(f"| `{m}` | " + " | ".join(cells) + " |")
    return head + "\n".join(rows) + "\n"


def deployed_table(res: dict[str, Any]) -> str:
    dep = res.get("deployed", {})
    head = ("| Artefact | Input | Params | MACs / patch | File | Accuracy | "
            "1 bay (ms) | 32 bays (ms) |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for name in sorted(dep, key=lambda n: (*_model_order(n.removesuffix("-int8")),
                                           n.endswith("-int8"))):
        d = dep[name]
        size, lat = d.get("size", {}), {x["batch"]: x for x in d.get("latency", [])}
        macs = size.get("macs")
        rows.append(
            f"| `{name}` | {size.get('input_size', '–')} | {size.get('params', 0):,} | "
            f"{'–' if macs is None else f'{macs / 1e6:.2f} M'} | "
            f"{size.get('bytes', 0) / 1024:.0f} KB | {_pct(d['accuracy']['accuracy'])} | "
            f"{lat[1]['median_ms']:.2f} | {lat[32]['median_ms']:.2f} |"
        )
    return head + "\n".join(rows) + "\n"


def robustness_table(res: dict[str, Any], kind: str) -> str:
    dep = res.get("deployed", {})
    names = [n for n in sorted(dep, key=lambda n: _model_order(n.removesuffix("-int8")))
             if not n.endswith("-int8")]
    if not names:
        return ""
    levels = list(dep[names[0]]["robustness"][kind])
    label = "Occluded share" if kind == "occlusion" else "Gamma"
    head = ("| Model | " + " | ".join(levels) + " |\n|---|" + "---:|" * len(levels) + "\n")
    head = head.replace("| Model |", f"| Model ({label}) |")
    rows = [f"| `{n}` | " + " | ".join(_pct(dep[n]["robustness"][kind][lv]) for lv in levels)
            + " |" for n in names]
    return head + "\n".join(rows) + "\n"


def pareto_plot(res: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dep = res.get("deployed", {})
    acc = res.get("accuracy", {}).get("unseen-cameras", {})
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    default = ""
    meta_path = out.parent.parent.parent / "models" / "default.json"
    if meta_path.exists():
        default = str(json.loads(meta_path.read_text()).get("name", "")).removesuffix("-int8")
    for name, d in dep.items():
        if not name.startswith("sparknet") or name.endswith("-int8") or name not in acc:
            continue
        macs = d["size"]["macs"] / 1e6
        y = 100 * acc[name]["accuracy"]
        chosen = name == default
        ax.scatter(macs, y, s=70 if chosen else 30, zorder=3,
                   color="#d97706" if chosen else "#2563eb")
        label = name.removeprefix("sparknet-") + ("  (architecture shipped)" if chosen else "")
        ax.annotate(label, (macs, y), textcoords="offset points",
                    xytext=(-110, 8) if chosen else (5, -10), fontsize=7,
                    color="#d97706" if chosen else None)
    for base, colour in (("hog-linear", "#b7791f"), ("edge-energy", "#d64545")):
        if base in acc:
            ax.axhline(100 * acc[base]["accuracy"], color=colour, ls="--", lw=1,
                       label=f"{base} baseline")
    ax.set_xscale("log")
    from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

    ticks = [0.1, 0.2, 0.5, 1, 2, 5]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.08, 8)
    ax.set_xlabel("Multiply-accumulates per patch (millions, log scale)")
    ax.set_ylabel("Accuracy on unseen cameras (%)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _replay_order(name: str) -> tuple[int, float, int, bool]:
    base = name.removesuffix("-int8")
    if base.startswith(("sparknet", "edge", "hog")):
        return (*_model_order(base), name.endswith("-int8"))
    return (3, 0.0, 0, False)


def replay_table(replay: dict[str, Any]) -> str:
    head = ("| Artefact | Frames | Bays scored | Bay accuracy | Free-count error (mean, bays) | "
            "Count exact | Within 1 | Within 2 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    rows = []
    for name in sorted(replay, key=_replay_order):
        o = replay[name]["overall"]
        model = replay[name].get("model", name)
        label = f"`{name}`" if model == name else f"`{name}` (`{model}`)"
        rows.append(f"| {label} | {o['frames']:,} | {o['bays_scored']:,} | "
                    f"{_pct(o['bay_accuracy'])}% | {o['free_count_mae']:.2f} | "
                    f"{_pct(o['frames_exact'], 0)}% | {_pct(o['frames_within_1'], 0)}% | "
                    f"{_pct(o['frames_within_2'], 0)}% |")
    return head + "\n".join(rows) + "\n"


def write_report(results_path: Path, report_path: Path) -> None:
    res = json.loads(results_path.read_text())
    tables = {
        "ACCURACY": accuracy_table(res),
        "DEPLOYED": deployed_table(res),
        "OCCLUSION": robustness_table(res, "occlusion"),
        "LIGHTING": robustness_table(res, "gamma"),
    }
    splits = res.get("splits", {})
    split_rows = [
        f"| `{k}` | {v['description']} | {v['train']:,} | {v['val']:,} | {v['test']:,} |"
        for k, v in splits.items()
    ]
    tables["SPLITS"] = ("| Protocol | What it tests | Train | Val | Test |\n"
                        "|---|---|---:|---:|---:|\n" + "\n".join(split_rows) + "\n")
    replay_path = results_path.parent / "replay.json"
    if replay_path.exists():
        tables["REPLAY"] = replay_table(json.loads(replay_path.read_text()))
    run = res.get("run", {})
    tables["RUN"] = (f"Run started {run.get('started', '?')}, finished {run.get('finished', '?')}"
                     f", torch {run.get('torch', '?')}, {run.get('epochs', '?')} epochs per model"
                     f"{', QUICK MODE (subsampled)' if run.get('quick') else ''}.\n")
    text = report_path.read_text()
    for key, table in tables.items():
        start, end = f"<!-- {key} -->", f"<!-- /{key} -->"
        if start in text and end in text:
            a, rest = text.split(start, 1)
            _, b = rest.split(end, 1)
            text = f"{a}{start}\n{table}{end}{b}"
    report_path.write_text(text)
    pareto_plot(res, report_path.parent / "img" / "pareto.png")
