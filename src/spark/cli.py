"""``spark`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path.cwd()


def _data(args: argparse.Namespace) -> int:
    from spark.data import cnrpark

    root = Path(args.root)
    if args.action == "download":
        which = ["patches", "splits"] + (["full"] if args.full else [])
        cnrpark.download(root, which)
    elif args.action == "cache":
        cnrpark.build_cache(root)
    return 0


def _bench(args: argparse.Namespace) -> int:
    if args.action == "annotate":
        from spark.bench.replay import annotate

        print(annotate(Path(args.root), Path(args.model[0]), args.camera, Path(args.out),
                       args.frame))
        return 0
    if args.action == "replay":
        from spark.bench.replay import main as replay

        models = [Path(m) for m in args.model]
        res = replay(Path(args.root), models, Path(args.replay_out))
        for m in models:
            print(m.stem, json.dumps(res[m.stem]["overall"]))
        return 0
    if args.action == "run":
        from spark.bench.run import run

        run(Path(args.root), Path(args.results), Path(args.models), quick=args.quick,
            epochs=args.epochs)
    from spark.bench.report import write_report

    write_report(Path(args.results), Path(args.report))
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from spark.server.app import create_app
    from spark.server.settings import Settings

    settings = Settings.from_env(db_path=args.db, lots_dir=args.lots)
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")
    return 0


def _edge(args: argparse.Namespace) -> int:
    from spark.edge.agent import AgentConfig, run_agent

    cfg = AgentConfig(
        server=args.server,
        lot_path=Path(args.lot),
        model_path=Path(args.model),
        source=args.source,
        token=args.token,
        fps=args.fps,
        speed=args.speed,
        loop=args.loop,
    )
    run_agent(cfg)
    return 0


def _demo(args: argparse.Namespace) -> int:
    from spark.demo import run_demo

    return run_demo(port=args.port, synthetic=args.synthetic, root=Path(args.root),
                    camera=args.camera, speed=args.speed, days=args.days)


def _sim(args: argparse.Namespace) -> int:
    from spark.sim.evaluate import compare_policies

    out = compare_policies(days=args.days, seeds=args.seeds)
    text = json.dumps(out, indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spark", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("data", help="download and cache CNRPark+EXT")
    d.add_argument("action", choices=["download", "cache"])
    d.add_argument("--root", default="data")
    d.add_argument("--full", action="store_true", help="also fetch full frames (1 GB)")
    d.set_defaults(func=_data)

    b = sub.add_parser("bench", help="run the benchmark or rebuild its report")
    b.add_argument("action", choices=["run", "report", "replay", "annotate"])
    b.add_argument("--camera", type=int, default=8, help="for annotate")
    b.add_argument("--frame", default="", help="for annotate, e.g. 2016-01-08_1407")
    b.add_argument("--out", default="docs/img/camera8.jpg", help="for annotate")
    b.add_argument("--model", nargs="+", default=["models/default.onnx"], help="for replay")
    b.add_argument("--replay-out", default="results/replay.json")
    b.add_argument("--root", default="data")
    b.add_argument("--results", default="results/benchmark.json")
    b.add_argument("--report", default="docs/benchmark.md")
    b.add_argument("--models", default="models")
    b.add_argument("--epochs", type=int, default=8)
    b.add_argument("--quick", action="store_true")
    b.set_defaults(func=_bench)

    s = sub.add_parser("serve", help="run the availability and pricing server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--db", default="spark.db")
    s.add_argument("--lots", default="configs/lots")
    s.set_defaults(func=_serve)

    e = sub.add_parser("edge", help="run the camera agent")
    e.add_argument("--server", default="http://127.0.0.1:8000")
    e.add_argument("--lot", required=True)
    e.add_argument("--model", default="models/default.onnx")
    e.add_argument("--source", required=True,
                   help="camera index, video file, RTSP URL or folder of frames")
    e.add_argument("--token", default="")
    e.add_argument("--fps", type=float, default=2.0)
    e.add_argument("--speed", type=float, default=1.0,
                   help="replay speed for timestamped frame folders")
    e.add_argument("--loop", action="store_true")
    e.set_defaults(func=_edge)

    m = sub.add_parser("demo", help="server, edge agent and apps in one process")
    m.add_argument("--port", type=int, default=8000)
    m.add_argument("--root", default="data")
    m.add_argument("--camera", type=int, default=8)
    m.add_argument("--speed", type=float, default=600.0)
    m.add_argument("--days", type=float, default=None,
                   help="synthetic only: stop after this many simulated days")
    m.add_argument("--synthetic", action="store_true",
                   help="simulated occupancy, no dataset needed")
    m.set_defaults(func=_demo)

    q = sub.add_parser("sim", help="compare pricing policies in simulation")
    q.add_argument("--days", type=int, default=28)
    q.add_argument("--seeds", type=int, default=5)
    q.add_argument("--out", default="")
    q.set_defaults(func=_sim)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
