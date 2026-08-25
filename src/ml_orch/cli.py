"""

Every command here is a thin shell over `pipelines.py`,
which is the same thing the Dagster assets call, so there is exactly one
implementation of each pipeline no matter how it was triggered.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from .data import source_from_config
from .pipelines import load_config, run_lifecycle, run_triage
from .registry import ModelRegistry

log = logging.getLogger("ml_orch")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.pipeline == "lifecycle":
        chunk = None
        if args.chunk:
            chunks = source_from_config(cfg["source"]).chunks()
            match = [c for c in chunks if c.chunk_id == args.chunk or str(c.path) == args.chunk]
            if not match:
                log.error("no chunk matching %r; available: %s",
                          args.chunk, [c.chunk_id for c in chunks])
                return 2
            chunk = match[0]
        result = run_lifecycle(cfg, chunk=chunk)
        log.info("=> %s (%s)", result.status, result.model_version or "no model")
    else:
        result = run_triage(cfg)
        log.info("=> %s: %d/%d shortlisted -> %s", result.status,
                 result.n_shortlisted, result.n_candidates, result.shortlist_path)

    # A failed gate is a normal outcome. Exit 0 and say so loudly; a non-zero
    # exit would tell a scheduler that the *pipeline* broke, which it did not.
    return 0


def cmd_backfill(args: argparse.Namespace) -> int:
    """Replay every chunk in arrival order. Idempotent, so it is safe to rerun."""
    cfg = load_config(args.config)
    chunks = source_from_config(cfg["source"]).chunks()
    for c in chunks:
        result = run_lifecycle(cfg, chunk=c)
        log.info("%s -> %s", c.chunk_id, result.status)
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Trigger on data arrival, via filesystem events.

    The same `run_lifecycle` the CLI and the Dagster sensor call. Debounced,
    because a file being written is several events, and idempotent on content
    hash, because a debounce is a heuristic and correctness should not rest on
    one.
    """
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    cfg = load_config(args.config)
    watch_cfg = cfg.get("trigger", {}).get("watch", {})
    path = Path(watch_cfg.get("path", cfg["source"]["path"]))
    debounce = float(watch_cfg.get("debounce_seconds", 5))
    path.mkdir(parents=True, exist_ok=True)

    class Handler(FileSystemEventHandler):
        def __init__(self) -> None:
            self.pending: dict[str, float] = {}

        def on_any_event(self, event) -> None:
            if event.is_directory or not str(event.src_path).endswith(".csv"):
                return
            self.pending[str(event.src_path)] = time.time()

    handler = Handler()
    observer = Observer()
    observer.schedule(handler, str(path), recursive=False)
    observer.start()
    log.info("watching %s (debounce %.1fs) — ctrl-c to stop", path, debounce)
    try:
        while True:
            time.sleep(1.0)
            now = time.time()
            ready = [p for p, t in handler.pending.items() if now - t >= debounce]
            for p in ready:
                handler.pending.pop(p, None)
                log.info("data arrival: %s", p)
                chunks = source_from_config(cfg["source"]).chunks()
                match = [c for c in chunks if str(c.path) == p]
                if match:
                    result = run_lifecycle(cfg, chunk=match[0])
                    log.info("%s -> %s", match[0].chunk_id, result.status)
    except KeyboardInterrupt:
        log.info("stopping")
    finally:
        observer.stop()
        observer.join()
    return 0


def cmd_registry(args: argparse.Namespace) -> int:
    reg = ModelRegistry(args.root)
    if args.action == "list":
        prod = reg.production()
        print(f"{'version':38} {'stage':11} {'score':>8}  data")
        for rec in reg.records():
            score = rec.metrics.get("golden", {}).get("normalized_score")
            marker = " *" if prod and rec.version == prod.version else "  "
            print(f"{rec.version:38} {rec.stage:11} "
                  f"{score if score is None else f'{score:8.4f}'}  "
                  f"{rec.data_version[:24]}{marker}")
        return 0
    if args.action == "rollback":
        rec = reg.rollback()
        log.info("rolled back to %s", rec.version)
        return 0
    if args.action == "show":
        rec = reg.get(args.version) if args.version else reg.production()
        if rec is None:
            log.error("no such model")
            return 2
        print(json.dumps({"version": rec.version, "stage": rec.stage,
                          "versions": rec.versions, "metrics": rec.metrics,
                          "gate": rec.gate_decision}, indent=2))
        return 0
    return 2


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="ml-orch", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a pipeline once, on demand")
    run.add_argument("pipeline", choices=["lifecycle", "triage"])
    run.add_argument("--config", required=True)
    run.add_argument("--chunk", help="chunk id or path (lifecycle; default: newest)")
    run.set_defaults(func=cmd_run)

    bf = sub.add_parser("backfill", help="replay every chunk in arrival order")
    bf.add_argument("--config", required=True)
    bf.set_defaults(func=cmd_backfill)

    watch = sub.add_parser("watch", help="run the lifecycle on new data arrival")
    watch.add_argument("--config", required=True)
    watch.set_defaults(func=cmd_watch)

    reg = sub.add_parser("registry", help="inspect and control the registry")
    reg.add_argument("action", choices=["list", "rollback", "show"])
    reg.add_argument("--root", default="registry")
    reg.add_argument("--version", help="for `show`; defaults to production")
    reg.set_defaults(func=cmd_registry)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
