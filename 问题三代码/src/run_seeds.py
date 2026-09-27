from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential multi-seed training launcher")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[20260924, 20260925, 20260926, 20260927, 20260928]
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted config key=value; may be repeated and is forwarded to every run",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        output = output_root / f"seed_{seed}"
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--config",
            args.config,
            "--data",
            args.data,
            "--output",
            str(output),
            "--seed",
            str(seed),
            "--device",
            args.device,
        ]
        for override in args.override:
            command.extend(["--set", override])
        print(f"\n===== Training seed {seed} -> {output} =====", flush=True)
        subprocess.run(command, check=True)
    print("\nAll requested seeds completed successfully.")


if __name__ == "__main__":
    main()
