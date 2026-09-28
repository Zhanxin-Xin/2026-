#!/usr/bin/env python3
"""Print a few augmented samples for manual pipeline inspection."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
local_deps = PROJECT_ROOT / ".python_deps"
if local_deps.is_dir():
    sys.path.insert(0, str(local_deps))

from data.dataset import MOSEIAlignedDataset
from data.missing_augmentation import ContinuousMissingAugmentation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="train")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    augmentation = ContinuousMissingAugmentation(seed=args.seed)
    dataset = MOSEIAlignedDataset(args.data_path, split=args.split, transform=augmentation)
    rng = random.Random(args.seed)
    indices = rng.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    for index in indices:
        sample = dataset[index]
        print(f"id={sample['id']} missing_type={sample['missing_type']} "
              f"modalities={sample['missing_modalities'] or '-'}")
        for modality in ("text", "audio", "vision"):
            interval = sample[f"missing_interval_{modality}"].tolist()
            ratio = sample[f"missing_ratio_{modality}"].item()
            valid_length = int(sample[f"valid_mask_{modality}"].sum().item())
            print(f"  {modality}: ratio={ratio:.4f}, interval={interval}, "
                  f"valid_length={valid_length}")


if __name__ == "__main__":
    main()
