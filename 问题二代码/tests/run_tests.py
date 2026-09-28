#!/usr/bin/env python3
"""Run the lightweight forward/backward tests without pytest."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_anchor_ablation import (  # noqa: E402
    test_all_anchor_modes_forward_backward_and_parameter_match,
    test_legacy_checkpoint_compatibility,
)
from tests.test_model import (  # noqa: E402
    test_all_losses_backward_finite,
    test_forward_shapes_probabilities_and_range,
    test_no_hidden_target_leakage,
    test_text_anchor_residual_gates,
)


def main() -> None:
    tests = (
        test_forward_shapes_probabilities_and_range,
        test_no_hidden_target_leakage,
        test_all_losses_backward_finite,
        test_text_anchor_residual_gates,
        test_all_anchor_modes_forward_backward_and_parameter_match,
        test_legacy_checkpoint_compatibility,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
