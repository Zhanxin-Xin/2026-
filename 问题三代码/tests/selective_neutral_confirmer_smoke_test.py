from __future__ import annotations

import numpy as np

from src.train_selective_neutral_confirmer import (
    ConfirmationRule,
    apply_rule,
    minimal_neutral_projection,
    select_rule,
)


def main() -> None:
    parent = np.asarray(
        [
            [0.10, 0.44, 0.46],
            [0.50, 0.45, 0.05],
            [0.05, 0.90, 0.05],
            [0.05, 0.10, 0.85],
        ],
        dtype=np.float64,
    )
    expert = np.asarray([0.9, 0.9, 0.9, 0.1])
    ontology = np.asarray([0.8, 0.8, 0.8, 0.1])
    polar = np.asarray([0.05, 0.05, 0.05, 0.9])
    rule = ConfirmationRule(True, 0.08, 0.8, 0.5, 0.1)
    fused, changed = apply_rule(parent, expert, ontology, polar, rule)
    assert changed.tolist() == [True, True, False, False]
    assert fused.argmax(axis=1).tolist() == [1, 1, 1, 2]
    assert np.allclose(fused.sum(axis=1), 1.0)
    before_odds = parent[changed, 2] / parent[changed, 0]
    after_odds = fused[changed, 2] / fused[changed, 0]
    assert np.allclose(before_odds, after_odds)

    identity = minimal_neutral_projection(parent, np.zeros(len(parent), dtype=bool))
    assert np.allclose(identity, parent)
    selected, metrics, _ = select_rule(
        parent,
        np.zeros(len(parent)),
        np.zeros(len(parent)),
        np.ones(len(parent)),
        parent.argmax(axis=1),
        [ConfirmationRule(False), rule],
    )
    assert not selected.enabled
    assert metrics["changed_decisions"] == 0
    print("selective neutral confirmer smoke test passed")


if __name__ == "__main__":
    main()
