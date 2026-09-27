from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.train_pretrained_fusion import build_dialogue_contexts


def main() -> None:
    arrays = SimpleNamespace(
        ids=np.asarray(["videoA$_$0", "videoA$_$1", "videoA$_$2", "videoB$_$0"]),
        raw_text=np.asarray(["first", "middle", "last", "isolated"]),
        size=4,
    )
    previous = build_dialogue_contexts(arrays, 1, "previous")
    bidirectional = build_dialogue_contexts(arrays, 1, "bidirectional")
    assert previous == ["", "first", "middle", ""]
    assert bidirectional[0] == "Following utterances: middle"
    assert bidirectional[1] == (
        "Previous utterances: first\nFollowing utterances: last"
    )
    assert bidirectional[2] == "Previous utterances: middle"
    assert bidirectional[3] == ""
    print("Bidirectional split-local context smoke test passed.")


if __name__ == "__main__":
    main()

