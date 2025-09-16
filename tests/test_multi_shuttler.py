import json
from pathlib import Path

from mqt.ionshuttler.multi_shuttler.main import main


def test_main() -> None:
    config_file = Path(__file__).absolute().parent.parent / "inputs/algorithms_heuristic/qft_06_1pz.json"
    with Path(config_file).open("r", encoding="utf-8") as f:
        config = json.load(f)

    main(config)
