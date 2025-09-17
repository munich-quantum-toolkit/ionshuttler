import argparse
import json
import pathlib

from .main import main as single_shuttler_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute exact shuttling schedules")
    parser.add_argument("config_file", help="path to json config file")
    parser.add_argument("--plot", action="store_true", help="plot grid")
    args = parser.parse_args()

    with pathlib.Path(args.config_file).open("r", encoding="utf-8") as f:
        config = json.load(f)

    single_shuttler_main(config, args.plot)


if __name__ == "__main__":
    main()
