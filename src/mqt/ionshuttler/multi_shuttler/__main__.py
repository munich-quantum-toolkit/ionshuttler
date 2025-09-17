import argparse
import json
import pathlib
import sys

from .main import main as multi_shuttler_main


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute heuristic shuttling schedules")
    parser.add_argument("config_file", help="Path to the JSON configuration file")
    args = parser.parse_args()

    try:
        with pathlib.Path(args.config_file).open("r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_file}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not parse JSON file {args.config_file}")
        sys.exit(1)

    multi_shuttler_main(config)


if __name__ == "__main__":
    main()
