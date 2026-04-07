"""Tests for the CLI entry points (console_scripts).

Subprocess tests are used for --help / missing-arg smoke tests.
In-process tests (via monkeypatch of sys.argv) are used for config-based
runs so that the coverage tool can track the ``__main__`` code paths.
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from typing import Any, cast

# Keep string-literal cast for Ruff TC006 in this repo.
pytest = cast("Any", importlib.import_module("pytest"))

# ===================================================================
# Subprocess smoke tests (--help, missing args)
# ===================================================================


class TestCLISmoke:
    """Subprocess smoke tests that the CLIs are installed and respond."""

    def test_exact_cli_help(self) -> None:
        """The exact CLI should respond to --help."""
        result = subprocess.run(
            [sys.executable, "-m", "mqt.ionshuttler.single_shuttler", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0
        assert "config_file" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_heuristic_cli_help(self) -> None:
        """The heuristic CLI should respond to --help."""
        result = subprocess.run(
            [sys.executable, "-m", "mqt.ionshuttler.multi_shuttler", "--help"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode == 0
        assert "config_file" in result.stdout.lower() or "usage" in result.stdout.lower()

    def test_exact_cli_missing_config_exits_nonzero(self) -> None:
        """Running the exact CLI without a config file should fail gracefully."""
        result = subprocess.run(
            [sys.executable, "-m", "mqt.ionshuttler.single_shuttler"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode != 0

    def test_heuristic_cli_missing_config_exits_nonzero(self) -> None:
        """Running the heuristic CLI without a config file should fail gracefully."""
        result = subprocess.run(
            [sys.executable, "-m", "mqt.ionshuttler.multi_shuttler"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        assert result.returncode != 0


# ===================================================================
# In-process CLI tests (counted by coverage)
# ===================================================================


class TestCLIInProcess:
    """In-process tests that invoke __main__.main() with monkeypatched sys.argv."""

    def test_exact_cli_with_config(
        self,
        exact_config_full_register: dict[str, object],
        tmp_path: Any,
        monkeypatch: Any,
    ) -> None:
        """Running the exact CLI main() with a valid config should succeed."""
        from mqt.ionshuttler.single_shuttler.__main__ import main

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(exact_config_full_register))

        monkeypatch.setattr("sys.argv", ["mqt-ionshuttler-exact", str(config_file)])
        main()  # Should not raise

    def test_heuristic_cli_with_config(
        self,
        heuristic_config_1pz: dict[str, object],
        tmp_path: Any,
        monkeypatch: Any,
    ) -> None:
        """Running the heuristic CLI main() with a valid config should succeed."""
        from mqt.ionshuttler.multi_shuttler.__main__ import main

        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(heuristic_config_1pz))

        monkeypatch.setattr("sys.argv", ["mqt-ionshuttler-heuristic", str(config_file)])
        main()  # Should not raise

    def test_exact_cli_missing_config_raises(self, monkeypatch: Any) -> None:
        """The exact CLI main() should raise SystemExit when no config is given."""
        from mqt.ionshuttler.single_shuttler.__main__ import main

        monkeypatch.setattr("sys.argv", ["mqt-ionshuttler-exact"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_heuristic_cli_missing_config_raises(self, monkeypatch: Any) -> None:
        """The heuristic CLI main() should raise SystemExit when no config is given."""
        from mqt.ionshuttler.multi_shuttler.__main__ import main

        monkeypatch.setattr("sys.argv", ["mqt-ionshuttler-heuristic"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0

    def test_heuristic_cli_nonexistent_config(self, tmp_path: Any, monkeypatch: Any) -> None:
        """The heuristic CLI should sys.exit(1) for a missing config file."""
        from mqt.ionshuttler.multi_shuttler.__main__ import main

        monkeypatch.setattr("sys.argv", ["mqt-ionshuttler-heuristic", str(tmp_path / "no_such_file.json")])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1

    def test_heuristic_cli_invalid_json(self, tmp_path: Any, monkeypatch: Any) -> None:
        """The heuristic CLI should sys.exit(1) for an invalid JSON file."""
        from mqt.ionshuttler.multi_shuttler.__main__ import main

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json!!!")
        monkeypatch.setattr("sys.argv", ["mqt-ionshuttler-heuristic", str(bad_file)])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1
