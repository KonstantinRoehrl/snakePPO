import pytest

from snakeppo.cli import main


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])
    assert exit_info.value.code == 0
    assert "snakeppo" in capsys.readouterr().out
