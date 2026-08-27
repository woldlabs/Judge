"""CLI parser tests (no GUI)."""

from judge.__main__ import build_parser, main


def test_parser_analyze_and_demo():
    parser = build_parser()
    args = parser.parse_args(["analyze", "foo.mp4", "--sensitivity", "0.4", "--format", "json"])
    assert args.command == "analyze"
    assert args.paths == ["foo.mp4"]
    assert args.sensitivity == 0.4
    assert args.format == "json"

    demo = parser.parse_args(["demo", "--duration", "12"])
    assert demo.command == "demo"
    assert demo.duration == 12


def test_version_command(capsys):
    rc = main(["version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("judge 0.")
