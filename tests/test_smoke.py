def test_import():
    import deepforestvision
    assert deepforestvision.__version__

def test_cli_parser_builds():
    from deepforestvision.cli import build_parser
    p = build_parser()
    args = p.parse_args([])
    assert args.data_dir is not None