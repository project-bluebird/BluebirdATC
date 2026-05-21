import pathlib

import pytest
from mktestdocs import check_md_file


def check_md_file_in_tmp(fpath, tmp_path, monkeypatch):
    source_path = fpath.resolve()
    # Run examples from a temporary working directory so snippets can create
    # files without leaving artefacts in the repository checkout.
    monkeypatch.chdir(tmp_path)
    check_md_file(fpath=source_path)


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_docs_examples(fpath, tmp_path, monkeypatch):
    check_md_file_in_tmp(fpath, tmp_path, monkeypatch)


@pytest.mark.parametrize("fpath", pathlib.Path("bluebird-dt/docs").glob("**/*.md"), ids=str)
def test_docs_examples_bluebird_dt(fpath, tmp_path, monkeypatch):
    check_md_file_in_tmp(fpath, tmp_path, monkeypatch)


@pytest.mark.parametrize("fpath", pathlib.Path("bluebird-gymnasium/docs").glob("**/*.md"), ids=str)
def test_docs_examples_bluebird_gym(fpath, tmp_path, monkeypatch):
    check_md_file_in_tmp(fpath, tmp_path, monkeypatch)
