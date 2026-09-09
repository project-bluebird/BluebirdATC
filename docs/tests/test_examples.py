import pathlib

import pytest
from mktestdocs import check_md_file


def iter_markdown_files(root: str):
    root_path = pathlib.Path(root)
    for fpath in root_path.rglob("*.md"):
        # Skip virtualenvs, caches, and other hidden content nested under docs roots.
        if any(part.startswith(".") for part in fpath.relative_to(root_path).parts):
            continue
        yield fpath


def check_md_file_in_tmp(fpath, tmp_path, monkeypatch):
    source_path = fpath.resolve()
    # Run examples from a temporary working directory so snippets can create
    # files without leaving artefacts in the repository checkout.
    monkeypatch.chdir(tmp_path)
    check_md_file(fpath=source_path)


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("fpath", iter_markdown_files("docs/src"), ids=str)
def test_docs_examples(fpath, tmp_path, monkeypatch):
    check_md_file_in_tmp(fpath, tmp_path, monkeypatch)


@pytest.mark.parametrize("fpath", iter_markdown_files("bluebird-dt/docs"), ids=str)
def test_docs_examples_bluebird_dt(fpath, tmp_path, monkeypatch):
    check_md_file_in_tmp(fpath, tmp_path, monkeypatch)


@pytest.mark.parametrize("fpath", iter_markdown_files("bluebird-gymnasium/docs"), ids=str)
def test_docs_examples_bluebird_gym(fpath, tmp_path, monkeypatch):
    check_md_file_in_tmp(fpath, tmp_path, monkeypatch)
