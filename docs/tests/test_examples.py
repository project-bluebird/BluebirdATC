import pathlib
import pytest

from mktestdocs import check_md_file


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_docs_examples(fpath):
    check_md_file(fpath=fpath)


@pytest.mark.parametrize(
    "fpath", pathlib.Path("bluebird-dt/docs").glob("**/*.md"), ids=str
)
def test_docs_examples_bluebird_dt(fpath):
    check_md_file(fpath=fpath)


@pytest.mark.parametrize(
    "fpath", pathlib.Path("bluebird-gymnasium/docs").glob("**/*.md"), ids=str
)
def test_docs_examples_bluebird_gym(fpath):
    check_md_file(fpath=fpath)
