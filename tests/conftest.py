"""Shared fixtures: a small synthetic OULAD dataset generated once per test session."""

import pytest

from oulad_risk import synthetic


@pytest.fixture(scope="session")
def tables():
    return synthetic.generate(n_students=400, seed=1)


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    return synthetic.write(tmp_path_factory.mktemp("oulad"), n_students=400, seed=1)
