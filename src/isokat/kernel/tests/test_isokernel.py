"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

from sklearn.datasets import load_iris
from isoml import IsoKernel
import pytest

method = ["inne", "anne"]


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize("method", method)
def test_IsoKernel_fit(data, method):
    X = data[0]
    ik = IsoKernel(method=method, n_estimators=200)
    ik.fit(X)
    assert ik.is_fitted_


@pytest.mark.parametrize("method", method)
def test_IsoKernel_similarity(data, method):
    X = data[0]
    ik = IsoKernel(method=method, n_estimators=200)
    ik.fit(X)
    similarity = ik.similarity(X)
    assert similarity.shape == (X.shape[0], X.shape[0])


@pytest.mark.parametrize("method", method)
def test_IsoKernel_transform(data, method):
    X = data[0]
    max_samples = 16
    ik = IsoKernel(method=method, max_samples=max_samples)
    ik.fit(X)
    transformed_X = ik.transform(X)
    assert transformed_X.shape == (X.shape[0], ik.n_estimators * max_samples)
