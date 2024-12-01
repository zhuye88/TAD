"""
Copyright 2024 Xin Han. All rights reserved.
Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file.
"""

from sklearn.datasets import load_iris
from isoml.kernel import IsodisKernel
import pytest

method = ["inne", "anne"]


@pytest.fixture
def data():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize("method", method)
def test_IsoDisKernel_fit(data, method):
    X = data[0]
    idk = IsodisKernel(method=method, n_estimators=200)
    idk.fit(X)
    assert idk.is_fitted_


@pytest.mark.parametrize("method", method)
def test_IsoDisKernel_similarity(data, method):
    X = data[0]
    idk = IsodisKernel(method=method, n_estimators=200)
    idk.fit(X)
    D_i = X[:10]
    D_j = X[-10:]
    similarity = idk.similarity(D_i, D_j)
    assert similarity == 0.0


@pytest.mark.parametrize("method", method)
def test_IsoDisKernel_transform(data, method):
    X = data[0]
    max_samples = 16
    idk = IsodisKernel(method=method, n_estimators=200, max_samples=max_samples)
    idk.fit(X)
    D_i = X[:10]
    D_j = X[-10:]
    transformed_D_i, transformed_D_j = idk.transform(D_i, D_j)
    assert transformed_D_i.shape == (10, idk.n_estimators * max_samples)
    assert transformed_D_j.shape == (10, idk.n_estimators * max_samples)
