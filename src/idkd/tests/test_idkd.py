# Copyright 2023 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import time
import pytest
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
    assert_array_equal,
    ignore_warnings,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(sys.path)

from idkd._idkd import IDKD

rng = check_random_state(0)


@pytest.fixture
def data():
    return load_wine(return_X_y=True)


def test_idkd():
    """Check Isolation NNE for various parameter settings."""
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid(
        {
            "n_estimators": [100, 200],
            "max_samples": [10, 20, 30],
            "algorithm": ["inne", "anne"],
        }
    )

    with ignore_warnings():
        for params in grid:
            IDKD(random_state=0, **params).fit(X_train).predict(X_test)


def test_idkd_performance():
    """Test Isolation NNE performs well"""

    # Generate train/test data
    rng = check_random_state(2)
    X = 0.3 * rng.randn(120, 2)
    X_train = np.r_[X + 2, X - 2]
    X_train = X[:100]

    # Generate some abnormal novel observations
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    X_test = np.r_[X[100:], X_outliers]
    y_test = np.array([0] * 20 + [1] * 20)

    # fit the model
    clf = IDKD(n_estimators=200, max_samples=16, algorithm="anne").fit(X_train)

    # predict scores (the lower, the more normal)
    y_pred = -clf.decision_function(X_test)

    # check that there is at most 6 errors (false positive or false negative)
    assert roc_auc_score(y_test, y_pred) > 0.98


@pytest.mark.parametrize("contamination", [0.25, "auto"])
def test_idkd_works(contamination):
    # toy sample (the last two samples are outliers)
    X = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [6, 3], [-4, 7]]

    # Test IsolationForest
    clf = IDKD(random_state=0, contamination=contamination)
    clf.fit(X)
    decision_func = -clf.decision_function(X)
    pred = clf.predict(X)
    # assert detect outliers:
    assert np.min(decision_func[-2:]) >= np.max(decision_func[:-2])
    assert_array_equal(pred, 6 * [1] + 2 * [-1])


def test_score_samples():
    X_train = [[1, 1], [1, 2], [2, 1]]
    clf1 = IDKD(contamination=0.1, algorithm="inne")
    clf1.fit(X_train)
    clf2 = IDKD(contamination=0.1, algorithm="anne")
    clf2.fit(X_train)
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]),
        clf1.decision_function([[2.0, 2.0]]) + clf1.offset_,
    )
    assert_array_equal(
        clf2.score_samples([[2.0, 2.0]]),
        clf2.decision_function([[2.0, 2.0]]) + clf2.offset_,
    )
    assert_array_equal(
        clf1.score_samples([[2.0, 2.0]]), clf2.score_samples([[2.0, 2.0]])
    )


def test_fit_time(data):
    X, y = data
    clf = IDKD(n_estimators=200, max_samples=100, algorithm="inne")
    t1 = time.time()
    clf.fit(data)
    t2 = time.time()
    anomaly_labels = clf.predict(data)
    t3 = time.time()
    print(t2 - t1)
    print(t3 - t2)

    clf2 = IDKD(n_estimators=200, max_samples=100, algorithm="anne")
    t1 = time.time()
    clf2.fit(data)
    t2 = time.time()
    anomaly_labels = clf2.predict(data)
    t3 = time.time()
    print(t2 - t1)
    print(t3 - t2)
