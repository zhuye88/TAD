import pytest

from sklearn.utils.estimator_checks import check_estimator

from IsoKernel import TemplateEstimator
from IsoKernel import TemplateClassifier
from IsoKernel import IsoKernel


@pytest.mark.parametrize(
    "estimator",
    [TemplateEstimator(), IsoKernel(), TemplateClassifier()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
