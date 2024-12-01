from __future__ import annotations

import abc
import inspect
import itertools
import os
import pathlib
import re

__all__ = ["Dataset", "FileDataset"]

REG = "Regression"
TRANJECTORY_AD = "Trajectory anomaly detection"
MULTI_CLF = "Multi-class classification"
MO_BINARY_CLF = "Multi-output binary classification"
MO_REG = "Multi-output regression"


def get_data_home():
    """Return the location where remote datasets are to be stored."""

    data_home = os.environ.get("RIVER_DATA", os.path.join("~", "river_data"))
    data_home = os.path.expanduser(data_home)
    if not os.path.exists(data_home):
        os.makedirs(data_home)
    return data_home


class Dataset(abc.ABC):
    """Base class for all datasets.

    All datasets inherit from this class, be they stored in a file or generated on the fly.

    Parameters
    ----------
    n_features
        Number of features in the dataset.
    n_samples
        Number of samples in the dataset.
    n_classes
        Number of classes in the dataset, only applies to classification datasets.
    n_outputs
        Number of outputs the target is made of, only applies to multi-output datasets.
    anomaly_ratio
        The ratio of anomalies in the dataset.
    sparse
        Whether the dataset is sparse or not.

    """

    def __init__(
        self,
        n_features,
        n_samples=None,
        n_classes=None,
        n_outputs=None,
        anomaly_ratio=None,
        sparse=False,
    ):
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.n_classes = n_classes
        self.sparse = sparse
        self.anomaly_ratio = anomaly_ratio

    # @abc.abstractmethod
    # def __iter__(self):
    #     raise NotImplementedError

    # def take(self, k: int):
    #     """Iterate over the k samples."""
    #     return itertools.islice(self, k)

    @property
    def desc(self):
        """Return the description from the docstring."""
        desc = re.split(pattern=r"\w+\n\s{4}\-{3,}", string=self.__doc__, maxsplit=0)[0]
        return inspect.cleandoc(desc)

    @property
    def _repr_content(self):
        """The items that are displayed in the __repr__ method.

        This property can be overridden in order to modify the output of the __repr__ method.

        """

        content = {}
        content["Name"] = self.__class__.__name__
        if self.n_samples:
            content["Samples"] = f"{self.n_samples:,}"
        if self.n_features:
            content["Features"] = f"{self.n_features:,}"
        if self.n_outputs:
            content["Outputs"] = f"{self.n_outputs:,}"
        if self.n_classes:
            content["Classes"] = f"{self.n_classes:,}"
        if self.anomaly_ratio:
            content["Anomaly Ratio"] = f"{self.anomaly_ratio:.2f}"
        content["Sparse"] = str(self.sparse)

        return content

    def __repr__(self):
        l_len = max(map(len, self._repr_content.keys()))
        r_len = max(map(len, self._repr_content.values()))

        out = f"{self.desc}\n\n" + "\n".join(
            k.rjust(l_len) + "  " + v.ljust(r_len)
            for k, v in self._repr_content.items()
        )

        if "Parameters\n    ----------" in self.__doc__:
            params = re.split(
                r"\w+\n\s{4}\-{3,}",
                re.split("Parameters\n    ----------", self.__doc__)[1],
            )[0].rstrip()
            out += f"\n\nParameters\n----------{params}"

        return out


class FileDataset(Dataset):
    """Base class for datasets that are stored in a local file.

    Small datasets that are part of the river package inherit from this class.

    Parameters
    ----------
    filename
        The file's name.
    directory
        The directory where the file is contained. Defaults to the location of the `datasets`
        module.
    desc
        Extra dataset parameters to pass as keyword arguments.

    """

    def __init__(self, filename, directory=None, **desc):
        super().__init__(**desc)
        self.filename = filename
        self.directory = directory

    def load(self, **kwargs):
        return NotImplementedError

    @property
    def path(self):
        if self.directory:
            return pathlib.Path(self.directory).joinpath(self.filename)
        return pathlib.Path(__file__).parent.joinpath(self.filename)

    @property
    def _repr_content(self):
        content = super()._repr_content
        content["Path"] = str(self.path)
        return content
