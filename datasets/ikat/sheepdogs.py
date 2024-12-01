from .base import FileDataset
import pandas as pd
import numpy as np
import datetime


class SheepDogs(FileDataset):
    """SheepDogs dataset.

    This dataset is a trajectory dataset. It is collected from MoveBank, where each dataset records trajectories of a kind of animal over a period.

    References
    ----------
    [^1]: [Movebank](https://www.movebank.org/cms/movebank-main)
    [^2]: Wang, Y., Wang, Z., Ting, K. M., & Shang, Y. (2024).
    A Principled Distributional Approach to Trajectory Similarity Measurement and
    its Application to Anomaly Detection. Journal of Artificial Intelligence Research, 79, 865-893.
    """

    def __init__(self):
        super().__init__(
            n_features=2,
            n_samples=None,
            n_classes=2,
            filename="sheepdogs.zip",
        )
        self.anomaly_ratio = None

    def load(self, return_timestamp_X_y=False):
        data = pd.read_csv(
            self.path,
            usecols=[
                "timestamp",
                "location-long",
                "location-lat",
                "individual-local-identifier",
            ],
            parse_dates=["timestamp"],
        )

        individual_local = data["individual-local-identifier"].value_counts().index
        selected_individuals = [0, 1, 2, 6, 14]
        mask = data["individual-local-identifier"].isin(
            individual_local[selected_individuals]
        )
        sub_data = data[mask]
        sub_data.dropna(
            subset=["location-long", "location-lat"], how="any", inplace=True
        )

        gap = datetime.timedelta(hours=1)
        # current_id = sub_data["individual-local-identifier"].iloc[0]
        tmp_traj = []
        anomaliers = []
        normal_traj = []
        timestamp_lst = []

        # Process trajectories
        for i in range(1, len(sub_data)):
            previous_location_long, previous_location_lag = (
                sub_data["location-long"].iloc[i - 1],
                sub_data["location-lat"].iloc[i - 1],
            )

            previous_timestamp = sub_data["timestamp"].iloc[i - 1]
            current_timestamp = sub_data["timestamp"].iloc[i]
            previous_individual = sub_data["individual-local-identifier"].iloc[i - 1]
            current_individual = sub_data["individual-local-identifier"].iloc[i]
            tmp_traj.append(
                (previous_timestamp, previous_location_long, previous_location_lag)
            )

            if (
                current_timestamp - previous_timestamp > gap
                or previous_individual != current_individual
            ):
                if len(tmp_traj) > 10:
                    if previous_individual == individual_local[14]:
                        anomaliers.append(tmp_traj)
                    else:
                        normal_traj.append(tmp_traj)
                tmp_traj = []

        subset_anomaliers = [x for x in anomaliers if len(x) > 16]
        all_traj = normal_traj + subset_anomaliers
        labels = np.array([0] * len(normal_traj) + [1] * len(subset_anomaliers))
        self.anomaly_ratio = len(subset_anomaliers) / len(all_traj)
        self.n_samples = len(all_traj)
        if return_timestamp_X_y:
            all_traj = [np.array(x)[:, 1:] for x in all_traj]
            timestamp_lst = [np.array(x)[:, 0] for x in all_traj]
            return timestamp_lst, all_traj, labels
        else:
            return {
                "X": all_traj,
                "y": labels,
            }
