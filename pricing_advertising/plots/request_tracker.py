from __future__ import annotations

import csv
from io import StringIO

import mlflow
import numpy as np


class RequestTracker:
    """Accumulates per-day request metrics and logs them to mlflow, per training split.

    Two metrics are produced (keys suffixed with `split`, e.g. "train"/"val"):
      - "num_requests/by_request_day/{split}": one point per simulation day, averaged
        across all episodes seen so far.
      - "num_requests/by_reservation_day/{split}": one point per simulation day, averaged
        across all episodes seen so far.

    CSV artifacts are logged per episode for both metrics.
    """

    def __init__(self, split: str, log_to_mlflow: bool = True) -> None:
        self.split = split
        self.log_to_mlflow = log_to_mlflow

        self._episode_id: int = 0
        self._episode_num_requests: list[int] = []
        self._episode_t_reservation: list[float] = []

        self._day_num_requests_sum: np.ndarray | None = None
        self._day_num_requests_count: np.ndarray | None = None
        self._day_t_reservation_sum: np.ndarray | None = None
        self._day_t_reservation_count: np.ndarray | None = None

    def record_day(self, num_requests: int, t_reservation_mean: float) -> None:
        """Record request metrics for a single day.

        Args:
            num_requests: Number of requests presented on this day.
            t_reservation_mean: Mean t_reservation (as ordinal day offset) of requests.
        """
        self._episode_num_requests.append(int(num_requests))
        self._episode_t_reservation.append(float(t_reservation_mean))

    def end_episode(self) -> tuple[float | None, float | None]:
        """Flush the current episode: write CSV artifacts, accumulate per-day
        values into running averages, and return (mean_num_requests, mean_t_reservation)
        or (None, None) if no data was recorded.
        """
        if not self._episode_num_requests or not self._episode_t_reservation:
            return None, None

        episode_num_requests = np.asarray(self._episode_num_requests, dtype=int)
        episode_t_reservation = np.asarray(self._episode_t_reservation, dtype=float)
        mean_num_requests = float(episode_num_requests.mean())
        mean_t_reservation = float(episode_t_reservation.mean())

        if self.log_to_mlflow:
            # Log num_requests CSV
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "num_requests"])
            writer.writerows(enumerate(self._episode_num_requests))
            mlflow.log_text(
                buf.getvalue(),
                f"episodes/{self.split}/episode_{self._episode_id:05d}_num_requests.csv",
            )

            # Log t_reservation CSV
            buf = StringIO()
            writer = csv.writer(buf)
            writer.writerow(["day", "t_reservation_mean"])
            writer.writerows(enumerate(self._episode_t_reservation))
            mlflow.log_text(
                buf.getvalue(),
                f"episodes/{self.split}/episode_{self._episode_id:05d}_t_reservation.csv",
            )

        # Accumulate num_requests
        if self._day_num_requests_sum is None or self._day_num_requests_count is None:
            self._day_num_requests_sum = np.zeros_like(episode_num_requests, dtype=float)
            self._day_num_requests_count = np.zeros_like(episode_num_requests, dtype=int)
        if self._day_num_requests_sum.shape != episode_num_requests.shape:
            # Pad shorter arrays
            padding_len = len(self._day_num_requests_sum) - len(episode_num_requests)
            padding = np.zeros(padding_len, dtype=int)
            counts_padding = np.zeros(padding_len, dtype=int)
            episode_num_requests = np.concatenate([episode_num_requests, padding])
            counts_considering_padding = np.concatenate(
                [
                    np.ones_like(
                        episode_num_requests[: len(self._episode_num_requests)], dtype=int
                    ),
                    counts_padding,
                ]
            )
            self._day_num_requests_count += counts_considering_padding
        else:
            self._day_num_requests_count += 1
        self._day_num_requests_sum += episode_num_requests.astype(float)

        # Accumulate t_reservation
        if self._day_t_reservation_sum is None or self._day_t_reservation_count is None:
            self._day_t_reservation_sum = np.zeros_like(episode_t_reservation, dtype=float)
            self._day_t_reservation_count = np.zeros_like(episode_t_reservation, dtype=int)
        if self._day_t_reservation_sum.shape != episode_t_reservation.shape:
            # Pad shorter arrays
            padding_len = len(self._day_t_reservation_sum) - len(episode_t_reservation)
            padding = np.zeros(padding_len, dtype=float)
            counts_padding = np.zeros(padding_len, dtype=int)
            episode_t_reservation = np.concatenate([episode_t_reservation, padding])
            counts_considering_padding = np.concatenate(
                [
                    np.ones_like(
                        episode_t_reservation[: len(self._episode_t_reservation)], dtype=int
                    ),
                    counts_padding,
                ]
            )
            self._day_t_reservation_count += counts_considering_padding
        else:
            self._day_t_reservation_count += 1
        self._day_t_reservation_sum += episode_t_reservation

        self._episode_num_requests = []
        self._episode_t_reservation = []
        self._episode_id += 1

        return mean_num_requests, mean_t_reservation

    def finalize_training(self) -> None:
        """Finalize training: flush any remaining episode data and log
        per-day averages to mlflow.
        """
        if self._episode_num_requests:
            self.end_episode()

        if not self.log_to_mlflow:
            return

        # Log per-day num_requests
        if self._day_num_requests_sum is not None and self._day_num_requests_count is not None:
            for day in range(len(self._day_num_requests_sum)):
                mlflow.log_metric(
                    f"num_requests/by_request_day/{self.split}",
                    float(self._day_num_requests_sum[day] / self._day_num_requests_count[day]),
                    step=day,
                )

        # Log per-day t_reservation
        if self._day_t_reservation_sum is not None and self._day_t_reservation_count is not None:
            for day in range(len(self._day_t_reservation_sum)):
                mlflow.log_metric(
                    f"num_requests/by_reservation_day/{self.split}",
                    float(self._day_t_reservation_sum[day] / self._day_t_reservation_count[day]),
                    step=day,
                )
