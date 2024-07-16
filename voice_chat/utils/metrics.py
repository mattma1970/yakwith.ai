import numpy as np
from attrs import field, define
from typing import Dict, Union, Tuple, Callable
from enum import Enum
import os
import random
import string
import json
from .text_processing import safe_key
from .file import createIfMissing
import logging

import configs.AppConfig as AppConfig

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    INFO = 0
    DEBUG = 1


class MetricType(Enum):
    POINT = 0
    INTERVAL = 1


@define
class Metric:
    """
    Metric data class that records tuples of data, tuple[str, Union[float,datetime]] and calculate statistics
    Using Tuple as the base data element allows the value to be tagged and the tag used for analysis.
    Metrics should be managed through an instance of the MetricLogger class and not directly on the metric.
    """

    name: str = field(default="")
    level: LogLevel = field(default=LogLevel.DEBUG)
    enabled: bool = field(default=True)
    data: list[Tuple] = field(factory=list)
    type: MetricType = field(default=MetricType.POINT)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def log(self, value: Union[float, Tuple]):
        if not self.enabled:
            return
        if isinstance(value, tuple):
            self.data.append(value)
        elif isinstance(value, float):
            self.data.append(("", value))

    def calculate_iqr(self, test_data) -> float:
        """
        Calculate the interquartile range (IQR) of given data.
        @ args:
            test_data: data set to calculate IQR on. This could be the raw data or data derived from it.

        """
        q1 = np.percentile(test_data, 25)
        q3 = np.percentile(test_data, 75)
        return q3 - q1

    def calculate_mean(self, test_data: np.array) -> float:
        return np.mean(test_data)

    def calcualate_median(self, test_data: np.array) -> float:
        return np.median(test_data)

    def get_summary_stats(self, filtered_data):
        mean: float = self.calculate_mean(filtered_data)
        median: float = self.calcualte_median(filtered_data)
        iqr: float = self.calculate_iqr(filtered_data)
        return {"n": len(filtered_data), "mean": mean, "median": median, "iqr": iqr}

    def filter_data(self, filter: Callable) -> np.array:
        filtered_data: np.array = None
        if filter:
            filtered_data = np.array(
                [datum[1] for datum in self.data if filter(datum[0])]
            )
        else:
            filtered_data = np.array([datum[1] for datum in self.data])
        return filtered_data

    def get_point_stats(self, filter: Callable = None) -> Dict:
        """
        Get summary stats : mean, median, IQR for point data.
        @args:
            filter: a callable function applied to the first element in a
            data tuple which, if true, then the element is included.
        @return:
            Dict of stats {n,mean, median, iqr}
        """
        filtered_data: np.array = self.filter_data(filter)

        return self.get_summary_stats(filtered_data)

    def get_interval_stats(self, filter: Callable = None) -> Dict:
        """
        Get statistics on the difference between values
        """
        filtered_data: np.array = None
        if filter:
            filtered_data = np.array(
                [datum[1] for datum in self.data if filter(datum[0])]
            )
        else:
            filtered_data = np.array([datum[1] for datum in self.data])

        interval_data: list[float] = []
        if len(filtered_data) >= 2:
            for i in range(1, len(self.data)):
                interval_data[i - 1] = self.data[1] - self.data[i - 1]

        return self.get_summary_stats(interval_data)


@define
class MetricLogManager:
    """
    Management object for all metric loggers.
    """

    metrics: Dict[str, Metric] = field(factory=dict)
    level: LogLevel = field(default=LogLevel.DEBUG)
    prefix: str = field(default="")  # For creating unique filenames
    path: os.PathLike = field(default="")
    auto_dump: bool = field(
        default=False
    )  # Automatically write the data out when the length of the metric data increase by auto_dump_length
    auto_dump_length: int = field(default=5)
    last_dump_index: Dict = field(factory=dict)

    def __attrs_post_init__(self):
        self.prefix = self.random_prefix(5) + "_"

    def create_metric(
        self, name: str, level: LogLevel, type: MetricType = MetricType.POINT
    ):
        if safe_key(name) in self.metrics:
            logger.error(f"Metric Name {name} already in use.")
            return
        metric: Metric = Metric(name=name, level=level, type=type)
        self.metrics[safe_key(name)] = metric
        self.last_dump_index[safe_key(name)] = 0

    def random_prefix(self, size: int = 5):
        """
        Generate 5 random letters.
        """
        return "".join(random.choices(string.ascii_letters, k=size))

    def debug(self, metric_name: str, value: Union[float, Tuple]):
        metric_name = safe_key(metric_name)
        if metric_name not in self.metrics:
            self.create_metric(metric_name, LogLevel.DEBUG)

        if (len(self.metrics[metric_name]) % self.auto_dump_length) == 0:
            self.dump_all_data("append")

        if (
            self.metrics[metric_name].level == LogLevel.DEBUG
            and self.metrics[metric_name].enabled
        ):
            self.metrics[metric_name].log(value)
        else:
            return

    def info(self, metric_name: str, value: Union[float, Tuple]):
        metric_name = safe_key(metric_name)
        if metric_name not in self.metrics:
            self.create_metric(metric_name, LogLevel.INFO)
        if (
            self.metrics[metric_name].level == LogLevel.INFO
            and self.metrics[metric_name].enabled
        ):
            self.metrics[metric_name].log(value)
        else:
            return

    def dump_data(self, metric_name: str, mode: str = "append"):
        """
        @args:
            path: osLike: folder where logs are to be saved.
        """
        attempts: int = 0
        ok: bool = False
        while attempts < 5:
            fqn: str = os.path.join(
                self.path, "".join([self.prefix, safe_key(metric_name), ".log"])
            )
            exists: bool = createIfMissing(fqn)
            if mode == "write":
                attempts += 1
                if not exists:
                    ok = True
                    break
            else:
                ok = True
                break
        if ok:
            with open(fqn, "a") as f:
                start_index = self.last_dump_index[safe_key(metric_name)]
                output = [
                    ",".join(list(map(lambda x: str(x), datum)))
                    for datum in self.metrics[safe_key(metric_name)][start_index:]
                ]
                if len(output) > 0:
                    data = "\n".join(output)
                    f.write(data + "\n")
            # Update the pointer to the last element exported
            self.last_dump_index[safe_key(metric_name)] = len(
                self.metrics[safe_key(metric_name)]
            )
        else:
            logger.warning(
                "Failed to save metric log {metric_name} after multiple attempts at finding avaiable name."
            )

    def dump_all_data(self, mode: str = "append"):
        """
        Dump all log raw data to files
        """
        for name in self.metrics:
            self.dump_data(safe_key(name), mode)

    def dump_stats(self, title: str, stats: Union[str, Dict]):
        fqn: str = os.path.join(self.path, "".join([self.prefix, "stats.log"]))
        createIfMissing(fqn)
        with open(fqn, "a") as f:
            f.write(title)
            if isinstance(stats, Dict):
                f.write("\n".join(json.dumps(stats)))
            else:
                f.write("\n".join(stats))

    def calculate_metric_point_stats(
        self, metric_name: str, filter: Callable = None, save: bool = False
    ):
        """
        Caculate the statistics of the points within the data.
        """
        metric = safe_key(metric_name)
        stats = self.metrics[metric].get_point_stats(filter)
        if metric in self.metrics:
            if save:
                self.dump_stats(metric_name, stats)
            else:
                logger.debug(json.dumps(stats))
        else:
            return None

    def calculate_metric_intervals(
        self, metric_name: str, filter: Callable = None, save: bool = False
    ):
        """
        Calculate the statistics of the intervals formed by subsequent datapoints.
        """
        metric = safe_key(metric_name)
        if metric in self.metrics:
            stats = self.metrics[metric].get_interval_stats(filter)
            if save:
                self.dump_stats(metric_name, stats)
            else:
                logger.debug(json.dumps(stats))
        else:
            return None

    def download_and_clear(self, reset: bool = False):
        """
        Export all raw data, create summary states for all and clear all logs.
        """
        self.dump_all_data(mode="append")
        for metric_key in self.metrics:
            self.calculate_metric_point_stats(metric_key, None, True)
        if reset:
            self.metrics = {}
            self.last_dump_index = {}
            logger.debug("Metric logs cleared")
        return None


metric_logger = MetricLogManager(
    path=AppConfig.Configurations.logging.metric_logs_folder
)  # TODO make this a singleton object.
