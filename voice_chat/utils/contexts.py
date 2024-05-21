import time
import logging
from attrs import define, field

import voice_chat.utils.metrics as GlobalMetricLogger

metric_logger = GlobalMetricLogger.metric_logger


@define
class TimerContextManager:
    """
    Context manager for timing an operation.
    If metric_name is passed in then the time is logged in the global metric with that name
    @args:
        name: Label used for displaying the metrics on the metrics on the cli
        metric_name: name of the metric in the metric dictionary. If present, the timer value will be logged
        datum_label: str: a label for the individual datapoint within the metric log.
    """

    name: str = field(default="")
    logger: logging.Logger = field(default=None)
    level: int = field(default=logging.INFO)
    metric_name: str = field(default="")  # name of the metric
    datum_label: str = field(
        default=""
    )  # string value to tag the datum with. e.g.the phrase or phrase length.
    start_time: float = field(default=0.0)
    end_time: float = field(default=0.0)

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        if self.level == logging.DEBUG:
            self.logger.debug(f"{self.name}: Time elapsed: {elapsed_time} (s)")
            if self.metric_name.strip() != "" and metric_logger:
                if self.datum_label != "":
                    datum = (self.datum_label, elapsed_time)
                else:
                    datum = elapsed_time
                metric_logger.debug(self.metric_name, datum)
        elif self.level == logging.INFO:
            self.logger.info(f"{self.name}: Time elapsed: {elapsed_time} (s)")
            if self.metric_name.strip() != "" and metric_logger:
                if self.datum_label != "":
                    datum = (self.datum_label, elapsed_time)
                else:
                    datum = elapsed_time
                metric_logger.info(self.metric_name, datum)
        else:
            pass
