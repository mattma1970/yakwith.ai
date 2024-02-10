import time
import logging
from attrs import define, field, Factory


@define
class TimerContextManager:
    name: str = field(default="")
    logger: logging.Logger = field(default=None)
    level: int = field(default=logging.INFO)
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
        elif self.level == logging.INFO:
            self.logger.info(f"{self.name}: Time elapsed: {elapsed_time} (s)")
        else:
            pass
