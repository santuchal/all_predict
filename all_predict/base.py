# base.py
"""Base classes and utilities for all_predict.
Handles configuration, logging, timing, and parallelization support.
"""

import logging
import time
from dataclasses import dataclass, field

@dataclass
class BasePredictor:
    verbose: bool = True
    random_state: int = 42
    n_jobs: int = -1
    save_models: bool = False

    def __post_init__(self):
        self._configure_logging()

    def _configure_logging(self):
        logging_level = logging.DEBUG if self.verbose else logging.WARNING
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(message)s",
            level=logging_level
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        if self.verbose:
            self.logger.info("Initialized with verbose logging enabled.")

    def _time_function(self, func, *args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        self.logger.debug(f"Function {func.__name__} executed in {elapsed:.4f} seconds")
        return result, elapsed

    def _set_random_state(self, estimator):
        """Set random_state for models that support it."""
        try:
            valid_params = estimator.get_params(deep=True)
        except Exception:
            return estimator

        if 'random_state' in valid_params:
            estimator.set_params(random_state=self.random_state)
        return estimator


    # def _set_random_state(self, estimator):
    #     valid_params = estimator.get_params()
    #     if 'random_state' in valid_params:
    #         estimator.set_params(random_state=self.random_state)
    #     return estimator

    # def _set_random_state(self, estimator):
    #     if hasattr(estimator, 'random_state'):
    #         estimator.set_params(random_state=self.random_state)
    #     return estimator
