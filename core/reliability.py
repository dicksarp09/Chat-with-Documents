import time
import logging
import functools
from typing import Callable, Any, TypeVar, Optional, Generic
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

T = TypeVar("T")

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    exponential_base: float = 2.0

    def get_delay(self, attempt: int) -> float:
        if self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay_seconds * (self.exponential_base**attempt)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay_seconds * attempt
        else:
            delay = self.base_delay_seconds
        return min(delay, self.max_delay_seconds)


RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


def with_retry(config: RetryConfig = None):
    config = config or RetryConfig()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as e:
                    last_exception = e
                    if attempt < config.max_retries:
                        delay = config.get_delay(attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{config.max_retries} for {func.__name__} "
                            f"after {delay:.1f}s: {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"All retries exhausted for {func.__name__}: {e}")
                except Exception as e:
                    logger.error(f"Non-retryable error in {func.__name__}: {e}")
                    raise

            raise last_exception

        return wrapper

    return decorator


def with_timeout(seconds: float):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(
                    f"Function {func.__name__} timed out after {seconds}s"
                )

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result

        return wrapper

    return decorator


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    pass


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        half_open_max_calls: int = 3,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitState.CLOSED

        self._lock = None

    def call(self, func: Callable, *args, **kwargs) -> Any:
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
            else:
                logger.warning(f"Circuit breaker {self.name} is OPEN")
                raise CircuitBreakerOpenError(f"Circuit {self.name} is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time > self.recovery_timeout

    def _on_success(self):
        self.failure_count = 0
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} CLOSED")

    def _on_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(
                f"Circuit breaker {self.name} OPEN after {self.failure_count} failures"
            )

    def get_state(self) -> dict:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
        }


class FallbackStrategy(Generic[T]):
    def __init__(
        self,
        primary: Callable[..., T],
        fallback_value: Optional[T] = None,
        fallback_fn: Optional[Callable[..., T]] = None,
        use_retry: bool = True,
    ):
        self.primary = primary
        self.fallback_value = fallback_value
        self.fallback_fn = fallback_fn
        self.use_retry = use_retry

    def execute(self, *args, **kwargs) -> T:
        try:
            if self.use_retry:
                return with_retry()(self.primary)(*args, **kwargs)
            return self.primary(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Primary failed, using fallback: {e}")
            if self.fallback_fn:
                return self.fallback_fn(*args, **kwargs)
            return self.fallback_value


class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: int = 60):
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self.calls: list[float] = []

    def is_allowed(self) -> bool:
        now = time.time()
        self.calls = [t for t in self.calls if now - t < self.window_seconds]

        if len(self.calls) >= self.max_calls:
            return False

        self.calls.append(now)
        return True

    def wait_time(self) -> float:
        if not self.calls:
            return 0
        now = time.time()
        oldest_in_window = now - self.window_seconds
        recent_calls = [t for t in self.calls if t > oldest_in_window]
        if len(recent_calls) < self.max_calls:
            return 0
        return self.window_seconds - (now - min(recent_calls))


class ErrorAccumulator:
    def __init__(self, window_seconds: int = 300):
        self.window_seconds = window_seconds
        self.errors: dict = defaultdict(list)

    def record(self, error_type: str):
        now = time.time()
        self.errors[error_type].append(now)
        self._cleanup(now)

    def get_rate(self, error_type: str) -> float:
        self._cleanup(time.time())
        return len(self.errors.get(error_type, []))

    def _cleanup(self, now: float):
        cutoff = now - self.window_seconds
        for key in list(self.errors.keys()):
            self.errors[key] = [t for t in self.errors[key] if t > cutoff]
            if not self.errors[key]:
                del self.errors[key]


def retry_on_failure(func: Callable = None, max_retries: int = 3, delay: float = 1.0):
    if func is None:
        return functools.partial(retry_on_failure, max_retries=max_retries, delay=delay)
    return with_retry(RetryConfig(max_retries=max_retries, base_delay_seconds=delay))(
        func
    )


def fallback_on_error(primary: Any, fallback: Any = None):
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Error in {func.__name__}: {e}, using fallback")
                if callable(fallback):
                    return fallback(*args, **kwargs)
                return fallback

        return wrapper

    if callable(primary):
        return decorator(primary)
    return decorator
