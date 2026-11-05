import os
import time
import functools
import inspect
from typing import Callable, TypeVar
from ..utils.debug import debug

F = TypeVar("F", bound=Callable)


def debug_decorator(func: F) -> F:
    '''
    Wrapper around calls to print some debug info
    and measure execution time.

    If func has no **kwargs catch-all, it only receives the pipe keys it
    declares, and unknown explicit module options raise a TypeError.
    '''
    params = inspect.signature(func).parameters
    accepts_any = any(p.kind is p.VAR_KEYWORD for p in params.values())

    @functools.wraps(func)
    def wrapper(self, call_number=None, call_total=None, *args, **kwargs):
        if call_number is not None and call_total is not None:
            self.logger.info(f"Run {self.name} {call_number+1}/{call_total}")

        if call_total is not None and call_total > 1:
            self.file_id = f'_{call_number:05d}'
        else:
            self.file_id = ''

        # Possibly debug input
        debug(self, kwargs, call_number, call_total, 'input')

        # ---- timing starts here ----
        start = time.perf_counter()

        kwargs = {'call_number': call_number, 'call_total': call_total,
                  **kwargs}
        if not accepts_any:
            unknown = set(getattr(self, 'module_options', {})) - set(params)
            if unknown:
                raise TypeError(
                    f"{self.name} got unknown option(s) {sorted(unknown)}, "
                    f"valid: {[p for p in params if p != 'self']}")
            kwargs = {k: v for k, v in kwargs.items() if k in params}

        result = func(self, *args, **kwargs)

        elapsed = time.perf_counter() - start
        # ---- timing ends here ----

        # Log timing
        self.logger.info(f"  took {elapsed:.3f} s")

        # Possibly debug output
        debug(self, result, call_number, call_total, 'output')

        return result

    return wrapper  # type: ignore[return-value]


class Module():
    create_folder = True
    '''
    Args:
    '''
    def __init__(self, save_dir=None, logger=None):
        self.save_dir = save_dir
        self.logger = logger

        self.save_dir_original = save_dir

        self.save_dir = os.path.join(self.save_dir, self.__class__.__name__)
        if self.create_folder:
            if not os.path.isdir(self.save_dir):
                os.makedirs(self.save_dir)

    def __call__(
            self, **kwargs):
        # Return kwargs which are not 'input'/'data'
        return kwargs

    def info(self, logging_string):
        ''' 'Shorter' method for logger.info, with indent. '''
        self.logger.info(f"  {logging_string}")

    def debug(self, logging_string):
        ''' 'Shorter' method for logger.debug, with indent. '''
        self.logger.debug(f"  {logging_string}")

    def start(self, **kwargs):
        # Store kwargs
        self.kwargs = kwargs
        # Initialize loop generator
        self.loop_generator_instance = self.loop_generator()

    @property
    def name(self):
        return self.__class__.__name__

    def loop_generator(self):
        """
        Default-beteende: returnera bara en gång, samma som __call_
        """
        yield self.__call__(**self.kwargs)

    def __iter__(self):
        """
        Gör modulen itererbar
        """
        return self.loop_generator_instance
