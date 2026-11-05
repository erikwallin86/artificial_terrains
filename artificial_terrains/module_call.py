"""Typed constructors for (module name, options) settings entries.

``constructor`` turns a module's ``__call__`` (or ``start``) into a function
with the same parameters (type checkers infer them via ParamSpec, so IDEs
complete them) that returns a ``ModuleCall``: a plain ``(name, options)``
tuple, the same format ``artificial_terrains.run`` already accepts.
"""
import inspect
from typing import Any, Callable, Concatenate, NamedTuple, ParamSpec

P = ParamSpec("P")


class ModuleCall(NamedTuple):
    name: str
    options: dict


def constructor(
        name: str, call: Callable[Concatenate[Any, P], Any]
) -> Callable[P, ModuleCall]:
    """Build a ModuleCall constructor for module ``name``."""
    signature = inspect.signature(call)
    self_name = next(iter(signature.parameters))

    def make(*args: P.args, **kwargs: P.kwargs) -> ModuleCall:
        bound = signature.bind_partial(None, *args, **kwargs)
        options = dict(bound.arguments)
        options.pop(self_name)
        # Flatten a catch-all **kwargs parameter into the options
        for param in signature.parameters.values():
            if param.kind is param.VAR_KEYWORD:
                options.update(options.pop(param.name, {}))
        return ModuleCall(name, options)

    make.__name__ = make.__qualname__ = name
    make.__doc__ = call.__doc__
    return make
