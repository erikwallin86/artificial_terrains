import inspect

from . import terrain
from . import modifiers
from . import generating
from . import generating_function
from . import combining
from . import other
from . import loop
from . import general
from . import blender

MODULES: dict[str, type] = {}

module_files = [
    general, blender, terrain, modifiers,
    generating, generating_function, combining,
    other, loop
]
for mod in module_files:
    for name, cls in inspect.getmembers(mod, inspect.isclass):
        # Optional: only include classes actually defined in that module file
        if cls.__module__ != mod.__name__:
            continue
        MODULES[name] = cls
