import sys
import inspect
import modules.data
import modules.blender
import modules.terrain
import modules.modifiers
import modules.generating
import modules.generating_function
import modules.combining
import modules.other
import modules.pipes
import modules.loop
import modules.las

MODULES = {}


# ## Add from all module files ###
# data
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.data.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# modifiers
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.modifiers.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# generating
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.generating.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# generating_function
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.generating_function.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# combining
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.combining.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# other
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.other.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# (multi) pipes
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.pipes.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# loop over pipes
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.loop.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# las
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.las.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# blender
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.blender.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v

# terrain
clsmembers_pairs = inspect.getmembers(
    sys.modules[modules.terrain.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    MODULES[k] = v
