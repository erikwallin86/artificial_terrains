import sys
import inspect
import datahandlers.data
import datahandlers.blender
import datahandlers.terrain
import datahandlers.modifiers
import datahandlers.generating
import datahandlers.generating_function
import datahandlers.combining
import datahandlers.other
import datahandlers.las

DATAHANDLERS = {}


# ## Add from all datahandler files ###
# data
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.data.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    DATAHANDLERS[k] = v

# modifiers
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.modifiers.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    DATAHANDLERS[k] = v

# generating
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.generating.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    DATAHANDLERS[k] = v

# generating_function
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.generating_function.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    DATAHANDLERS[k] = v

# combining
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.combining.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    DATAHANDLERS[k] = v

# other
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.other.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    DATAHANDLERS[k] = v

# las
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.las.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    DATAHANDLERS[k] = v

# blender
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.blender.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    DATAHANDLERS[k] = v

# terrain
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.terrain.__name__], inspect.isclass)
for (k, v) in clsmembers_pairs:
    DATAHANDLERS[k] = v
