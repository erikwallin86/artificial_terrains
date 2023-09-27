import sys
import inspect
import datahandlers.data
import datahandlers.blender
import datahandlers.terrain


DATAHANDLERS = {}


# ## Add from all datahandler files ###
# data
clsmembers_pairs = inspect.getmembers(
    sys.modules[datahandlers.data.__name__], inspect.isclass)
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
