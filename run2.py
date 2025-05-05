import os
import sys
try:
    import bpy
    # Make sure current dir is in path. A blender py workaround
    dir = os.path.dirname(bpy.data.filepath)
    if dir not in sys.path:
        sys.path.append(dir)
except Exception:
    pass
else:
    from modules.blender import fix_blender_argv, fix_blender_path
    fix_blender_path()
    fix_blender_argv()

from utils.parse_utils import create_parser, get_args_combined_with_settings
from modules.modules import MODULES
from utils.logging_utils import get_logger, level_map


def main():
    parser = create_parser(specific_args=['modules'])
    args = get_args_combined_with_settings(parser)

    # Create folder if needed
    save_dir = args.save_dir
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Add logger and set level
    logger = get_logger(save_dir=save_dir)
    logger.setLevel(level_map[args.debug])

    # Parse settings
    settings = args.settings if args.settings is not None else {}

    # Check that all modules exist
    for key, value in args.modules:
        if key not in MODULES.keys():
            logger.info(f"Invalid choice {key}. ({MODULES.keys()})")
            exit(1)

    pipe = {
        'size': 50,  # default
        'ppm': 2,  # default
        'extent': [-25, 25, -25, 25],  # default
        'call_number': 0,
        'call_total': 1,
    }

    # Extract general settings (not given by dicts)
    general_kwargs = {
    }
    for k, v in settings.items():
        if type(v) is not dict:
            general_kwargs[k] = v

    list_of_modules_kwargs_tuples = []

    # Setup list of modules and kwargs
    for module, options in args.modules:
        # Skip if no modules
        if module is None:
            continue
        module_class = MODULES[module]
        module_obj = module_class(save_dir, logger)
        # Extract possible settings kwargs
        if isinstance(options, dict):
            kwargs = options
        elif options is None:
            kwargs = {}
        else:
            # Store value as 'default'
            kwargs = {'default': options}

        # combine with general kwargs
        kwargs = {**general_kwargs, **kwargs}
        # Add to list
        list_of_modules_kwargs_tuples.append((module_obj, kwargs))

    # Start recursive call of modules
    recursive_module_call(
        list_of_modules_kwargs_tuples, pipe=pipe, logger=logger,
    )


def recursive_module_call(
        list_of_modules_kwargs_tuples, index=0, pipe={}, logger=None):
    '''
    Call next level recursivly
    '''
    logger.debug(f"### index: {index}")
    # Base case: reached the end of the module list
    if index >= len(list_of_modules_kwargs_tuples):
        return None

    # Get module and kwargs
    module_obj, kwargs = list_of_modules_kwargs_tuples[index]
    logger.debug(f"  module_obj: {module_obj}")
    logger.debug(f"  kwargs: {kwargs}")
    logger.debug(f"  pipe: {pipe}")

    module_obj.start(**kwargs, **pipe)
    for returned_data in module_obj:
        if isinstance(returned_data, dict):
            pipe = {**pipe, **returned_data}

        recursive_module_call(
            list_of_modules_kwargs_tuples, index+1, pipe, logger)
        # # Update pipe, with data from the module to 'the right'
        # if isinstance(return_from_right, dict):
        #     pipe = {**pipe, **return_from_right}

    # # When we return something here, it will be given to the module on the left
    # return pipe


if __name__ == "__main__":
    main()
