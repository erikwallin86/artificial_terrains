import os
from utils.parse_utils import create_parser, get_args_combined_with_settings
from datahandlers.datahandlers import DATAHANDLERS
from utils.logging_utils import get_logger, level_map
from datahandlers.blender import fix_blender_argv, fix_blender_path

fix_blender_path()


def main():
    fix_blender_argv()

    parser = create_parser(specific_args=['datahandlers'])
    args = get_args_combined_with_settings(parser)

    # Create folder if needed
    save_dir = args.save_dir
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Add logger
    logger = get_logger(save_dir=save_dir)
    logger.setLevel(level_map[args.debug])

    # Parse settings
    settings = args.settings if args.settings is not None else {}

    # Check that all datahandlers exist
    for key, value in args.datahandlers:
        if key not in DATAHANDLERS.keys():
            logger.info(f"Invalid choice {key}. ({DATAHANDLERS.keys()})")
            exit(1)

    # Prepend 'BasicSetup' datahandler
    args.datahandlers = [('BasicSetup', None)] + args.datahandlers

    pipe = {
        'size': 50,  # default
        'resolution': 100,  # default
    }
    pipes = [pipe]

    # Extract general settings (not given by dicts)
    general_kwargs = {}
    for k, v in settings.items():
        if type(v) is not dict:
            general_kwargs[k] = v

    # Create and run datahandlers
    for datahandler, options in args.datahandlers:
        # Skip if no datahandlers
        if datahandler is None:
            continue
        datahandler_class = DATAHANDLERS[datahandler]
        datahandler_obj = datahandler_class(save_dir, logger)
        # Extract possible settings kwargs
        if isinstance(options, dict):
            kwargs = options
        elif options is None:
            kwargs = {}
        else:
            # Store value as 'default'
            kwargs = {'default': options}

        new_pipes = []
        # Run datahandler for each pipe
        for i, pipe in enumerate(pipes):

            # Run datahandler object, with different inputs, and settings dict
            returned_data = datahandler_obj(
                call_number=i,
                call_total=len(pipes),
                **{**pipe, **general_kwargs, **kwargs},
            )
            logger.debug(f"type(returned_data):{type(returned_data)}")

            if returned_data is None:
                logger.debug("### returned_data is None:")
                # Simply reuse same pipe
                new_pipes.append(pipe)
            elif type(returned_data) is dict:
                logger.debug("### type(returned_data) is dict:")
                # Update pipe and add to new pipes
                pipe = {**pipe, **returned_data}
                # Clean pipe, if any of the returns are None
                pipe = {k: v for k, v in pipe.items() if v is not None}
                new_pipes.append(pipe)
            elif type(returned_data) is list:
                logger.debug("### type(returned_data) is list:")
                # Use returned list
                new_pipes.extend(returned_data)
            elif returned_data == 'remove':
                logger.debug("### returned_data == 'remove':")
                pass
            else:
                logger.debug(f"### returned_data is {type(returned_data)}:")
                # Simply reuse same pipe
                new_pipes.append(pipe)

        # Update pipes before next run
        logger.debug(f"len(new_pipes):{len(new_pipes)}")
        pipes = new_pipes


if __name__ == "__main__":
    main()
