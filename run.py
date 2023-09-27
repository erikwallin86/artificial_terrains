import os
import argparse
from utils.utils import StoreTuplePair, StoreDict
from datahandlers.datahandlers import DATAHANDLERS
from utils.logging_utils import get_logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--settings', type=str, nargs='+', action=StoreDict,
        help='Overwrite settings (e.g. use_pid:True control:"torque")'
    )
    parser.add_argument(
        '--datahandlers', type=str, help='Datahandlers to use',
        nargs='+', action=StoreTuplePair,
    )
    parser.add_argument('--settings-file', type=str, default=None,
                        help='File to load settings from')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='where to save data')
    # Parse arguments
    args, _ = parser.parse_known_args()
    print(f"args.datahandlers:{args.datahandlers}")

    # Create folder if needed
    save_dir = args.save_dir
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Add logger
    logger = get_logger(save_dir=save_dir)
    # import logging
    # logger.setLevel(logging.DEBUG)

    # Parse settings
    settings = args.settings if args.settings is not None else {}

    # Check that all datahandlers exist
    for key, value in args.datahandlers:
        if key not in DATAHANDLERS.keys():
            logger.info(f"Invalid choice {key}. ({DATAHANDLERS.keys()})")
            exit(1)

    pipe = {}
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
