import logging
import sys
from datetime import datetime
import os


# Map debug levels to logging levels
level_map = {
    0: logging.NOTSET,
    1: logging.DEBUG,
    2: logging.INFO,
    3: logging.WARNING
}


def get_logger(name='logger', save_dir=None, level=logging.INFO):
    '''
    Create logger with stdout- (and possibly file-) handler
    '''
    # Make sure directory exists
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Construct filename
    if save_dir is not None:
        filename = os.path.join(save_dir, f'{name}.txt')
    else:
        filename = None

    logger = logging.getLogger(name)
    # Add stdout handler
    stream = logging.StreamHandler(sys.stdout)
    stream.setLevel(logging.DEBUG)  # ?
    logger.addHandler(stream)
    if filename is not None:
        # Add file handler
        hdlr = logging.FileHandler(filename)
        logger.addHandler(hdlr)
    # Set overall level
    logger.setLevel(level)

    # Output command line input
    command_line = ' '.join(sys.argv)
    command_line = parse_command(command_line)

    # workaround
    if 'python blender.py' in command_line:
        command_line = command_line.replace(
            'python blender.py',
            'blender --python blender.py --')

    logger.info(f'### {command_line}')
    logger.info(datetime.now())

    return logger


def parse_command(command_line):
    ''' Fix things disappeared from command line string

    Add double qoutes around dict(..)
    Add 'python' to start
    '''
    import re
    pattern = r"dict\([^)]+\)"

    def add_quotation_marks(match):
        matched_string = match.group(0)
        return f'"{matched_string}"'

    # Replace occurrences of dict(...) with 'dict(...)'
    modified_string = re.sub(pattern, add_quotation_marks, command_line)
    modified_string = 'python ' + modified_string

    return modified_string
