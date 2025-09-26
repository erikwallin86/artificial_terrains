import argparse
import sys
import yaml
import os
import re


def create_parser(parser=None, specific_args=[], positional=[]):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Common arguments
    parser.add_argument(
        '--save-dir', type=str, default=None, required=True,
        help='Explicit save dir')
    parser.add_argument(
        '--settings-file', type=str, default=None,
        help='File to load Env settings from')
    parser.add_argument(
        '--settings', type=str, nargs='+', action=StoreDict,
        help='Overwrite settings (e.g. log:cylinder pile:single)'
    )
    parser.add_argument(
        '--debug', type=int, choices=[0, 1, 2, 3], default=2,
        help='Set debug level: 0 (NOTSET), 1 (DEBUG), 2 (INFO), 3 (WARNING)')

    # Handle specific arguments
    for name in specific_args:
        if name == 'modules':
            parser.add_argument(
                '--modules', type=str, help='Modules to use',
                action=StoreTuplePair,
                default=(None,), nargs='+', required=True,
            )

    return parser


def parse_options(options):
    ''' parse 'options' to make it a dict '''
    # Extract possible settings kwargs
    if isinstance(options, dict):
        kwargs = options
    elif options is None:
        kwargs = {}
    else:
        # Store value as 'default'
        kwargs = {'default': options}

    return kwargs


def get_args_combined_with_settings(parser):
    '''
    Combine args with settings, complementing argparse from settings-file

    Input can be given via
     1) a yaml settings file,
     2) a settings dict, or
     3) argparse argument

    # Examples

    1): python script.py --settings-file with
    spawn-type:
      AtRock:
        num_spawns: 1
        plot: True

    (not designed for TuplePairs, misses information here)
    2): python script --settings spawn-type:AtRock

    3) python script.py --spawn-type AtRockEdge:"dict(num_spawns=1)"

    Priorities, from lowest to highest
      --settings-file
      --settings
      --argparse argument
    '''
    # Get args
    args, _ = parser.parse_known_args()

    # Make args dict
    args_dict = vars(args)

    # Construct a mapping between args-names and 'actions'
    # To know if an argument is a StoreDict, TuplePair
    argument_action_mapping = {}
    for action in parser._actions:
        try:
            action_name = type(action).__name__
            name = action.option_strings[0].replace('--', '').replace('-', '_')
            argument_action_mapping[name] = action_name
        except IndexError:
            pass

    # Construct combined 'settings' from settings-file and --settings
    settings = load_settings(args.settings_file, args)

    # Possibly take options from settings?
    for args_name, args_value in args_dict.items():
        lowercase_name = args_name
        dash_name = f"{lowercase_name.replace('_', '-')}"
        from_commandline = f'--{dash_name}' in sys.argv[1:]
        # Get action type
        action = None
        if lowercase_name in argument_action_mapping:
            action = argument_action_mapping[lowercase_name]

        # If in settings, and not explicitly given via command line
        # --> take from settings
        if not from_commandline and dash_name in settings:
            settings_value = settings[dash_name]

            # Handle TuplePairs
            if action == 'StoreTuplePair':
                if type(settings_value) is str:
                    tuple_pair = [(settings_value, None)]
                elif type(settings_value) is dict:
                    tuple_pair = []
                    for k, v in settings_value.items():
                        tuple_pair.append((k, v))
                        # tuple_pair.append([k, v])
                elif type(settings_value) is list:
                    tuple_pair = settings_value

                setattr(args, lowercase_name, tuple_pair)
            if action == '_StoreAction':
                setattr(args, lowercase_name, settings_value)
            elif action == 'StoreDict':
                print(f"action:{action}")
                setattr(args, lowercase_name, settings_value)
            elif action is None:
                setattr(args, lowercase_name, settings_value)

        # if explicitly given via command line
        # --> add to settings
        if from_commandline:
            if action == '_StoreAction':
                settings[dash_name] = args_value
            elif action == 'StoreTuplePair':
                # Change from list of tuple, to dict.
                # From: [('AtRock', {'num_spawns': 100})]
                # To:   {'AtRock': {'num_spawns': 100}}

                # Fix tuple to list
                args_value = [list(t) for t in args_value]
                # settings[dash_name] = {args_value[0][0]: args_value[0][1]}
                # Append to list if already in settings and is a list.
                if dash_name in settings and isinstance(settings[dash_name], list):
                    settings[dash_name].extend(args_value)
                else:
                    # other wise we just set the value
                    settings[dash_name] = args_value
            elif action == 'StoreDict':
                # TODO: Not tested
                settings[dash_name] = args_value
            elif action is None:
                # TODO: Not tested
                settings[dash_name] = args_value

        # if not in settings
        # --> add to settings
        if dash_name not in settings:
            settings[dash_name] = args_value

    # Possibly remove 'settings-file' from settings to avoid recursion
    if 'settings-file' in settings:
        del settings['settings-file']
    if 'settings' in settings:
        del settings['settings']

    # Create save_dir
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # ## module specific code ###
    # if not [name, option] pair, (only name)
    from modules.modules import MODULES
    for i, pair in enumerate(args.modules):
        if not isinstance(pair, (tuple, list)):
            key = pair
            if key in MODULES.keys():
                args.modules[i] = [key, None]

    # Save settings
    if args.save_dir is not None:
        filename = os.path.join(args.save_dir, 'settings.yml')

        # ## module specific code ###
        # Process modules to simplify yaml file
        # Remove [name, None] and turn to 'name'
        if 'modules' in settings:
            # Process settings to remove unnecessary null's
            processed_modules = [
                [name, options] if options is not None else
                name for name, options in settings['modules']]
            settings['modules'] = processed_modules

        with open(filename, 'w') as f:
            yaml.dump(settings, f, sort_keys=False, default_flow_style=None)

    return args


def load_settings(yaml_file, args):
    import yaml

    # Yaml bug workaround,
    # https://stackoverflow.com/questions/30458977/yaml-loads-5e-6-as-string-and-not-a-number
    # (seem to be work to add this to yaml main)
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    if yaml_file is not None:
        # Load settings from yaml file
        with open(yaml_file, 'r') as f:
            # settings_dict = yaml.safe_load(f)
            settings = yaml.load(f, Loader=loader)
            # if env_str in list(settings_dict.keys()):
            #     settings = settings_dict[env_str]
            # else:
            #     # raise ValueError("Settings not found for {}".format(env_str))
            #     settings = settings_dict
    else:
        # Else populate with empty dict
        settings = {}

    # Overwrite hyperparams from command line, if exists
    if args is not None and args.settings is not None:
        for k, v in args.settings.items():
            if type(v) is not dict:
                settings[k] = v
            # Solution for overwriting single kwargs of second 'level'
            elif k not in settings:
                settings[k] = v
            else:
                for k2, v2 in v.items():
                    settings[k][k2] = v2

    return settings


class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings,
                                        dest,
                                        nargs=nargs,
                                        **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            arg_dict_local = self.split(arguments)
            arg_dict = {**arg_dict, **arg_dict_local}
        setattr(namespace, self.dest, arg_dict)

    def split(self, arguments):
        arg_dict = {}
        key = arguments.split(":")[0]
        value = ":".join(arguments.split(":")[1:])
        # Evaluate the string as python code
        try:
            if ':' in value:
                arg_dict_lower = self.split(value)
                arg_dict[key] = arg_dict_lower
            else:
                arg_dict[key] = eval(value)
        except NameError:
            arg_dict[key] = value
        except SyntaxError:
            return {key: value}

        return arg_dict


class StoreTuplePair(argparse.Action):
    """
    Custom argparse action for storing tuple pair.

    In:
    Out:
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreTuplePair, self).__init__(
            option_strings,
            dest,
            nargs=nargs,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # Store list of tuple-pairs
        tuple_pair_list = []
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            try:
                tuple_pair_list.append((key, eval(value)))
            except NameError:
                tuple_pair_list.append((key, value))
            except SyntaxError:
                if value == "":
                    tuple_pair_list.append((key, None))
                else:
                    tuple_pair_list.append((key, value))
        setattr(namespace, self.dest, tuple_pair_list)


class StoreTuplePair2(argparse.Action):
    """
    Another way of implementing the TuplePair.
    I'm not sure if they are different...

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreTuplePair2, self).__init__(
            option_strings,
            dest,
            nargs=nargs,
            **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        # arg_dict = {}
        tuple_pair_list = []
        for arguments in values:
            arg_dict_local = self.split(arguments)
            for key, value in arg_dict_local.items():
                tuple_pair_list.append((key, value))

        setattr(namespace, self.dest, tuple_pair_list)

    def split(self, arguments):
        arg_dict = {}
        key = arguments.split(":")[0]
        value = ":".join(arguments.split(":")[1:])
        # Evaluate the string as python code
        try:
            if ':' in value:
                arg_dict_lower = self.split(value)
                arg_dict[key] = arg_dict_lower
            else:
                arg_dict[key] = eval(value)
        except NameError:
            arg_dict[key] = value
        except SyntaxError:
            return {key: value}

        return arg_dict
