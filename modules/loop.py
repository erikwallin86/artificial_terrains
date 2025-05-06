from modules.data import Module
import numpy as np
from itertools import zip_longest



class Loop(Module):
    ''' Basic loop '''
    create_folder = False

    def start(self, default=None, n_loops=2, loop_id=None, loop_id_r=None, call_total=None,
              call_number=None, parameter=None, expression=None, values=None,
              **kwargs):

        # Handle if default is given a parameter=expression
        if isinstance(default, str) and '=' in default:
            parameter, expression = default.split('=')
            values = eval(expression)
            self.n_loops = len(values)
        elif isinstance(default, int):
            # Length of the loop
            self.n_loops = default if default is not None else n_loops
        elif isinstance(default, str) and 'x' in default:
            try:
                a, b = map(int, default.lower().split('x'))
                self.n_loops = a * b
            except ValueError:
                raise ValueError(f"Invalid format for default: {default}. Expected format like '2x3'.")
            else:
                # Construct a list of extent 'edges'
                x_min, x_max, y_min, y_max = kwargs['extent']
                x_split = np.linspace(x_min, x_max, a+1).astype(float).tolist()
                y_split = np.linspace(y_min, y_max, b+1).astype(float).tolist()
                # Construct a list of new extents
                self.list_of_new_extents = []
                for i in range(a):
                    for j in range(b):
                        extent = x_split[i:i+2] + y_split[j:j+2]
                        self.list_of_new_extents.append(extent)
        else:
            pass

        # Store any parameter or values
        self.parameter = parameter
        self.values = values if values is not None else []

        # Create a instance of the generator
        self.loop_generator_instance = self.loop_generator()

        # Initialize a 'loop_id', and number of digits to use in it
        self.loop_id = "" if loop_id is None else loop_id
        # 'reverse' loop_id, showing remaining calls
        self.loop_id_r = "" if loop_id_r is None else loop_id_r
        self.digits = int(2 + np.log10(self.n_loops))

        # We save the raw input
        self.call_total = call_total
        self.call_number = call_number

    def loop_generator(self):
        for call_number, value in zip_longest(range(self.n_loops), self.values):
            # Custom print, as the debug_decorator prints don't work for the
            # generator setup. Also, here we loop over 'self.n_loops'
            self.logger.info(f"Run {self.name} {call_number+1}/{self.n_loops}")

            # Update loop-id, to be used in filenames
            loop_id = self.loop_id + f"_{call_number:0{self.digits}d}"

            loop_id_r = self.loop_id_r + f"_{self.n_loops - call_number - 1:0{self.digits}d}"

            # Construct return_dict, and fix the call_total and call_number
            # in case of several loops.
            # call_total: multiply with length of the number of loops in this
            # call_number: add number of previous
            return_dict = {
                'call_total': self.call_total * self.n_loops,
                'call_number': call_number + self.call_number*self.n_loops,
                'loop_id': loop_id,
                'loop_id_r': loop_id_r,
            }

            # Possibly take new extent etc. from list
            try:
                extent = self.list_of_new_extents[call_number]
                return_dict['extent'] = extent
                return_dict['size'] = [extent[1] - extent[0], extent[3] - extent[2]]
            except AttributeError:
                pass

            if self.parameter is not None and value is not None:
                return_dict[self.parameter] = value

            yield return_dict


class Unloop(Module):
    ''' Test... '''
    create_folder = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Create a dict which will be used to merge the pipes from the
        # 'collected loops'
        self.new_pipe = {}

    def start(self, default=None, call_total=None, loop_id=None, loop_id_r=None,
              call_number=None, parameter=None, expression=None, values=None,
              overwrite=False,  # just to keep it from entering 'pipe'
              merge_function=None,  # TODO: Implement?
              **kwargs):
        # We save the raw input
        self.call_total = call_total

        self.loop_id = loop_id
        self.loop_id_r = loop_id_r

        self.loop_generator_instance = self.loop_generator()

        # We 'restart' with the call-number
        self.call_number = 0

        # Merge data
        for k, v in kwargs.items():
            if k == 'terrain_temp':
                if 'terrain_temp' in self.new_pipe:
                    self.new_pipe['terrain_temp'].extend(v)
                else:
                    self.new_pipe['terrain_temp'] = v
            elif k == 'terrain_heap':
                if 'terrain_heap' in self.new_pipe:
                    self.new_pipe['terrain_heap'].extend(v)
                else:
                    self.new_pipe['terrain_heap'] = v
            else:
                self.new_pipe[k] = v
  
    def loop_generator(self):
        remaining_on_last_loop = int(self.loop_id_r.split("_")[-1])

        if remaining_on_last_loop != 0:
            # Return, i.e. don't be iterated over.
            # The effect is that the 'next module' will not be called,
            # and instead we return to the previous 'loop' module.
            # The idea is that we do this to 'collect' data, until the
            # last value from the loop. Then, we somehow 'merge'
            # all the collected data, and (finally) pass control along to the next module
            return
        else:
            # Calculate 'remaining' loop id
            loop_id = self.loop_id.rsplit('_', maxsplit=1)[0]
            loop_id_r = self.loop_id_r.rsplit('_', maxsplit=1)[0]

            # Construct a list of remaining iterations in each loop
            remaining = [int(id_r) for id_r in loop_id_r.split("_") if id_r]
            # Use this to calculated the remaining number of calls: call_total
            if len(remaining) > 0:
                call_total = np.prod(np.array(remaining)) + 1
            else:
                call_total = 1

            result = {
                'loop_id': loop_id,
                'loop_id_r': loop_id_r,
                'call_number': self.call_number,
                'call_total': call_total,
            }

            self.call_number += 1

            # Update result, and reset 'new_pipe'
            result = {**self.new_pipe, **result}
            # Reset
            self.new_pipe = {}

            yield result
