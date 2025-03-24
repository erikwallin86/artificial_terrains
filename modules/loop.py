from modules.data import Module
import numpy as np


class Loop(Module):
    ''' Basic loop '''
    create_folder = False

    def start(self, default=None, n_loops=2, loop_id=None, call_total=None,
              call_number=None, **kwargs):
        self.n_loops = default if default is not None else n_loops
        self.loop_generator_instance = self.loop_generator()

        self.loop_id = "" if loop_id is None else loop_id
        self.digits = int(2 + np.log10(self.n_loops))

        # We save the raw input
        self.call_total = call_total
        self.call_number = call_number

        self.logger.info(f"INIT {self.name}")
        print(f"##### self.call_total:{self.call_total}")
        print(f"##### self.call_number:{self.call_number}")

    def loop_generator(self):
        for call_number in range(self.n_loops):
            # Custom print, as the debug_decorator prints don't work for the
            # generator setup. Also, here we loop over 'self.n_loops'
            self.logger.info(f"Run {self.name} {call_number+1}/{self.n_loops}")

            # Update loop-id, to be used in filenames
            loop_id = self.loop_id + f"_{call_number:0{self.digits}d}"

            # Construct return_dict, and fix the call_total and call_number
            # in case of several loops.
            # call_total: multiply with length of the number of loops in this
            # call_number: add number of previous
            return_dict = {
                'call_total': self.call_total * self.n_loops,
                'call_number': call_number + self.call_number*self.n_loops,
                'loop_id': loop_id,
            }
            yield return_dict
