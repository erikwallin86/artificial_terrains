from modules.data import Module
import numpy as np


class Loop(Module):
    ''' Basic loop '''
    create_folder = False

    def start(self, default=None, n_loops=2, loop_id=None, call_total=None, **kwargs):
        self.n_loops = default if default is not None else n_loops
        self.loop_generator_instance = self.loop_generator()

        self.loop_id = "" if loop_id is None else loop_id
        self.digits = int(2 + np.log10(self.n_loops))

        self.call_total = call_total * self.n_loops

    def loop_generator(self):
        for call_number in range(self.n_loops):
            # Custom print, as the debug_decorator prints don't work for the
            # generator setup. Also, here we loop over 'self.n_loops'
            self.logger.info(f"Run {self.name} {call_number+1}/{self.n_loops}")

            # Update loop-id, to be used in filenames
            loop_id = self.loop_id + f"_{call_number:0{self.digits}d}"

            return_dict = {
                'call_number': call_number,
                'call_total': self.call_total,
                'loop_id': loop_id,
            }
            yield return_dict
