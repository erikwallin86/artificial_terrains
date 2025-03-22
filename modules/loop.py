from modules.data import Module


class Loop(Module):
    ''' Basic loop '''
    create_folder = False

    def start(self, default=None, n_loops=2, **kwargs):
        n_loops = default if default is not None else n_loops
        self.n_loops = n_loops
        self.loop_generator_instance = self.loop_generator()

    def loop_generator(self):
        call_total = self.n_loops
        for call_number in range(self.n_loops):
            self.logger.info(f"Run {self.name} {call_number+1}/{call_total}")
            return_dict = {
                'call_number': call_number,
                'call_total': call_total,
            }
            yield return_dict
