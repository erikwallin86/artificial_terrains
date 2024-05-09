from utils.plots import debug_plot
import os


global plot_debug_counter
plot_debug_counter = 0
plot_debug_enabled = False


def debug(datahandler, kwargs, call_number, call_total, name):
    global plot_debug_enabled, plot_debug_counter

    if not plot_debug_enabled or kwargs is None:
        return

    # Plot input
    filename = f'debug_{plot_debug_counter:05d}_{name}_{datahandler.name}_{call_number+1}_{call_total}.png'
    filename = os.path.join(datahandler.save_dir_original, filename)
    debug_plot(filename, **kwargs)
    plot_debug_counter += 1
