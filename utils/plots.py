import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.cm as cm
matplotlib.pyplot.switch_backend('Agg')  # To run on keb
import colorcet as cc
import matplotlib.ticker as plticker


def new_fig(nrows=1, ncols=1, **kwargs):
    """Create a new matplotlib figure containing one axis

    Return:
      (fig, ax) or (fig, list[ax])
    """
    fig = Figure(**kwargs)
    FigureCanvas(fig)
    if nrows == 1 and ncols == 1:
        # Create a single axis
        ax = fig.add_subplot(1, 1, 1)
        return fig, ax
    else:
        # Create a list of axes
        axs = [fig.add_subplot(nrows, ncols, index) for index in
               range(1, nrows*ncols+1)]
        return fig, axs


def new_3d_fig(**kwargs):
    """ Test 3D subplot """
    fig = Figure(**kwargs)
    FigureCanvas(fig)
    ax = fig.add_subplot(111, projection='3d')

    return fig, ax


def save_all_axes(fig, base_filename, ticks_in=True, delta=0.05,
                  dx=None, dy=None, clear_legend=False, axis_off=False,
                  dpi=None, **savefig_kwargs):
    '''
    Save all axes as separate images.

    Args:
      ...
      delta:
      dx: explicity delta in x
      dy: explicity delta in y
    '''
    for i, ax in enumerate(fig.get_axes()):
        if ticks_in:
            ax.tick_params(direction="in", which='both')

        if axis_off:
            ax.axis('off')

        if clear_legend:
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_text(' '*int(1.5*len(text.get_text())))

        filename = base_filename.replace('.png', '_{:01d}.png'.format(i))
        save_subfig(fig, ax, filename, delta=delta, dpi=dpi, **savefig_kwargs)


def strip_axis(ax):
    '''
    Remove all labels etc. from axis
    '''
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    try:
        ax.get_legend().remove()
    except AttributeError:
        pass


def strip_figure(fig):
    '''
    Remove all visible text from figure
    '''
    for txt in fig.texts:
        txt.set_visible(False)
        txt.set_text("")


def save_subfig(fig, ax, filename, delta=0.05, extent=None, dx=None, dy=None,
                dpi=None, **savefig_kwargs):
    '''
    Save subfigure containing axis
    '''
    if extent is None:
        extent = get_bbox(fig, ax, delta, dx, dy)
        print("extent:{}".format(extent))
        print("extent.width:{}".format(extent.width))
        print("extent.height:{}".format(extent.height))
    fig.savefig(filename, bbox_inches=extent, dpi=dpi, **savefig_kwargs)


def get_bbox(fig, ax, delta=0.05, dx=None, dy=None):
    '''
    Calculate bounding box around axis
    '''
    extent = ax.get_window_extent().transformed(
        fig.dpi_scale_trans.inverted())
    # get points
    p = extent.get_points()
    # Calculate new extent
    x0 = p[0, 0]
    y0 = p[0, 1]
    w = p[1, 0] - x0
    h = p[1, 1] - y0
    # d = delta
    dx = delta if dx is None else dx
    dy = delta if dy is None else dy
    extent = extent.from_bounds(x0 - dx, y0 - dy, w + 2 * dx, h + 2 * dy)

    return extent


def add_right_panel(fig, ax):
    [[x0, y0], [x1, y1]] = ax.get_position().get_points()
    left, width = x0, x1-x0
    bottom, height = y0, y1-y0
    spacing = 0.02
    margin_height = 0.07
    rect = [left + width + spacing, bottom, margin_height, height]

    ax_histy = fig.add_axes(rect, sharey=ax)

    # Trick to turn of tick-labels just on marginal axis
    plt.setp(ax_histy.get_yticklabels(), visible=False)

    return ax_histy


def add_right_histogram(ax_right, values, barh_kwargs={}, hist_kwargs={}):
    ''' Given an axis, to the right of the main canvas, with 'share-y' '''
    hist, bin_edges = np.histogram(values, **hist_kwargs)
    bins = bin_edges[:-1] + (bin_edges[1] - bin_edges[0]) / 2
    height = 0.9 * (bin_edges[1] - bin_edges[0])
    barh = ax_right.barh(y=bins, width=hist, height=height, **barh_kwargs)

    return barh


def add_multicolored_line(
        ax, y, x=None, colors=None, color_values=None,
        linewidth=2, cmap=None,
):
    ''' Add plot to axis, with multiple colors

    Each line consists of multiple segments, and these are all
    colored differently

    Args:
      ax: (plt axis)
      y: (np.array) of shape (N,)
      x: (np.array) of shape (N,), or None (-> infered from y)
      colors: (np.array) of shape (N, 4), or None
      color_values: (np.array) of shape (N,) or None
      cmap: None or matplotlib.cm

    Return:
      ax, line

    '''
    # From
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html

    from matplotlib.collections import LineCollection

    # Infer x from y, if it is not given
    if x is None:
        x = np.linspace(0, y.shape[0], y.shape[0])

    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create 'LineCollection'
    # Set colors either from 4-vector or 1-vector and cmap
    if colors is not None:
        lc = LineCollection(segments, colors=colors)
    else:
        lc = LineCollection(segments, cmap=cmap)
        # Use color_values. Has to be 1D vector
        lc.set_array(color_values.squeeze())

    lc.set_linewidth(linewidth)

    line = ax.add_collection(lc)

    return ax, line


def add_4m_grid(ax, **_):
    # loc = plticker.MultipleLocator(base=4.0)
    # ax.xaxis.set_minor_locator(loc)
    # ax.yaxis.set_minor_locator(loc)
    # ax.grid(color='k', which='major')
    # ax.grid(color='k', which='minor', linewidth=0.25)
    add_grid(ax, size=4)
    add_grid(ax, size=20, linewidth=0.5)


def add_grid(ax, size=10, linewidth=0.25, color='k', **kwargs):
    '''
    Add a grid

    Args:
      size (float or [float, float])
    '''
    # TODO: Does not handle left/bottom limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xmax = xlim[1]
    ymax = ylim[1]

    if isinstance(size, (float, int)):
        size_x = size
        size_y = size
    elif isinstance(size, list):
        size_x, size_y = size

    vlines = np.arange(0, xmax, size_x)
    hlines = np.arange(0, ymax, size_y)

    # Add to kwargs
    kwargs['linewidth'] = linewidth
    kwargs['color'] = color

    ax.vlines(vlines, 0, ymax, **kwargs)
    ax.hlines(hlines, 0, xmax, **kwargs)


def plot_obstacles(
        obstacles,
        filename=None, fig_ax=None,
        xlim=None, ylim=None,
        dpi=400, color='radius',
):
    '''
    Plot obstacles from arrays of position and radius

    Args:

      color: str or np.array
    '''
    radius = obstacles.radius
    position = obstacles.position

    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = new_fig()

    # If no limits are provided, and ax-limits have default values
    # Estimate limits
    # print(f"ax.get_autoscalex_on():{ax.get_autoscalex_on()}")
    if xlim is None and ax.get_xlim() == (0, 1):
        xlim = [np.min(position[0, :]), np.max(position[0, :])]
    if ylim is None and ax.get_ylim() == (0, 1):
        ylim = [np.min(position[1, :]), np.max(position[1, :])]

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if isinstance(color, str) and color == 'radius':
        c = radius
    else:
        c = color

    # Draw canvas to be able to get size
    xlim = ax.get_xlim()
    ax.set_aspect(1)
    ax_width = xlim[1] - xlim[0]
    fig.canvas.draw()

    # Plot points
    factor = ((2 * ax.get_window_extent().width/ax_width * 72./fig.dpi) ** 2)
    s = factor * radius**2
    ax.scatter(*position, edgecolor='none', s=s, cmap=cc.cm.rainbow, c=c)

    if filename is not None:
        fig.savefig(filename, dpi=dpi)
    else:
        return fig, ax


def plot_image(
        array_2d, filename=None, fig_ax=None,
        extent=None, cmap=cc.cm.gray, xlim=None, ylim=None, **imshow_kwargs):
    '''
    Plot some image...

    Args:
      extent=None or [xmin, xmax, ymin, ymax],
    '''
    if fig_ax is not None:
        fig, ax = fig_ax
    else:
        fig, ax = new_fig()

    # Set limits, unless already set
    # Use extent if given, or else size of array
    if xlim is None:
        if extent is None:
            xlim = [0, array_2d.shape[0]]
        else:
            xlim = extent[:2]
    if ylim is None:
        if extent is None:
            ylim = [0, array_2d.shape[1]]
        else:
            ylim = extent[2:]

    # Set extent unless it has already been set
    # As default is [-0.5, numcols-0.5] we have to set this
    if extent is None:
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    im = ax.imshow(
            array_2d.T, extent=extent,
            cmap=cmap, origin='lower', **imshow_kwargs)
    fig.colorbar(im)

    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    else:
        return fig, ax


def plot_probability(array_2d, filename=None, cmap=cc.cm.gray, vmin=0, **kwargs):
    return plot_image(array_2d, cmap=cmap, filename=filename, vmin=vmin, **kwargs)


def plot_diverging(array_2d, filename=None, cmap=cc.cm.coolwarm, vmin=-1, vmax=1, **kwargs):
    return plot_image(
        array_2d,
        filename=filename,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        **kwargs)


def plot_y_array(array_2d, grid, filename=None, cmap=cc.cm.blues, vmin=1, vmax=5, **kwargs):
    return plot_image(
        array_2d,
        filename=filename,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=grid.extent,
        **kwargs)


def plot_histogram(array, bins=10, range=None, filename=None):
    # Construct histogram
    hist, bin_edges = np.histogram(array, bins, range)
    # Plot histogram
    fig, ax = new_fig()
    ax.bar(np.arange(bins)+1, hist)

    if filename is not None:
        fig.savefig(filename)
    else:
        return fig, ax


def plot_hf(hf_array, extent, ax=None, fig=None,
            label='Height (m)', **_):
    '''
    plot hf
    '''
    if ax is None:
        fig, ax = new_fig()

    # Plot image
    im = ax.imshow(hf_array.T, origin='lower', extent=extent)
    fig.colorbar(im, label=label)

    return fig, ax


def plot_terrain(terrain, ax=None, fig=None,
                 label='Height (m)', **_):
    '''
    plot hf
    '''
    if ax is None:
        fig, ax = new_fig()

    # Plot image
    im = ax.imshow(terrain.array.T, origin='lower', extent=terrain.extent)
    fig.colorbar(im, label=label)

    return fig, ax
