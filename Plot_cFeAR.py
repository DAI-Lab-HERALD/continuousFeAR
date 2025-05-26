import copy

import numpy as np

# -----------------------------------------------------------------------------
# figures.py needed to make some example plots
from math import sqrt
from shapely import affinity
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from shapely.plotting import plot_polygon, plot_points, plot_line, patch_from_polygon
from matplotlib import cm
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmcrameri.cm as cmc
import networkx as nx
import os
from collections.abc import Iterable

import PlotGWorld
from PlotGWorld import special_spectral_cmap

import ContinuousFeAR as cfear

SIZE = (10, 12)
# SIZE = (5, 6)


# BLUE = '#6699cc'
BLUE = '#407e9c'
# RED = '#ff3333'
RED = '#c3553a'

GRAY = '#999999'
LIGHTGRAY = '#cccccc'
DARKGRAY = '#333333'
YELLOW = '#ffcc33'
GREEN = '#339933'
BLACK = '#000000'
WHITE = '#ffffff'

CMAP = cm.get_cmap('Spectral')


def add_origin(ax, geom, origin):
    x, y = xy = affinity.interpret_origin(geom, origin, 2)
    ax.plot(x, y, 'o', color=GRAY, zorder=1)
    ax.annotate(str(xy), xy=xy, ha='center',
                textcoords='offset points', xytext=(0, 8))


def set_limits(ax, x0, xN, y0, yN):
    ax.set_xlim(x0, xN)
    ax.set_xticks(range(x0, xN + 1))
    ax.set_ylim(y0, yN)
    ax.set_yticks(range(y0, yN + 1))
    ax.set_aspect("equal")
    ax.grid(True, linestyle=':')


def match_xylims_to_fit(ax1=None, ax2=None):
    """
    Function to match the xlim and ylim of two axes so that the new lims fit all the content of both axes

    Parameters
    ----------
    ax1 : Matplotlib ax object
    ax2 : Matplotlib ax object

    Returns
    -------
    None
    """

    if ax1 is None:
        print('Missing ax1 !')
        return False

    if ax2 is None:
        print('Missing ax2 !')
        return False

    # Get current xlim and ylim for both figures
    xlim1 = ax1.get_xlim()
    ylim1 = ax1.get_ylim()

    xlim2 = ax2.get_xlim()
    ylim2 = ax2.get_ylim()

    # Determine the combined xlim and ylim
    combined_xlim = (min(xlim1[0], xlim2[0]), max(xlim1[1], xlim2[1]))
    combined_ylim = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))

    # Set the combined xlim and ylim to both axes
    ax1.set_xlim(combined_xlim)
    ax1.set_ylim(combined_ylim)

    ax2.set_xlim(combined_xlim)
    ax2.set_ylim(combined_ylim)


def synchronize_axes_limits(fig):
    # Initialize variables to store the combined limits
    combined_xlim = [float('inf'), float('-inf')]
    combined_ylim = [float('inf'), float('-inf')]

    # Iterate through all axes in the figure to find the min and max limits
    for ax in fig.get_axes():
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        combined_xlim = [min(combined_xlim[0], xlim[0]), max(combined_xlim[1], xlim[1])]
        combined_ylim = [min(combined_ylim[0], ylim[0]), max(combined_ylim[1], ylim[1])]

    # Set the combined limits to all axes
    for ax in fig.get_axes():
        ax.set_xlim(combined_xlim)
        ax.set_ylim(combined_ylim)


# -----------------------------------------------------------------------------


def main():
    pass


if __name__ == "__main__":
    main()


def get_new_fig(arg=111, figsize=SIZE, dpi=300):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(arg)
    ax.grid(True, linestyle=':')
    ax.set_aspect("equal")
    make_axes_gray(ax)

    return fig, ax


def plot_box(ax=None, x=0, y=0, lx=0.1, ly=0.1, colour=BLUE, linewidth=1):
    if ax is None:
        fig = plt.figure(figsize=SIZE, dpi=90)
        ax = fig.add_subplot(121)

    xl = x - lx / 2
    xu = x + lx / 2
    yl = y - ly / 2
    yu = y + ly / 2

    ax.plot([xl, xu, xu, xl, xl], [yl, yl, yu, yu, yl], color=colour, linewidth=linewidth)
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':')


def plot_traj_boxes_from_xy(ax=None, x=np.array([0]), y=np.array([0]), lx=1, ly=1, colour=BLUE, label=None,
                            linewidth=None, stripped_axes=True):
    # For plotting the trajectory of one agent from the trajectory (x, y)
    if ax is None:
        fig = plt.figure(figsize=SIZE, dpi=90)
        ax = fig.add_subplot(121)

    boxes = cfear.get_boxes_for_a_traj(x, y, lx, ly)
    plot_box(ax, x=x[0], y=y[0], lx=lx, ly=ly, colour=BLACK, linewidth=1)

    for ii, box in enumerate(boxes):
        if ii == 0:
            plot_polygon(box, ax=ax, add_points=False, color=colour, label=label, linewidth=linewidth)
        else:
            plot_polygon(box, ax=ax, add_points=False, color=colour, linewidth=linewidth)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=':')
    set_xy_labels(ax)

    make_axes_gray(ax)

    if stripped_axes:
        strip_axes(ax=ax, strip_title=False)

    return boxes, ax


def plot_boxes_for_traj(traj_boxes, ax=None, colour=BLUE, label=None, linewidth=None, hatch=None, \
                        facecolor=None, stripped_axes=True):
    # For plotting the trajectory of one agent
    if ax is None:
        fig = plt.figure(figsize=SIZE, dpi=90)
        ax = fig.add_subplot(121)

    for ii, box in enumerate(traj_boxes):
        if ii == 0:
            plot_polygon(box, ax=ax, add_points=False, color=colour, label=label, linewidth=linewidth, hatch=hatch,
                         facecolor=facecolor)
        else:
            plot_polygon(box, ax=ax, add_points=False, color=colour, linewidth=linewidth, hatch=hatch,
                         facecolor=facecolor)

    # plot_polygon_outline(traj_boxes[0], color=DARKGRAY, ax=ax, linewidth=2)
    plot_polygon_outline(traj_boxes[0].buffer(0.2), color=BLACK, ax=ax, linewidth=2)

    ax.set_aspect("equal")
    ax.grid(True, linestyle=':')
    set_xy_labels(ax)

    make_axes_gray(ax)

    if stripped_axes:
        strip_axes(ax=ax, strip_title=False)

    return ax


def make_all_axes_gray(ax):
    for ax_ in ax.get_figure().axes:
        make_axes_gray(ax_)


def make_axes_gray(ax):
    plt.setp(ax.spines.values(), color='lightgray')
    ax.tick_params(labelcolor='dimgray', colors='lightgray')

    ax.xaxis.label.set_color('darkgray')
    ax.yaxis.label.set_color('darkgray')

    # ax.tick_params(axis='x', colors='lightgray')
    # ax.tick_params(axis='y', colors='lightgray')


def plot_boxes_for_trajs(trajs_boxes, obstacle=None, ax=None, title='', padding=10, colour=None, labels=None,
                         scale_scenario_size=1, show_legend=True,
                         linewidth=1.5, do_hatch=True, facecolor=None):
    # For plotting the trajectory of all agents
    if ax is None:
        fig = plt.figure(figsize=tuple(s * 0.75 * scale_scenario_size for s in SIZE))
        ax = fig.add_subplot()

    try:
        n_trajs, _ = trajs_boxes.shape
    except AttributeError:
        trajs_boxes = np.array(trajs_boxes)
        n_trajs, _ = trajs_boxes.shape

    if labels is not None:
        if len(labels) != n_trajs:
            print('Number of labels do not match the number of trajs. Reverting to default labels.')
            labels = None

    # show_legend = True

    # colours = special_spectral_cmap(n_colours=n_trajs)
    # colours = special_cmap(n_colours=n_trajs)
    colours = scientific_cmap(n_colours=n_trajs)

    for ii in range(n_trajs):
        # color = CMAP(ii / n_trajs)
        if colour is None:
            # color = CMAP(ii / len(trajs_hulls))
            color = colours[ii]
        else:
            color = colour
            if labels is None:
                show_legend = False  # Do not plot legend when no labels passed and all of them are the same colour
        # print(f'{trajs_boxes[ii, :]=}')

        if do_hatch is False:
            hatch = None
        # Cycling through Hatches for the agents to help distinguish them when their colours are similar
        elif ii % 3 == 0:
            hatch = None
        elif ii % 3 == 1:
            hatch = '....'
        else:
            hatch = '\\\\\\'

        if labels is None:
            plot_boxes_for_traj(traj_boxes=trajs_boxes[ii, :], ax=ax, colour=color, label=f'Agent {ii + 1}',
                                linewidth=linewidth, hatch=hatch, facecolor=facecolor)
        else:
            plot_boxes_for_traj(traj_boxes=trajs_boxes[ii, :], ax=ax, colour=color, label=f'Agent {labels[ii]}',
                                linewidth=linewidth, hatch=hatch, facecolor=facecolor)

    ax.grid(True, linestyle=':')
    ax.set_title(title)
    plt.axis()
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plot_obstacle(obstacle=obstacle, ax=ax, padding=padding)
    ax.set_aspect(1)
    extend_axis_limits(ax=ax, padding=padding)
    # plt.show()

    return ax


def plot_traj_hulls(traj_hulls, traj_boxes=None, ax=None, colour=BLUE, label=None, show_plot=False,
                    show_first=True, show_last=False, scale_scenario_size=1,
                    linewidth=None, hatch=None, facecolor=None, stripped_axes=False):
    if ax is None:
        fig = plt.figure(figsize=tuple(s * 0.75 * scale_scenario_size for s in SIZE))
        ax = fig.add_subplot()

    for ii, hull in enumerate(traj_hulls):
        if ii == 0:
            plot_polygon(hull, ax=ax, add_points=False, color=colour, label=label, linewidth=linewidth,
                         hatch=hatch, facecolor=facecolor)
        else:
            plot_polygon(hull, ax=ax, add_points=False, color=colour, linewidth=linewidth,
                         hatch=hatch, facecolor=facecolor)

    if show_first:
        if traj_boxes is not None:  # Plot outline of first box if available
            plot_polygon_outline(traj_boxes[0].buffer(0.2), color=BLACK, ax=ax, linewidth=2)
        else:
            plot_polygon_outline(traj_hulls[0].buffer(0.2), color=BLACK, ax=ax, linewidth=2)

    if show_last:
        if traj_boxes is not None:  # Plot outline of first box if available
            plot_polygon_outline(traj_boxes[-1].buffer(0.1), color=BLACK, ax=ax, linewidth=0.5)
        else:
            plot_polygon_outline(traj_hulls[-1].buffer(0.1), color=BLACK, ax=ax, linewidth=0.5)

    ax.set_aspect(1)
    ax.grid(True, linestyle=':')
    set_xy_labels(ax)

    make_axes_gray(ax)

    if stripped_axes:
        strip_axes(ax=ax, strip_title=False)

    if show_plot:
        plt.show()


def plot_hulls_for_trajs(trajs_hulls, trajs_boxes=None, obstacle=None, ax=None, padding=1,
                         title='', colour=None, show_plot=False, labels=None, linewidth=1.5,
                         show_legend=True, legend_inside=False, show_last=False, agents_to_colour=None,
                         scale_scenario_size=1, show_first=True,
                         do_hatch=True, facecolor=None):
    if ax is None:
        fig = plt.figure(figsize=tuple(s * 0.75 * scale_scenario_size for s in SIZE))
        ax = fig.add_subplot()

    if labels is not None:
        if len(labels) != len(trajs_hulls):
            print('Number of labels do not match the number of trajs. Reverting to default labels.')
            labels = None

    if colour is None:
        agents_to_colour = range(len(trajs_hulls))  # Colour all
    elif agents_to_colour is None:
        agents_to_colour = []  # Colour none

    # Check if colour is an array-like object
    if isinstance(colour, Iterable) and not isinstance(colour, (str, bytes)):
        # Ensure colour is a NumPy array (optional for easier handling)
        colour = np.array(colour)
        if len(colour) != len(trajs_hulls):
            raise ValueError("The length of the 'colour' array must match the number of trajs_hulls.")
        colours = colour  # Use the provided array of colours
        agents_to_colour = range(len(trajs_hulls))  # Colour all
    else:
        # colours = special_spectral_cmap(n_colours=len(trajs_hulls))
        # colours = special_cmap(n_colours=len(trajs_hulls))
        colours = scientific_cmap(n_colours=len(trajs_hulls))  # Default colour map

    # # colours = special_spectral_cmap(n_colours=len(trajs_hulls))
    # # colours = special_cmap(n_colours=len(trajs_hulls))
    # colours = scientific_cmap(n_colours=len(trajs_hulls))

    for ii, traj_hulls in enumerate(trajs_hulls):

        if ii in agents_to_colour:
            # color = CMAP(ii / len(trajs_hulls))
            color = colours[ii]
        else:
            color = colour
            if labels is None:
                show_legend = False  # Do not plot legend when no labels passed and all of them are the same colour
        # print(f'{color=}')

        if do_hatch is False:
            hatch = None
        # Cycling through Hatches for the agents to help distinguish them when their colours are similar
        elif ii % 3 == 0:
            hatch = None
        elif ii % 3 == 1:
            hatch = '....'
        else:
            hatch = '\\\\\\'

        if show_first and (trajs_boxes is None):  # No box information
            show_first = True  # Show first hull if no boxes are given.
        else:
            show_first = False

        if labels is None:
            plot_traj_hulls(ax=ax, traj_hulls=traj_hulls, colour=color, label=f'Agent {ii + 1}', show_first=show_first,
                            linewidth=linewidth, hatch=hatch, facecolor=facecolor)
        else:
            plot_traj_hulls(ax=ax, traj_hulls=traj_hulls, colour=color, label=f'Agent {labels[ii] + 1}',
                            show_first=show_first, linewidth=linewidth, hatch=hatch, facecolor=facecolor)

        if trajs_boxes is not None:  # Plot outline of first box
            plot_polygon_outline(trajs_boxes[ii][0].buffer(0.2), color=BLACK, ax=ax, linewidth=2)

            if show_last:
                # plot_polygon_outline(trajs_boxes[ii][-1].buffer(0.2), color=WHITE, ax=ax, linewidth=2, zorder=10)
                # plot_polygon_outline(trajs_boxes[ii][-1].buffer(0.2), color=GRAY, ax=ax, linewidth=0.5, zorder=10)
                # plot_polygon_outline(trajs_boxes[ii][-1].buffer(0.1), color=WHITE, ax=ax, linewidth=3, zorder=4)
                plot_polygon_outline(trajs_boxes[ii][-1].buffer(0.1), color=color, ax=ax, linewidth=2, zorder=4)

    plot_obstacle(obstacle=obstacle, ax=ax, padding=padding)
    set_xy_labels(ax)
    ax.set_title(title)
    ax.grid(True, linestyle=':')
    plt.axis()
    # print(f'{show_legend=}')

    if show_legend:
        if legend_inside:
            ax.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right')
        else:
            # ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            ax.legend(bbox_to_anchor=(1.0, 1.0), loc='lower right')

    ax.set_aspect(1)

    extend_axis_limits(ax=ax, padding=padding)

    if show_plot:
        plt.show()

    return ax


def set_xy_labels(ax, fontsize=16):
    # ax.set_xlabel('Location along x', fontsize=fontsize)
    # ax.set_ylabel('Location along y', fontsize=fontsize)

    ax.set_xlabel('x position', fontsize=fontsize)
    ax.set_ylabel('y position', fontsize=fontsize)


def plot_velocity(x, y, vx, vy, ax=None, color='black', arrow_width=0.1, scale=1, markersize=5):
    if ax is None:
        _, ax = get_new_fig()
        print('Creating fig, ax.')

    ax.plot(x - vx * scale, y - vy * scale, marker='o', color='white', zorder=5, markersize=markersize + 2)
    ax.plot(x - vx * scale, y - vy * scale, marker='o', color=color, zorder=5, markersize=markersize)
    ax.arrow(x - vx * scale, y - vy * scale, vx * scale, vy * scale, ls='-', color=color, zorder=5,
             width=arrow_width, head_width=arrow_width * 3,
             length_includes_head=True)
    return ax


def set_fontsizes(ax, title_fontsize=20, other_fontsize=16, legend_fontsize=None):
    """
    Set the font sizes of all elements in the given Matplotlib Axes object.

    Parameters:
    - ax: Matplotlib Axes object
    - title_fontsize: Font size for the title
    - other_fontsize: Font size for all other elements (labels, tick labels, legend, annotations)
    - legend_fontsize: Font size for legend ( set to other_fontsize by default)
    """

    if legend_fontsize is None:
        legend_fontsize = other_fontsize

    # Set title font size
    ax.title.set_fontsize(title_fontsize)

    # Set label font sizes
    ax.xaxis.label.set_size(other_fontsize)
    ax.yaxis.label.set_size(other_fontsize)

    # Set tick label font sizes
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(other_fontsize)

    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(other_fontsize)

    # Set legend font size
    if ax.get_legend():
        for text in ax.get_legend().get_texts():
            text.set_fontsize(legend_fontsize)

    # Set annotation font sizes
    for text in ax.texts:
        text.set_fontsize(other_fontsize)


def set_figure_fontsizes(obj, title_fontsize=20, other_fontsize=16):
    """
    Set the font sizes of all elements in all Axes objects in the given Matplotlib Figure or Axes.

    Parameters:
    - obj: Matplotlib Figure or Axes object
    - title_fontsize: Font size for the titles
    - other_fontsize: Font size for all other elements (labels, tick labels, legend, annotations)
    """
    if isinstance(obj, plt.Figure):
        axes = obj.get_axes()
    elif isinstance(obj, plt.Axes):
        axes = obj.figure.get_axes()  # Get all axes from the figure that contains this ax
    else:
        raise TypeError("Input must be a Matplotlib Figure or Axes object")

    for ax in axes:
        set_fontsizes(ax, title_fontsize, other_fontsize)


def plot_velocities_for_agents(x, y, vx, vy, ax=None, colour=None, arrow_width=0.45, scale=3, markersize=8,
                               padding=1,
                               mark_location=True):
    if ax is None:
        _, ax = get_new_fig()
        print('Creating fig, ax.')

    x = x.flatten()
    y = y.flatten()
    vx = vx.flatten()
    vy = vy.flatten()

    if colour is None:
        # colours = special_spectral_cmap(n_colours=len(x))
        # colours = special_cmap(n_colours=len(x))
        colours = scientific_cmap(n_colours=len(x))
        for ii in range(len(x)):
            # ax = plot_velocity(x=x[ii][0], y=y[ii][0], vx=vx[ii][0], vy=vy[ii][0], ax=ax, arrow_width=arrow_width,
            #                    markersize=markersize, scale=scale, color=adjust_lightness(colours[ii]))
            ax = plot_velocity(x=x[ii], y=y[ii], vx=vx[ii], vy=vy[ii], ax=ax, arrow_width=arrow_width,
                               markersize=markersize, scale=scale, color=adjust_lightness(colours[ii]))
    else:
        for ii in range(len(x)):
            # ax = plot_velocity(x=x[ii][0], y=y[ii][0], vx=vx[ii][0], vy=vy[ii][0], ax=ax, arrow_width=0.5,
            #                    markersize=markersize, scale=scale, color=colour)
            ax = plot_velocity(x=x[ii], y=y[ii], vx=vx[ii], vy=vy[ii], ax=ax, arrow_width=0.5,
                               markersize=markersize, scale=scale, color=colour)
    if mark_location:
        for ii in range(len(x)):
            ax.plot(x[ii], y[ii], marker='+', color='black', zorder=3, markersize=markersize)

    extend_axis_limits(ax=ax, padding=padding, reFit=True)

    return ax


def plot_accelerations_for_agents(x, y, a, theta, ax=None, colour=None, arrow_width=0.5, scale=3, markersize=9,
                                  padding=1):
    if ax is None:
        _, ax = get_new_fig()
        print('Creating fig, ax.')

    x = x.flatten()
    y = y.flatten()
    a = a.flatten()
    theta = theta.flatten()

    if colour is None:
        colours = scientific_cmap(n_colours=len(x))
        for ii in range(len(x)):
            ax = plot_acceleration(x=x[ii], y=y[ii], a=a[ii], theta=theta[ii], ax=ax,
                                   color=adjust_lightness(colours[ii]), arrow_width=arrow_width,
                                   scale=scale, markersize=markersize)
    else:
        for ii in range(len(x)):
            ax = plot_acceleration(x=x[ii], y=y[ii], a=a[ii], theta=theta[ii], ax=ax,
                                   color=colour, arrow_width=arrow_width,
                                   scale=scale, markersize=markersize)

    extend_axis_limits(ax=ax, padding=padding, reFit=True)

    return ax


def plot_acceleration(x, y, a, theta, ax=None, color='grey', arrow_width=0.05, scale=1, markersize=4):
    if ax is None:
        _, ax = get_new_fig()
        print('Creating fig, ax.')

    a_x = a * np.cos(theta)
    a_y = a * np.sin(theta)

    ax.plot(x, y, marker='o', color='white', zorder=4, markersize=markersize + 2)
    ax.plot(x, y, marker='o', color=color, zorder=4, markersize=markersize)

    ax.arrow(x, y, a_x * scale, a_y * scale, ls='-', color=color, zorder=4,
             width=arrow_width, head_width=arrow_width * 3,
             length_includes_head=True)
    return ax


def plot_trajectories(x=None, y=None, obstacle=None, ax=None, lim_threshold=10, scale_scenario_size=1):
    """Plot the trajectories of nn agents.

    Parameters
    ----------
    x : ndarray
        x coordinates of trajectories
    y : ndarray
        y coordinates of trajectories
    obstacle: shapely Polygon with all the static obstacles
    ax : Matplotlib Axes object
    lim_threshold : float
        Minimum threshold for the length of either axis

    Returns
    -------
    ax : Matplotlib Axes object
        Plot of the directed fear graph.

    """
    if x is None:
        print('No x passed!')
        return False

    if y is None:
        print('No y passed!')
        return False

    if ax is None:
        fig = plt.figure(figsize=tuple(s * 0.75 * scale_scenario_size for s in SIZE))
        ax = fig.add_subplot(121)
        # _, ax = get_new_fig()
        print('Creating fig, ax.')

    # colours = special_spectral_cmap(n_colours=len(x))
    # colours = special_cmap(n_colours=len(x))
    colours = scientific_cmap(n_colours=len(x))
    for ii, xx in enumerate(x):
        color = colours[ii]
        plt.plot(x[ii], y[ii], label=f'Trajectory {ii + 1}', color=color, marker='o')
        plt.plot(x[ii][0], y[ii][0], color='k', marker='.')
    ax.grid(True, linestyle=':')
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    ax.set_aspect('equal')
    set_xy_labels(ax, fontsize=16)

    # Making sure that the x-axis has at least some length
    xlim = ax.get_xlim()
    xlength = xlim[1] - xlim[0]
    if xlength < lim_threshold:
        new_xlim = get_newlim_with_min_thresholds(lim=xlim, threshold=lim_threshold)
        ax.set_xlim(new_xlim)

    if obstacle is not None:
        plot_obstacle(obstacle, ax=ax)

    # Making sure that the y-axis has at least some length
    ylim = ax.get_ylim()
    ylength = ylim[1] - ylim[0]
    if ylength < lim_threshold:
        new_ylim = get_newlim_with_min_thresholds(lim=ylim, threshold=lim_threshold)
        ax.set_ylim(new_ylim)

    return ax


def get_newlim_with_min_thresholds(lim, threshold=2):
    axis_length = lim[1] - lim[0]

    # Calculate the expansion needed on both sides
    expansion_needed = max(0, threshold - axis_length) / 2

    # Adjust the limits symmetrically
    new_lim = (
        lim[0] - expansion_needed,
        lim[1] + expansion_needed
    )
    return new_lim


def plot_squares(squares, color=BLUE, ax=None, title='Squares Multipolygon'):
    if ax is None:
        fig, ax = plt.subplots()
        print('Creating fig, ax.')

    for square in squares:
        plot_polygon(square, ax=ax, add_points=False, color=color)

    # ax.set_xlim(-12, 12)
    # ax.set_ylim(-12, 12)
    ax.set_aspect('equal', 'box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)


def plot_polygon_outline(polygon, ax=None, color=BLUE, linewidth=1, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        print('Creating fig, ax.')

    patch = patch_from_polygon(
        polygon, facecolor=color, edgecolor=color, linewidth=linewidth, fill=False, **kwargs
    )
    ax.add_patch(patch)


def plot_intersecting_polygons_one_by_one(intersection, ii, not_ii, other_squares_fixed=None, square_no=0, ax=None):
    if intersection:
        color = RED
    else:
        color = BLUE

    if ax is None:
        fig, ax = plt.subplots()
    plot_squares(not_ii, ax=ax, color=YELLOW)
    plot_squares([ii], color=color, ax=ax, title=f'Intersections for Polygon ii= {square_no}')

    if other_squares_fixed is not None:
        #         fig, ax = plt.subplots()
        plot_squares([other_squares_fixed], color=DARKGRAY, ax=ax, title=f'Fixed Multi_polygon= {square_no}')

    strip_axes(ax=ax, strip_title=False)
    make_axes_gray(ax=ax)

    return ax


def plot_intersecting_polygons(intersections, polygons, ax=None, title=''):
    if ax is None:
        fig, ax = plt.subplots()

    for intersection, polygon in zip(intersections, polygons):
        if intersection:
            color = RED
        else:
            color = BLUE

        plot_squares([polygon], color=color, ax=ax, title=title)

    strip_axes(ax=ax, strip_title=False)
    make_axes_gray(ax=ax)

    return ax


def plot_directed_fear_graph(fear, x0, y0, fear_threshold=0.05, ax=None,
                             hide_cbar=False, horizontal_cbar=False, normalise_cmap=False,
                             arrowsize=20, arrow_width=3,
                             zorder_lift=0, game_mode=False):
    """Plot a directed graph with edges colored and sized based on the fear values.

    Parameters
    ----------
    game_mode: bool
        Whether to plot for the Game of FeAR
    zorder_lift: int
        Layers to lift the plot by
    hide_cbar: bool
        Whether to hide to colorbar
    horizontal_cbar : bool
        If True, the colorbar will have horizontal orientation. Else, it will be vertical.
    normalise_cmap: bool
        Whether to normalise the cmap to [-1,1]
    fear : ndarray
        Array containing Feasible Action-Space Reduction values
    x0 : ndarray
        x coordinate of the starting locations of agents
    y0 : ndarray
        y coordinate of the starting locations of agents
    fear_threshold : float, optional
        If the absolute value of fear is below the fear_threshold, then these are not plotted
    arrowsize: int
        size of the arrows of the directed graph
    arrow_width: int
        width of the arrows of the directed graph

    ax : Matplotlib Axes object, optional

    Returns
    -------
    ax : Matplotlib Axes object
        Plot of the directed fear graph.
    """

    if game_mode:
        hide_cbar = True
        zorder_lift = 10

    x0 = x0.squeeze()
    y0 = y0.squeeze()

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if normalise_cmap:
        fear_min = -1
        fear_max = 1
        print(f'{fear_min=},{fear_max=}')

    else:
        non_diag = ~np.eye(fear.shape[0], dtype=bool)  # Create a boolean mask for off-diagonal elements
        fear_max = np.max(np.abs(fear[non_diag]))
        fear_min = - fear_max
        print(f'{fear_min=},{fear_max=}')

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    n_agents = len(x0)
    for i in range(n_agents):
        G.add_node(i, pos=(x0[i], y0[i]))

    # Add edges to the graph based on fear values
    for actor in range(n_agents):
        for affected in range(n_agents):
            if actor != affected:
                fear_value = fear[actor, affected]
                if abs(fear_value) > fear_threshold:
                    # Only add edge if fear values with abs() greater than a threshold
                    G.add_edge(actor, affected, weight=fear_value)

    # Get positions of nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Create colormap
    cmap = sns.diverging_palette(220, 20, as_cmap=True, sep=1)

    # Draw node labels as i + 1
    labels = {i: str(i + 1) for i in G.nodes()}

    # Draw the graph
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=300, node_color='moccasin', labels=labels,
            edge_color=weights, width=2, ax=ax, alpha=0.8,
            edge_cmap=cmap, edge_vmin=fear_min, edge_vmax=fear_max,
            horizontalalignment='center', verticalalignment='center_baseline',
            connectionstyle='arc3, rad = 0.1')

    # nodes_drawn = nx.draw_networkx_nodes(G, pos, node_size=300, node_color='moccasin', ax=ax, alpha=0.75)

    #     nx.draw_networkx_labels(G, pos, labels=labels, alpha=0.75, ax=ax)
    edges_drawn = nx.draw_networkx_edges(G, pos, edge_color=weights, width=arrow_width, arrowsize=arrowsize,
                                         ax=ax,
                                         edge_cmap=cmap, edge_vmin=fear_min, edge_vmax=fear_max,
                                         connectionstyle='arc3, rad = 0.1')

    #     nx.draw_networkx_labels(G, pos, labels=labels)

    if zorder_lift > 0:
        nodes_drawn = nx.draw_networkx_nodes(G, pos, node_size=300, node_color='moccasin',
                                             ax=ax, alpha=0.75)
        labels_drawn = nx.draw_networkx_labels(G, pos, labels=labels,
                                               horizontalalignment='center', verticalalignment='center_baseline')

        nodes_drawn.set_zorder(zorder_lift + 2)

        for _, label_drawn in labels_drawn.items():
            label_drawn.set_zorder(zorder_lift + 4)

        for edge in edges_drawn:
            edge.set_zorder(zorder_lift)

    # Add colorbar for edge colors
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=fear_min, vmax=fear_max))
    sm.set_array([])
    if not hide_cbar:
        divider = make_axes_locatable(ax)
        if horizontal_cbar:
            cax = divider.append_axes('bottom', size='5%', pad=1)
            cbar = fig.colorbar(sm, orientation='horizontal', cax=cax, shrink=0.7)
            cbar.ax.axvline(x=fear_threshold, ymin=0.1, ymax=0.9, c='grey')
            cbar.ax.axvline(x=-fear_threshold, ymin=0.1, ymax=0.9, c='grey')
            cbar.ax.axvspan(xmin=-fear_threshold, xmax=fear_threshold, ymin=0, ymax=1, color='w')
        else:  # Vertical Colour Bar
            # cax = divider.append_axes('right', size='5%', pad=1)
            # cbar = fig.colorbar(sm, orientation='vertical', cax=cax, shrink=0.7)
            cbar = fig.colorbar(sm, orientation='vertical', shrink=0.7)
            cbar.ax.axhline(y=fear_threshold, xmin=0.1, xmax=0.9, c='grey')
            cbar.ax.axhline(y=-fear_threshold, xmin=0.1, xmax=0.9, c='grey')
            cbar.ax.axhspan(ymin=-fear_threshold, ymax=fear_threshold, xmin=0, xmax=1, color='w')

        # --------------------

        ticks = [fear_min, -fear_threshold, fear_threshold, fear_max]  # Set the ticks manually

        # Set the font size of the colorbar ticks
        cbar.ax.tick_params(labelsize=16)

        if not horizontal_cbar:
            tweak_cbar_ticklabels(cbar=cbar, ticks=ticks, fontsize=16)

        # --------------------

        cbar.outline.set_visible(False)  # Remove the black outline
        make_axes_gray(cbar.ax)
        cbar.set_label('Fear Value', fontsize=16)

    ax.set_aspect(1)

    if not game_mode:
        ax.set_title('Directed Graph with Fear Values')

    return ax


def tweak_cbar_ticklabels(cbar, ticks=None, scale=0.7, v_offset=0.1, scale_width=1, fontsize=None):
    if ticks is None:
        # Get current ticks and labels
        ticks = cbar.get_ticks()

    tick_labels = [f'{t:.2g}' for t in ticks]  # Convert the original ticks to string labels

    # Customize -1 and +1 labels
    # tick_labels = ['-1    Courteous' if t == -1 else '+1   Assertive' if t == 1 else label for t, label in
    #                zip(ticks, tick_labels)]

    # cbar.shrink(0.7)
    rescale_cbar(cbar=cbar, scale=scale, scale_width=scale_width)

    if fontsize is None:
        cbar.ax.text(0.5, 0 - v_offset, 'Courteous', ha='left', va='center', transform=cbar.ax.transAxes,
                     color='dimgray')
        cbar.ax.text(0.5, 1 + v_offset, 'Assertive', ha='left', va='center', transform=cbar.ax.transAxes,
                     color='dimgray')
    else:
        cbar.ax.text(0.5, 0 - v_offset, 'Courteous', ha='left', va='center', transform=cbar.ax.transAxes,
                     color='dimgray', fontsize=fontsize)
        cbar.ax.text(0.5, 1 + v_offset, 'Assertive', ha='left', va='center', transform=cbar.ax.transAxes,
                     color='dimgray', fontsize=fontsize)

    # Set the tick locations and apply custom labels
    cbar.ax.set_yticks(ticks)  # Explicitly set tick locations
    cbar.ax.set_yticklabels(tick_labels)  # Apply custom tick labels


def plot_fear(fear=None, for_print=True, cbar=None, agent_coloured_ticks=True,
              fmt='.1f',
              swap_colours_2_3=True):
    """
    Function to plot the fear values - where the feal values are included as the diagonal elements

    Parameters
    ----------
    fmt
    swap_colours_2_3
    agent_coloured_ticks
    cbar
    for_print
    fear : ndarray

    Returns
    -------
    ax : Matplotlib axis
    """

    if fear is None:
        print('No fear passed!')
        return False

    nn, _ = fear.shape

    if for_print:
        if nn <= 2:
            finer = False
        elif 2 < nn <= 6:
            finer = True
        else:
            finer = False
            for_print = False

    ax = PlotGWorld.plotResponsibility(fear, FeAL=np.diagonal(fear), for_print=for_print, finer=finer, cbar=cbar,
                                       title='', annot_font_size=12, fmt=fmt)

    cbar = ax.collections[0].colorbar  # Get the colorbar associated with the heatmap

    if cbar:  # Check if there is a cbar
        # tweak_cbar_ticklabels(cbar=cbar)

        nn, _ = fear.shape
        if nn <= 4:
            scale = 0.6
            scale_width = 3
        else:
            scale = 0.7
            scale_width = 1

        if for_print:
            v_offset = 0.3
        else:
            v_offset = 0.1

        tweak_cbar_ticklabels(cbar=cbar, scale=scale, scale_width=scale_width, v_offset=v_offset)

    make_all_axes_gray(ax)

    if agent_coloured_ticks:
        nn, _ = fear.shape

        if nn <= 1:
            tick_pad = 5
        else:
            tick_pad = 5

        ax.tick_params(axis='x', length=0, pad=tick_pad)
        ax.tick_params(axis='y', length=0, pad=tick_pad)

        # Set the colormap for the ticks (Batlow colormap from cmcrameri)
        # batlow_cmap = cmc.batlow
        batlow_cmap = cmc.batlowK

        norm = mcolors.Normalize(vmin=0, vmax=nn - 1)  # Normalize values between 0 and n-1

        # Customize tick label and tick colors
        for tick, label in enumerate(ax.get_xticklabels()):
            if nn <= 1:
                label.set_color(batlow_cmap(norm(tick)))  # Label color
            else:
                if nn == 3 and swap_colours_2_3:
                    swap_2_3 = True
                else:
                    swap_2_3 = False
                add_agent_coloured_bg_to_tick(batlow_cmap, label, norm, tick, swap_2_3=swap_2_3)
                # ax.tick_params(pad=5)
                ax.xaxis.labelpad = 5 + 3

            # label.set_ha('center')  # Options: 'center', 'left', 'right'
            # label.set_va('top')  # Options: 'center', 'top', 'bottom', 'baseline'

        for tick, label in enumerate(ax.get_yticklabels()):
            if nn <= 1:
                label.set_color(batlow_cmap(norm(tick)))  # Label color
            else:
                if nn == 3 and swap_colours_2_3:
                    swap_2_3 = True
                else:
                    swap_2_3 = False
                add_agent_coloured_bg_to_tick(batlow_cmap, label, norm, tick, swap_2_3=swap_2_3)
                # ax.tick_params(pad=5)
                ax.yaxis.labelpad = 5

            # label.set_ha('right')  # Options: 'center', 'left', 'right'
            # label.set_va('baseline')  # Options: 'center', 'top', 'bottom', 'baseline'

    return ax


def add_agent_coloured_bg_to_tick(batlow_cmap, label, norm, tick, swap_2_3=False):
    if swap_2_3 and tick == 1:
        tick_ = 2
    elif swap_2_3 and tick == 2:
        tick_ = 1
    else:
        tick_ = tick

    color = batlow_cmap(norm(tick_))  # Background color from Batlow colormap
    color = (color[0], color[1], color[2], color[3] * 0.6)  # Reducing the alpha to 60%
    label.set_backgroundcolor(color)  # Set background color
    label.set_color(get_contrasting_text_color(color[:3]))  # Adjust text color (black/white)


def plot_fear_graph_on_trajs(fear=None, trajs_hulls=None, trajs_boxes=None, obstacle=None,
                             x0=None, y0=None, fear_threshold=None,
                             fear_threshold_percentile=None, ax=None):
    if ax is None:
        _, ax = get_new_fig(dpi=300)

    if fear_threshold is None:

        if fear_threshold_percentile is None:
            N, _ = trajs_boxes.shape
            # Dynamically setting the threshold for the fear values based on the number of agents
            fear_threshold_percentile = 0 + (N - 2) / N * (100 - 1)  # so that when as nn->inf, this -> 99

        fear_threshold = np.percentile(np.absolute(fear), fear_threshold_percentile)

    ax = plot_hulls_for_trajs(trajs_hulls, trajs_boxes=trajs_boxes, obstacle=obstacle,
                              colour='lightgrey', ax=ax)
    plot_directed_fear_graph(fear=fear, x0=x0, y0=y0, fear_threshold=fear_threshold, ax=ax, zorder_lift=10)

    return ax


def plot_fears(fears=None, size=SIZE, scatter=False):
    """
    Function to plot the FeAR values for a sequence of actions.

    Parameters
    ----------
    fears : ndarray
        Numpy array with the fears for the sequence. Array of shape (num_windows, n_agents, nn agents)
    size : tuple
        size of the figure
    scatter : boolean
        Plots the scatter plot if scatter==True, else plots the area plot

    Returns
    -------
    Matplotlib axes
    """

    if fears is None:
        print('No FeARs passed!')

    num_windows, n, _ = fears.shape
    max_fear = np.abs(fears).max()

    fig, axes = plt.subplots(n, n, figsize=size)

    cmap_fear = sns.diverging_palette(220, 20, as_cmap=True)
    cmap_FeAR_normalize = mcolors.Normalize(vmin=-max_fear, vmax=max_fear)
    cmap_fear_scalar_mappable = plt.cm.ScalarMappable(cmap=cmap_fear, norm=cmap_FeAR_normalize)

    # Loop through each agent
    for ii in range(n):
        for jj in range(n):
            # Plot fear values of agent ii on agent jj

            plot_fears_for_ij(fears=fears, ii=ii, jj=jj, ax=axes[n - 1 - jj, ii],
                              cmap_fear_scalar_mappable=cmap_fear_scalar_mappable, num_windows=num_windows,
                              scatter=scatter, print_xy_labels=False)
            # axes[ii, jj].set_ylim([-1.1, 1.1])
            axes[jj, ii].set_ylim([-max_fear - .1, max_fear + .1])

            if jj == 0:
                axes[ii, jj].set_ylabel('FeAR')
            else:
                axes[ii, jj].set_yticks([])

            if n - 1 - ii == 0:
                axes[ii, jj].set_xlabel('Time window')
            else:
                axes[ii, jj].set_xticks([])

    #         axes[ii, jj].legend()

    # Adjust layout and show plot
    fig.tight_layout()

    return axes[0][0]


def plot_fears_for_ij(fears, ii=None, jj=None, ax=None, cmap_fear_scalar_mappable=None, num_windows=None,
                      scatter=False, print_xy_labels=True, size=(10, 6)):
    """

    Parameters
    ----------
    fears : ndarray
        numpy of fears of shape (num_windows, n_agents, n_agents)
    ii : int
        actor agent id
    jj : int
        affected agent id
    ax : matplotlib axis
        figure axis on which to plot. If ax is None, a new figure is created.
    cmap_fear_scalar_mappable: matplotlib cmap object
        cmap for the colours of the fears plot
    num_windows : int
        number of windows of the sequential fears
    scatter : boolean
        plots scatter plot for fears if True
    print_xy_labels: boolean
        prints x and y labels for the ax if True
    size: tuple
        size of the figure to be used if a new figure is created

    Returns
    -------
    ax : matplotlib axis
    """

    if ax is None:
        _, ax = plt.subplots(figsize=size)
        print('Creating fig, ax.')

    if ii is None:
        print('Actor agent id not passed!')
        return False

    if jj is None:
        print('Affected agent id not passed!')
        return False

    if cmap_fear_scalar_mappable is None:
        cmap_fear = sns.diverging_palette(220, 20, as_cmap=True)
        cmap_FeAR_normalize = mcolors.Normalize(vmin=-1, vmax=1)  # Stick to [-1, 1] for the limits
        cmap_fear_scalar_mappable = plt.cm.ScalarMappable(cmap=cmap_fear, norm=cmap_FeAR_normalize)

    if num_windows is None:
        num_windows, _, _ = fears.shape

    if scatter:
        for kk in range(num_windows):
            # Plot fear value at time step k for agent j on agent i with colormap
            s = abs(fears[kk, ii, jj]) * 100 + 4
            ax.scatter(kk, fears[kk, ii, jj], s=s,
                       color=cmap_fear_scalar_mappable.to_rgba(fears[kk, ii, jj]))
    else:
        y1 = fears[:, ii, jj]  # The FeAR time series
        x = np.arange(len(y1))
        y2 = np.zeros(len(y1))

        # axes[ii, jj].plot(x, y1, color='white', linewidth=3)
        ax.plot(x, y1, color='grey', linewidth=0.5)

        ax.fill_between(x, y1, y2, where=(y1 > y2), color='#c3553a', alpha=0.7, interpolate=True)
        ax.fill_between(x, y1, y2, where=(y1 < y2), color='#407e9c', alpha=0.7, interpolate=True)

    # Set title for each subplot
    ax.set_title(f'FeAR$_{{({ii + 1},{jj + 1})}}$')
    ax.set_ylim([-1.1, 1.1])

    if ii == jj:
        ax.set_frame_on(True)

    else:
        ax.set_frame_on(False)

    if print_xy_labels:
        ax.set_ylabel('FeAR')
        ax.set_xlabel('Time window')

    return ax


def plot_hyper_fears_for_ij(fears, hyperparameter=None, hyperparameter_values=None,
                            ii=None, jj=None, ax=None,
                            print_x_labels=True,
                            print_y_labels=True, size=(10, 6)):
    """

    Parameters
    ----------
    fears : ndarray
        numpy of fears of shape (num_windows, n_agents, n_agents)
    hyperparameter: str
        name of the hyperparameter that is varied
    hyperparameter_values: ndarray
        values of the hyperparameter
    ii : int
        actor agent id
    jj : int
        affected agent id
    ax : matplotlib axis
        figure axis on which to plot. If ax is None, a new figure is created.
    print_x_labels: boolean
        prints x labels for the ax if True
    print_y_labels: boolean
        prints y labels for the ax if True
    size: tuple
        size of the figure to be used if a new figure is created

    Returns
    -------
    ax : matplotlib axis
    """

    if hyperparameter is None:
        hyperparameter = ''

    if ax is None:
        _, ax = plt.subplots(figsize=size)
        print('Creating fig, ax.')

    if ii is None:
        print('Actor agent id not passed!')
        return False

    if jj is None:
        print('Affected agent id not passed!')
        return False

    # if cmap_fear_scalar_mappable is None:
    #     cmap_fear = sns.diverging_palette(220, 20, as_cmap=True)
    #     cmap_FeAR_normalize = mcolors.Normalize(vmin=-1, vmax=1)  # Stick to [-1, 1] for the limits
    #     cmap_fear_scalar_mappable = plt.cm.ScalarMappable(cmap=cmap_fear, norm=cmap_FeAR_normalize)

    num_hyper, _, _ = fears.shape

    if hyperparameter_values is None:
        hyperparameter_values = range(1, num_hyper + 1)
    elif hyperparameter == 'compute_mdrs':
        # Just print the MdR number instead of printing the complete MdR dict
        hyperparameter_values = range(1, num_hyper + 1)
        hyperparameter = 'MdR'

    y1 = fears[:, ii, jj]  # The FeAR time series
    # if len(y1) == 1:
    #     y1 = y1[0] * np.ones((1, 3))
    #     hyperparameter_values = ['', hyperparameter_values[0], '']

    x = np.arange(len(y1))
    y2 = np.zeros(len(y1))

    # axes[ii, jj].plot(x, y1, color='white', linewidth=3)
    ax.plot(x, y1, color='grey', linewidth=0.5)

    ax.fill_between(x, y1, y2, where=(y1 > y2), color='#c3553a', alpha=0.7, interpolate=True)
    ax.fill_between(x, y1, y2, where=(y1 < y2), color='#407e9c', alpha=0.7, interpolate=True)

    # Set title for each subplot
    ax.set_title(f'FeAR$_{{({ii + 1},{jj + 1})}}$')
    ax.set_ylim([-1.1, 1.1])

    # Set the x-tick labels to be the hyperparameter values
    ax.set_xticks(x)
    # ax.set_xticklabels(hyperparameter_values)

    # Automatically determine a reasonable number of ticks
    # locator = MaxNLocator(integer=True, min_n_ticks=1, symmetric=False, prune='both')
    locator = MaxNLocator(integer=True, nbins='auto', min_n_ticks=min(3, len(hyperparameter_values)), symmetric=False,
                          prune='both')
    ax.xaxis.set_major_locator(locator)

    # Get the positions for the ticks
    ticks = ax.get_xticks()

    # print(f'{len(ticks)=}, {len(hyperparameter_values)=}')

    if len(ticks) < len(hyperparameter_values):
        # print('!!! len(ticks) < len(hyperparameter_values) !!!')
        # print(f'{ticks=}')

        # Set the tick labels to the corresponding hyperparameter values
        tick_labels = [hyperparameter_values[int(tick)] if int(tick) < len(hyperparameter_values) else '' for tick in
                       ticks]
        # print(f'{tick_labels=}')

        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=45)
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(hyperparameter_values, rotation=45)

    if ii == jj:
        ax.set_frame_on(True)
    else:
        ax.set_frame_on(False)

    if print_y_labels:
        ax.set_ylabel('FeAR')

    if print_x_labels:
        ax.set_xlabel(hyperparameter)

    return ax


def plot_hyper_fears(results=None, size=None):
    """
    Function to plot the FeAR values from a hyper_fear result.

    Parameters
    ----------
    results : dict
        Dictionary with the hyper_fear results
    size : tuple
        size of the figure

    Returns
    -------
    Matplotlib axes
    """

    if results is None:
        print('No results passed!')

    fears = results['fear_values']
    hyperparameter = results['hyperparameter']
    hyperparameter_values = results['values']

    num_hyper, n, _ = fears.shape
    max_fear = np.abs(fears).max()

    if num_hyper == 1:  # Repeat values for singleton result
        fears = np.append(fears, [fears[0]], axis=0)
        hyperparameter_values = np.append(hyperparameter_values, hyperparameter_values[0])

    if size is None:
        if n < 5:
            size = SIZE
        else:
            size = (20, 24)

    fig, axes = plt.subplots(n, n, figsize=size)

    cmap_fear = sns.diverging_palette(220, 20, as_cmap=True)
    cmap_FeAR_normalize = mcolors.Normalize(vmin=-max_fear, vmax=max_fear)
    cmap_fear_scalar_mappable = plt.cm.ScalarMappable(cmap=cmap_fear, norm=cmap_FeAR_normalize)

    # Loop through each agent
    for ii in range(n):
        for jj in range(n):

            if ii == 0:
                print_y_labels = True
            else:
                print_y_labels = False

            if jj == 0:
                print_x_labels = True
            else:
                print_x_labels = False

            # Plot fear values of agent ii on agent jj

            plot_hyper_fears_for_ij(fears=fears,
                                    hyperparameter=hyperparameter, hyperparameter_values=hyperparameter_values,
                                    ii=ii, jj=jj, ax=axes[n - 1 - jj, ii],
                                    print_y_labels=print_y_labels, print_x_labels=print_x_labels
                                    )
            # axes[ii, jj].set_ylim([-1.1, 1.1])
            # axes[n - 1 - jj, ii].set_ylim([-max_fear - .1, max_fear + .1])

    # Adjust layout and show plot
    fig.tight_layout()

    return axes[0][0]


def plot_fear_barplot(fear, indices_to_plot=None, ax=None, tweak_ticks=False):
    # If no indices are provided, use all indices of the matrix
    if indices_to_plot is None:
        indices_to_plot = list(np.ndindex(fear.shape))

    # Extract the values to plot from the matrix
    fears_to_plot = [fear[i, j] for i, j in indices_to_plot]

    # Generate labels for each bar from the indices
    # labels = [f"FeAR$_{{{i},{j}}}$" for i, j in indices_to_plot]
    labels = [f"({i + 1},{j + 1})" for i, j in indices_to_plot]

    # Set up the figure
    if ax is None:
        fig, ax = plt.subplots(figsize=[2.5, 5])
        print('Creating fig, ax.')

    # plt.rc('text', usetex=True, ax=ax)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=-1, vmax=1))
    c_del = 0.3

    # Create the bar plot
    bars = ax.bar(
        np.arange(len(fears_to_plot)),
        fears_to_plot,
        color=[RED if val > 0 else BLUE for val in fears_to_plot],
        edgecolor=[sm.to_rgba(1 - c_del) if val > 0 else sm.to_rgba(-c_del) if val < 0 else LIGHTGRAY for val in
                   fears_to_plot],  # Edge color based on value
        hatch=['//' if val > 0 else '..' for val in fears_to_plot],  # Hatch based on value
    )

    # Set y-limits
    ax.set_ylim([-1, 1])

    ticks = [-1, -0.5, 0, 0.5, 1]
    tick_labels = [f'{t:g}' for t in ticks]  # Convert the original ticks to string labels

    if tweak_ticks:
        # Customize first and last labels
        tick_labels[0] = f'Courteous   {ticks[0]:g}'
        tick_labels[-1] = f'Assertive   {ticks[-1]:g}'

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)

    # Remove background and grid
    ax.set_facecolor('none')
    ax.grid(False)

    # Move the x-axis (bottom spine) to the y=0 line
    ax.spines['bottom'].set_position(('data', 0))

    # # Set zorder for spines to be on top of the bars
    # ax.spines['bottom'].set_zorder(10)  # Bring the bottom spine to the top
    # ax.spines['bottom'].set_color(GRAY)  # Ensure it's gray to match the style

    # Customize x-axis: Add indices as xticklabels and rotate them by 135 degrees
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45)

    # Remove the top, right, and left frame (spines)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(direction='inout')

    # Adjust the position and direction of xticks (labels and ticks) based on bar values
    for i, (tick, label) in enumerate(zip(ax.get_xticks(), ax.get_xticklabels())):
        if fears_to_plot[i] >= 0:
            # For positive values: move the label below and ticks point downwards
            label.set_va('top')  # Label below the axis
            label.set_position((tick, -0.0))  # Slightly lower the position

        else:
            # For negative values: move the label above and ticks point upwards
            label.set_va('bottom')  # Label above the axis
            label.set_position((tick, 0.1))  # Move the position further up

    # ax.set_xlabel('$ FeAR_{i,j}$', fontsize=12)
    # ax.set_title('FeAR$_{i,j}$', fontsize=12)

    set_fontsizes(ax)
    make_axes_gray(ax)

    return ax


def plot_mdrs_ii(axes=None, ii=None, mdrs_a=None, mdrs_theta=None, a=None, theta=None, size=(10, 6)):
    if ii is None:
        print('Actor agent id not passed!')
        return False

    if a is None:
        print('No magnitudes of acceleration passed!')
        return False
    if theta is None:
        print('No directions of acceleration passed!')
        return False

    if mdrs_a is None:
        print('No mdrs_a passed!')
        return False
    if mdrs_theta is None:
        print('No mdrs_theta passed!')
        return False

    if axes is None:
        fig, axes = plt.subplots(2, figsize=size)
        print('Creating fig, ax.')

    _, num_windows = mdrs_a.shape
    x = np.arange(num_windows)

    axes[0].plot(x, a[ii, :], color='gold', linewidth=2, label='Actual a')
    axes[0].plot(x, mdrs_a[ii, :], color='gold', linestyle=':', linewidth=2, label='MdR a')
    axes[0].set_title(f'Magnitude of acceleration of agent {ii}')
    make_axes_gray(ax=axes[0])
    axes[0].grid(True, linestyle=':')
    axes[0].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    axes[0].set_xlabel('Timestep', fontsize=12)
    # axes[0].set_ylabel('Magnitude of Acceleration', fontsize=12)

    axes[1].plot(x, theta[ii, :] * 360 / np.pi, color='tab:blue', linewidth=2, label='Actual theta')
    axes[1].plot(x, mdrs_theta[ii, :] * 360 / np.pi, color='tab:blue', linestyle=':', linewidth=2, label='MdR theta')
    axes[1].set_title(f'Direction of acceleration of agent {ii}')
    axes[1].grid(True, linestyle=':')
    axes[1].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    axes[1].set_xlabel('Timestep', fontsize=12)
    # axes[1].set_ylabel('Direction of Acceleration', fontsize=12)
    make_axes_gray(ax=axes[1])

    # Adjust layout
    fig.tight_layout()

    return axes[0]


def plot_obstacle(obstacle, ax=None, colour='grey', keep_ax_lims=True, padding=12):
    """
    Plot static obstacles

    Parameters
    ----------
    obstacle : MultiPolygon or Polygon
    ax : Matplotlib ax
    keep_ax_lims : bool

    Returns
    -------
    ax of plot of static obstacles
    """

    if obstacle is None:
        return ax

    if ax is None:
        _, ax = get_new_fig()
        print('Creating fig, ax.')
        keep_ax_lims = False  # Making this False if there is no ax_lims to keep
    elif keep_ax_lims:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    plot_polygon(obstacle, ax=ax, add_points=False, linewidth=0.5, color=colour)

    if keep_ax_lims:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        extend_axis_limits(ax=ax, padding=padding, reFit=False)

    return ax


def plot_fear_grid_search_for_ij(fear_by_actor=None, actor_as=None, actor_thetas=None, ax=None, recalibrate_cmap=False,
                                 peak=None, verbose_flag=False, cbar=False, return_pcolormesh=False, cbar_layout=False,
                                 collisions_4_actor=None, plot_collisions=True, ylabel=False,
                                 title='', fine_hlines=False, plot_mins=False):
    if fear_by_actor is None:
        print('fear_by_actor not passed !')
        return False
    if actor_as is None:
        print('actor_as not passed !')
        return False
    if actor_thetas is None:
        print('actor_thetas not passed !')
        return False

    if ax is None:
        _, ax = get_new_fig(figsize=(10, 6))
        print('Creating fig, ax.')

    if peak is not None:
        vmax = peak
        vmin = -peak

    elif recalibrate_cmap:
        abs_max = np.abs(np.max(fear_by_actor))
        abs_min = np.abs(np.min(fear_by_actor))

        peak = max(abs_max, abs_min)

        vmax = peak
        vmin = -peak

    else:
        vmax = 1
        vmin = -1

    if verbose_flag:
        print(f'{title} = \n{np.array2string(fear_by_actor, separator=", ", max_line_width=np.inf)}')

    cmap = sns.diverging_palette(220, 20, as_cmap=True, sep=1)

    pcolormesh = ax.pcolormesh(actor_as, actor_thetas, fear_by_actor.T, shading='nearest',
                               vmin=vmin, vmax=vmax, cmap=cmap)

    if cbar:
        fig = ax.get_figure()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.3)

        cbar = fig.colorbar(pcolormesh, cax=cax, orientation='vertical', shrink=0.7)
        cbar.outline.set_visible(False)  # Remove the black outline

        # --------------
        # ticks = cbar.get_ticks()
        ticks = [vmin, 0, vmax]
        tick_labels = [f'{t:g}' for t in ticks]  # Convert the original ticks to string labels

        # Customize first and last labels
        # tick_labels[0] = f'{ticks[0]:g}   Courteous'
        # tick_labels[-1] = f'{ticks[-1]:g}   Assertive'

        # Set the tick locations and apply custom labels
        cbar.ax.set_yticks(ticks)  # Explicitly set tick locations
        cbar.ax.set_yticklabels(tick_labels)  # Apply custom tick labels

        # rescale_cbar(cbar=cbar, scale=0.7)

        cbar.ax.text(0.5, -0.1, 'Courteous', ha='left', va='center', transform=cbar.ax.transAxes, color='dimgray')
        cbar.ax.text(0.5, 1.1, 'Assertive', ha='left', va='center', transform=cbar.ax.transAxes, color='dimgray')

        # --------------

        make_axes_gray(cbar.ax)

    elif cbar_layout:
        # Add and remove cbar to help with the layout
        fig = ax.get_figure()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.3)

        cbar = fig.colorbar(pcolormesh, cax=cax, orientation='vertical', shrink=0.7)
        cbar.remove()

    if fine_hlines:
        hline_width = 0.5
    else:
        hline_width = 1

    hline_colour = 'darkgray'
    ax.axhline(y=0, color=hline_colour, linestyle=':', linewidth=hline_width)
    ax.axhline(y=np.pi / 2, color=hline_colour, linestyle=':', linewidth=hline_width)
    ax.axhline(y=-np.pi / 2, color=hline_colour, linestyle=':', linewidth=hline_width)

    if plot_mins:
        min_actors_a, min_actors_theta, min_value = cfear.minimise_fear(actor_as, actor_thetas, fear_by_actor,
                                                                        collisions_4_actor)

        # Plotting the minimum value point
        # ax.plot(min_actors_a, min_actors_theta, 'o', color='dimgray', markersize=hline_width * 1.5)
        # ax.plot(min_actors_a, min_actors_theta, 'o', color='white', markersize=hline_width * 0.5)

        ax.plot(min_actors_a, min_actors_theta, 'o', color='dimgray', markersize=hline_width * 4)
        ax.plot(min_actors_a, min_actors_theta, 'o', color='white', markersize=hline_width * 2)

        title = title + f'\nmin = {min_value:.2f}'

    if plot_collisions and (collisions_4_actor is not None):
        collision_a, collision_theta = cfear.find_actions_with_collisions(actor_as,
                                                                          actor_thetas,
                                                                          collisions_4_actor)
        # Marking the actions with collisions
        ax.plot(collision_a, collision_theta, 'x', color=RED, markersize=hline_width * 4, linewidth=0.5)

    ax.set_xlabel('$a$', fontsize=12)
    if ylabel: ax.set_ylabel('$\\theta$', fontsize=12)
    ax.set_title(title, fontsize=12)

    if return_pcolormesh:
        return ax, pcolormesh


def plot_fear_grid_search(fear_grid_search, verbose_flag=False, save_figs=False, results_file_name=''):
    for actor_string in fear_grid_search.keys():
        actor = int(actor_string)
        fears_by_actor = fear_grid_search[actor_string]['fears_by_actor']
        actor_as = fear_grid_search[actor_string]['actor_as']
        actor_thetas = fear_grid_search[actor_string]['actor_thetas']
        if 'collisions_4_actor' in fear_grid_search[actor_string].keys():
            collisions_4_actor = fear_grid_search[actor_string]['collisions_4_actor']
        else:
            collisions_4_actor = None

        if verbose_flag:
            print('\n--------------------------------------------------')
            print(f'      Grid Search - Fear by Actor {actor_string + 1}')
            print(f'{fears_by_actor=}')
            print(f'{actor_as=}')
            print(f'{actor_thetas=}')

        _, _, N = fears_by_actor.shape
        others = cfear.get_other_agents(n=N, ii=actor)

        fig, axes = plt.subplots(1, N - 1, figsize=(10, 6 * (N - 1)), dpi=200)
        for ax in axes:
            ax.grid(True, linestyle=':')
            ax.set_aspect("equal")
            make_axes_gray(ax)
        # fig.tight_layout()

        for jj, affected in enumerate(others):
            if verbose_flag:
                print(f'{affected=}')
            ax = axes[jj]
            fear_by_actor = fears_by_actor[:, :, affected]

            if jj != 0:
                ylabel = False
            else:
                ylabel = True

            plot_fear_grid_search_for_ij(fear_by_actor=fear_by_actor, actor_as=actor_as, actor_thetas=actor_thetas,
                                         collisions_4_actor=None, ylabel=ylabel,
                                         title=f'Affected = {affected + 1}', ax=ax, fine_hlines=True)
            if jj != 0:
                ax.set_ylabel('')
                ax.set_yticks([])

        fear_grid_affected_ax = axes[0]

        # --------------------- Aggregating FeAR values ----------------------------- #

        fear_on_others = fears_by_actor[:, :, others]

        fig, axes = plt.subplots(1, 3, figsize=(10, 6 * 3), dpi=200)

        for ax in axes:
            ax.grid(True, linestyle=':')
            ax.set_aspect("equal")
            make_axes_gray(ax)

        mean_fear_by_actor = np.mean(fear_on_others, axis=2)
        plot_fear_grid_search_for_ij(fear_by_actor=mean_fear_by_actor, actor_as=actor_as, actor_thetas=actor_thetas,
                                     collisions_4_actor=collisions_4_actor, cbar_layout=True, ylabel=True,
                                     title=f'Mean FeAR', ax=axes[0], plot_mins=True)
        min_fear_by_actor = np.min(fear_on_others, axis=2)
        plot_fear_grid_search_for_ij(fear_by_actor=min_fear_by_actor, actor_as=actor_as, actor_thetas=actor_thetas,
                                     collisions_4_actor=collisions_4_actor, cbar_layout=True,
                                     title=f'Minimum FeAR', ax=axes[1], plot_mins=True)
        max_fear_by_actor = np.max(fear_on_others, axis=2)
        plot_fear_grid_search_for_ij(fear_by_actor=max_fear_by_actor, actor_as=actor_as, actor_thetas=actor_thetas,
                                     collisions_4_actor=collisions_4_actor, cbar_layout=False,
                                     cbar=True, return_pcolormesh=False,
                                     title=f'Maximum FeAR', ax=axes[2], plot_mins=True)

        # cbar = fig.colorbar(pcolormesh, ax=axes[2], location='right', orientation='vertical', shrink=0.2)
        # cbar.outline.set_visible(False)  # Remove the black outline

        fig.tight_layout()

        fear_grid_aggregates_ax = axes[0]

        fig, axes = plt.subplots(1, 3, figsize=(10, 6 * 3), dpi=200)

        # Define gridspec with 1 row and 3 columns, and a colorbar column
        # gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])  # Add a fourth column for the colorbar
        # fig = plt.figure(figsize=(10, 6 * 3), dpi=200)
        #
        # # Create subplots
        # ax0 = fig.add_subplot(gs[0])
        # ax1 = fig.add_subplot(gs[1])
        # ax2 = fig.add_subplot(gs[2])
        # axes = [ax0, ax1, ax2]

        for ax in axes:
            ax.grid(True, linestyle=':')
            ax.set_aspect("equal")
            make_axes_gray(ax)

        number_negative_fear_by_actor = - np.sum(fear_on_others < 0, axis=2)
        number_positive_fear_by_actor = np.sum(fear_on_others > 0, axis=2)
        number_assertive_minus_courteous = number_positive_fear_by_actor + number_negative_fear_by_actor

        fear_numbers_by_actor = np.hstack((number_negative_fear_by_actor,
                                           number_negative_fear_by_actor,
                                           number_assertive_minus_courteous))

        peak_fear_number = np.max(np.abs(fear_numbers_by_actor))
        # So that all the number plots have the same cmap scale

        plot_fear_grid_search_for_ij(fear_by_actor=number_negative_fear_by_actor, recalibrate_cmap=True,
                                     actor_as=actor_as, actor_thetas=actor_thetas, peak=peak_fear_number,
                                     collisions_4_actor=collisions_4_actor, cbar_layout=True, ylabel=True,
                                     title=f'Number Courteous to', ax=axes[0], plot_mins=True)
        plot_fear_grid_search_for_ij(fear_by_actor=number_positive_fear_by_actor, recalibrate_cmap=True,
                                     actor_as=actor_as, actor_thetas=actor_thetas, peak=peak_fear_number,
                                     collisions_4_actor=collisions_4_actor, cbar_layout=True,
                                     title=f'Number Assertive to', ax=axes[1], plot_mins=True)
        plot_fear_grid_search_for_ij(fear_by_actor=number_assertive_minus_courteous,
                                     recalibrate_cmap=True,
                                     actor_as=actor_as, actor_thetas=actor_thetas,
                                     peak=peak_fear_number, cbar=True,
                                     collisions_4_actor=collisions_4_actor, cbar_layout=True,
                                     title=f'Number (Assertive - Courteous) to', ax=axes[2],
                                     plot_mins=True)
        fig.tight_layout()

        fear_grid_aggregates_counts_ax = axes[0]

        if save_figs:
            save_figure(ax=fear_grid_aggregates_ax,
                        filename=f"{results_file_name}_fear_grids_actor_{actor}_aggregates")
            save_figure(ax=fear_grid_aggregates_counts_ax,
                        filename=f"{results_file_name}_fear_grids_actor_{actor}_aggregates_counts")
            save_figure(ax=fear_grid_affected_ax,
                        filename=f"{results_file_name}_fear_grids_actor_{actor}_each_affected")
        else:
            plt.show()


def plot_fearless_actions_for_actor(a, actor, actor_as, actor_thetas, l, fear_by_actor, n_timesteps, obstacle,
                                    scale_velocities, strip_scenario_titles, t_action, theta, tweaked_title, v0x, v0y,
                                    x0, y0, arrow_width=0.45, arrow_markersize=8, collisions_4_actor=None
                                    ):
    min_actors_a, min_actors_theta, min_value = cfear.minimise_fear(actor_as, actor_thetas, fear_by_actor,
                                                                    collisions_4_actor)

    tweaked_a = copy.copy(a)
    tweaked_theta = copy.copy(theta)

    n_mins = len(min_actors_a)

    if n_mins > 6:
        n_rows = np.ceil(np.sqrt(n_mins)).astype(int)
        fig, axes = plt.subplots(n_rows, n_rows, figsize=(8 * n_rows, 8 * n_rows), layout="constrained", dpi=200)
        axes = axes.flatten()  # Flatten the 2D array of axes into a 1D array
        scale_font = n_rows
    else:
        fig, axes = plt.subplots(1, n_mins, figsize=(8, 8 * n_mins), layout="constrained", dpi=200)
        scale_font = 1

    if n_mins == 1:
        axes = [axes]

    for ax in axes:
        ax.grid(True, linestyle=':')
        ax.set_aspect("equal")
        make_axes_gray(ax)
        ax.set_anchor('N')
        strip_axes(ax=ax)

    for oo, (tweaked_actor_a, tweaked_actor_theta) in enumerate(zip(min_actors_a, min_actors_theta)):
        tweaked_a[actor] = tweaked_actor_a
        tweaked_theta[actor] = tweaked_actor_theta

        plot_tweaked_scenario(tweaked_a=tweaked_a, tweaked_theta=tweaked_theta,
                              tweaked_title=None,
                              v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                              l=l, n_timesteps=n_timesteps, obstacle=obstacle,
                              scale_velocities=scale_velocities,
                              arrow_width=arrow_width,
                              arrow_markersize=arrow_markersize,
                              strip_scenario_titles=strip_scenario_titles, t_action=t_action,
                              agents_to_highlight=[actor],
                              ax=axes[oo]
                              )
    synchronize_axes_limits(fig)
    fig.suptitle(tweaked_title, fontsize=12 * scale_font)

    return axes[0]


def extend_axis_limits(ax, padding=10, reFit=False):
    """
    Extend the xlims and ylims of a matplotlib axis by the given padding value.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axis to extend.
    padding : int
        The padding value to extend xlims and ylims by.
    reFit : bool,
        Whether to rescale the image to fit everything
    """

    if reFit:
        ax.autoscale()  # Adjust limits to fit all elements

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ax.set_xlim(xlim[0] - padding, xlim[1] + padding)
    ax.set_ylim(ylim[0] - padding, ylim[1] + padding)


def illustrate_subspace_collision_check(theta_step_threshold=np.pi / 8, a_step_threshold=10, title='', save_fig=False,
                                        mark_collisions=True, actor=0, affected=1, custom_feasible_facecolor=True,
                                        scenario_name=None, scenarios_json_file='Scenarios4cFeAR.json'):
    t_action = 5
    n_timesteps = 10

    dt = t_action / n_timesteps
    t = np.arange(0, t_action, dt)

    if scenario_name is None:
        l = 1

        x0_1 = 10
        y0_1 = 0
        x0_j = 0
        y0_j = 0

        v0x_1 = 1
        v0y_1 = 1
        v0x_j = 1
        v0y_j = 1

        a_1 = 2
        theta_1 = np.pi
        ax_1 = a_1 * np.cos(theta_1)
        ay_1 = a_1 * np.sin(theta_1)

        # a_j_l = 0.5
        # a_j_u = 1
        #
        # theta_j_l = np.pi / 2
        # theta_j_u = np.pi

        a_j_l = 1.0
        a_j_u = 1.5

        theta_j_l = 2.356194490192345
        theta_j_u = 2.552544031041707

        obstacle = None

    else:
        scenarios = cfear.Scenario.load_scenarios(scenarios_json_file)
        scenario = scenarios[scenario_name]

        # Populating variables with scenario data
        l = scenario.l

        x0 = scenario.x0
        y0 = scenario.y0
        v0x = scenario.v0x
        v0y = scenario.v0y
        a = scenario.a
        theta = scenario.theta

        if scenario.obstacle != "None":
            obstacle = scenario.obstacle
        else:
            obstacle = None

        x0_1 = x0[actor]
        y0_1 = y0[actor]
        x0_j = x0[affected]
        y0_j = y0[affected]

        v0x_1 = v0x[actor]
        v0y_1 = v0y[actor]
        v0x_j = v0x[affected]
        v0y_j = v0y[affected]

        a_1 = a[actor]
        theta_1 = theta[actor]

        ax_1 = a_1 * np.cos(theta_1)
        ay_1 = a_1 * np.sin(theta_1)

        # a_j_l = 1.0
        # a_j_u = 1.5
        #
        # # theta_j_l = np.pi / 4 + np.pi/6
        # # theta_j_u = np.pi * 3 / 4
        #
        # theta_j_l = np.pi * (4 / 8 + 1 / 16)
        # theta_j_u = np.pi * (6 / 8)

        a_j_l = 1.0
        a_j_u = 1.5

        theta_j_l = 2.356194490192345
        theta_j_u = 2.552544031041707

    x_1 = x0_1 + v0x_1 * t + ax_1 * t ** 2 / 2
    y_1 = y0_1 + v0y_1 * t + ay_1 * t ** 2 / 2
    boxes_1 = cfear.get_boxes_for_a_traj(x=x_1, y=y_1, lx=l, ly=l)  # Get traj_boxes
    traj_hulls_1 = cfear.get_traj_hulls(boxes_1)  # Get traj_hulls

    if abs(a_j_u - a_j_l) <= a_step_threshold:
        a_j_subs = np.array([a_j_l, a_j_u])
        n_a_subs = 2
    else:
        n_a_subs = np.ceil((a_j_u - a_j_l) / a_step_threshold).astype(int)
        a_j_subs = np.linspace(a_j_l, a_j_u, num=n_a_subs)

    if abs(theta_j_u - theta_j_l) <= theta_step_threshold:
        theta_j_subs = np.array([theta_j_l, theta_j_u])
        n_theta_subs = 2
    else:
        n_theta_subs = np.ceil((theta_j_u - theta_j_l) / theta_step_threshold).astype(int)
        theta_j_subs = np.linspace(theta_j_l, theta_j_u, num=n_theta_subs)

    n_subs = n_a_subs * n_theta_subs

    # ---------------------------------------------------------------------------------------
    # ----------------------Minimum resolution in subspaces----------------------------------
    xy_j_s = [cfear.make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
                                      t_action=t_action, n_timesteps=n_timesteps,
                                      a=a_, theta=theta_)
              for a_ in a_j_subs
              for theta_ in theta_j_subs]

    boxes_j = [cfear.get_boxes_for_a_traj(x=x_j_, y=y_j_, lx=l, ly=l)
               for x_j_, y_j_, _ in xy_j_s]

    # Get traj_hulls
    traj_hulls_j_ = [cfear.get_traj_hulls(boxes) for boxes in boxes_j]
    traj_hulls_j = np.array(traj_hulls_j_).transpose()

    last_boxes_j = np.array([boxes[-1] for boxes in boxes_j]).reshape((1, n_subs))
    last_hull_j = cfear.get_hull_of_polygons(last_boxes_j)

    last_hull_j = np.array([last_hull_j[0].buffer(0.1)])
    # To fix the discontinuity at the joint between the first and last edges.

    traj_hulls_of_hulls_j = cfear.get_hull_of_polygons(traj_hulls_j)

    # ---------------------------------------------------------------------------------------
    colour_last = BLUE

    feasible_facecolor = list(mcolors.to_rgba(BLUE))
    feasible_facecolor[-1] = 0.8
    feasible_facecolor = tuple(feasible_facecolor)

    facecolor_last = feasible_facecolor

    agent_colours = scientific_cmap(n_colours=2)
    hatches = [None, '....']

    for ii, hull in enumerate(traj_hulls_of_hulls_j):

        # fig = plt.figure(figsize=tuple(s * 0.75 for s in SIZE), dpi=200)
        fig = plt.figure(figsize=SIZE, dpi=200)
        ax = fig.add_subplot(121)
        # ax.set_title(title + f' $\\Delta t$:{ii + 1}')

        plot_traj_hulls(traj_hulls_1, traj_boxes=boxes_1, ax=ax, colour=LIGHTGRAY,
                        stripped_axes=True, hatch=hatches[actor])

        for jj, hulls_2 in enumerate(traj_hulls_j_):
            plot_traj_hulls(hulls_2, traj_boxes=boxes_j[jj], ax=ax, colour=LIGHTGRAY, hatch=hatches[affected],
                            stripped_axes=True)

        collision = hull.intersects(traj_hulls_1[ii])

        facecolor = None

        if collision:
            if mark_collisions:
                colour = RED
            colour_last = RED
            facecolor_last = None
        else:
            # colour = BLUE
            if mark_collisions:
                colour = BLUE
                if custom_feasible_facecolor:
                    facecolor = feasible_facecolor
            else:
                colour = agent_colours[affected]

        for hulls_2 in traj_hulls_j_:
            plot_polygon_outline(hulls_2[ii], ax=ax, color=colour)

        # plot_polygon(hull, ax=ax, color=colour, add_points=False, label='Affected agent')
        plot_polygon(hull, ax=ax, color=colour, facecolor=facecolor, add_points=False,
                     label='Affected agent', hatch=hatches[affected])

        # plot_polygon(traj_hulls_1[ii], ax=ax, color=YELLOW, add_points=False, label='Other agent', hatch='....')
        plot_polygon(traj_hulls_1[ii], ax=ax, color=agent_colours[actor], add_points=False,
                     label='Other agent', hatch=hatches[actor])

        plot_obstacle(obstacle=obstacle, ax=ax)
        # ax.legend(bbox_to_anchor=(0.99, 0.99), loc='upper right')
        ax.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left')

        ax.get_legend().remove()

        if save_fig:
            save_figure(ax=ax, filename=f'FeasibilityCheck_{ii}')

    #  Last Hull
    # fig = plt.figure(figsize=tuple(s * 0.75 for s in SIZE), dpi=200)
    fig = plt.figure(figsize=SIZE, dpi=200)
    ax = fig.add_subplot(121)
    ax.set_title(title)

    for jj, hulls_2 in enumerate(traj_hulls_j_):
        plot_traj_hulls(hulls_2, traj_boxes=boxes_j[jj], ax=ax, colour=LIGHTGRAY, stripped_axes=True,
                        hatch=hatches[affected])

    plot_polygon(last_hull_j[0], ax=ax, color='lightgray',
                 linewidth=6, add_points=False, label='')
    plot_polygon(last_hull_j[0], ax=ax, color=WHITE,
                 linewidth=5, add_points=False, label='')
    plot_polygon(last_hull_j[0], ax=ax, color=colour_last, facecolor=facecolor_last,
                 linewidth=2, add_points=False, label='Affected agent')
    # plot_traj_hulls(last_hull_j, ax=ax, colour=colour_last, label='Affected agent')

    # plot_traj_hulls(traj_hulls_1, traj_boxes=boxes_1, ax=ax, colour=LIGHTGRAY, stripped_axes=True,
    #                 hatch=hatches[actor], label='Other Agent')
    plot_traj_hulls(traj_hulls_1, traj_boxes=boxes_1, ax=ax, colour=agent_colours[actor], stripped_axes=True,
                    hatch=hatches[actor], label='Other Agent')

    plot_obstacle(obstacle=obstacle, ax=ax)
    ax.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left')
    ax.get_legend().remove()

    if save_fig:
        save_figure(ax=ax, filename='FeasibilityCheck_FinalConfig')


def adjust_lightness(color, amount=0.8, max_saturation=False):
    """
    From https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    Parameters
    ----------
    color
    amount

    Returns
    -------

    """
    import colorsys
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mcolors.to_rgb(c))

    if max_saturation:
        # set s=1
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), 1)

    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


def color_names_to_rgb(color_names):
    rgb_values = [mcolors.to_rgba(color) for color in color_names]
    return rgb_values


def special_cmap(n_colours=5, game_mode=False, ego_id=0, pastel=True):
    if game_mode:
        ego_colour = (0.5820686065676477, 0.7683811347467602, 0.8872124391839864)
        others_colour = (0.996078431372549, 0.8784313725490196, 0.5450980392156862)
        colours = []
        for ii in range(n_colours):
            if ii == ego_id:
                colours.append(ego_colour)
            else:
                colours.append(others_colour)
        return colours

    if pastel:
        if n_colours >= 5:
            return sns.color_palette("Spectral", n_colors=n_colours)
        elif n_colours == 2:
            return color_names_to_rgb(['#7ffeff', '#ff7ffe'])
        elif n_colours == 3:
            return color_names_to_rgb(['#7ffeff', '#ff7ffe', '#ffeb7f'])
        elif n_colours == 4:
            return color_names_to_rgb(['#7ffeff', '#ff7ffe', '#ffeb7f', '#d6ff97'])
        else:
            return color_names_to_rgb(['#7ffeff'])

        # if n_colours >= 6:
        #     return sns.color_palette("Spectral", n_colors=n_colours)
        # elif n_colours == 2:
        #     return color_names_to_rgb(['#19ebef', '#ffbc40'])
        # elif n_colours == 3:
        #     return color_names_to_rgb(['#19ebef', '#ffbc40', '#d6464b'])
        # elif n_colours == 4:
        #     return color_names_to_rgb(['#00c0e9', '#19ebef', '#fac288', '#d6464b'])
        # elif n_colours == 5:
        #     return color_names_to_rgb(['#29c6e7', '#4cdcdf', '#eec16c', '#eba35b', '#d06d70'])
        # else:
        #     return color_names_to_rgb(['#f49569'])

    # Retro Colours
    retro = ['#001219', '#005f73', '#0a9396', '#94d2bd', '#e9d8a6', '#ee9b00', '#ca6702', '#bb3e03', '#ae2012',
             '#9b2226']

    # Function to get the set of spectral colours I like

    if n_colours >= 5:
        return sns.color_palette("Spectral", n_colors=n_colours)
    elif n_colours == 2:
        return color_names_to_rgb([retro[i] for i in [2, 5]])
    elif n_colours == 3:
        return color_names_to_rgb([retro[i] for i in [2, 5, 9]])
    elif n_colours == 4:
        return color_names_to_rgb([retro[i] for i in [1, 2, 5, 9]])

    # elif n_colours == 5:
    #     return color_names_to_rgb([retro[i] for i in [1, 2, 5, 6, 9]])
    # elif n_colours == 6:
    #     return color_names_to_rgb([retro[i] for i in [1, 2, 3, 5, 6, 9]])
    # elif n_colours == 7:
    #     return color_names_to_rgb([retro[i] for i in [1, 2, 3, 4, 5, 6, 9]])
    # elif n_colours == 8:
    #     return color_names_to_rgb([retro[i] for i in [0, 1, 2, 3, 4, 5, 6, 9]])
    # elif n_colours == 9:
    #     return color_names_to_rgb([retro[i] for i in [0, 1, 2, 3, 4, 5, 6, 8, 9]])
    # elif n_colours == 10:
    #     return color_names_to_rgb(retro)
    else:
        colours = sns.color_palette("Spectral", n_colors=9)
        return [(0.9155324875048059, 0.6192233756247598, 0.65440215301807)]
        # return colours[0]


def scientific_cmap(n_colours=5, swap_2_3=True):
    # cmap = cmc.batlow
    cmap = cmc.batlowK

    # cmap = cmc.lipari
    # cmap = cmc.glasgow
    colors = [cmap(x) for x in np.linspace(0.05, 0.95, n_colours)]

    if n_colours == 3 and swap_2_3 == True:
        colors = [colors[ii] for ii in [0, 2, 1]]
    return colors


def save_figure(ax, filename, folder="Plots_cFeAR", file_extension='.png', size=None, dpi=600,
                title_fontsize=None, label_fontsize=None, tick_fontsize=None, close_after=True):
    """
    Save a matplotlib figure to a file.

    Parameters:
    - ax: matplotlib.axes.Axes object
        The axis of the plot to be saved.
    - filename: str
        The filename to save the figure as.
    - folder: str, optional (default=None)
        The folder where the figure should be saved. If None, the current directory is used.
    - file_extension: str, optional (default='.png')
        The file extension for the saved figure.
    - size: tuple, optional (default=None)
        Size of the figure in inches (width, height). If None, the current size is used.
    - dpi: int, optional (default=None)
        Dots per inch for the saved figure. If None, the current DPI is used.
    - title_fontsize: int, optional (default=None)
        Font size for the title of the plot. If None, the current font size is used.
    - label_fontsize: int, optional (default=None)
        Font size for the axis labels. If None, the current font size is used.
    - tick_fontsize: int, optional (default=None)
        Font size for the tick labels. If None, the current font size is used.
    - close_after: bool, optional (default=True)
        If True, the matplotlib figure will be closed at the end of the function.

    """

    # Set the size of the figure if specified
    fig = ax.figure
    if size is not None:
        fig.set_size_inches(size)

    # Set the DPI of the figure if specified
    if dpi is not None:
        fig.dpi = dpi

    # Apply settings to all axes in the figure
    for axis in fig.get_axes():
        # Set font sizes if specified
        if title_fontsize is not None:
            axis.title.set_fontsize(title_fontsize)
        if label_fontsize is not None:
            axis.xaxis.label.set_fontsize(label_fontsize)
            axis.yaxis.label.set_fontsize(label_fontsize)
        if tick_fontsize is not None:
            axis.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    # Create the full file path
    if folder is not None:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"{filename}{file_extension}")
    else:
        filepath = f"{filename}{file_extension}"

    # Save the figure
    print(f'Saving figure: {filepath} !')

    # with matplotlib.rc_context({'ps.fonttype': 42, 'pdf.fonttype': 42}):
    #     fig.savefig(filepath, bbox_inches='tight')

    fig.savefig(filepath, bbox_inches='tight')

    if close_after:
        plt.close(fig)


def strip_axes(ax, strip_title=True, strip_frame=True, strip_legend=True):
    """Function to remove the axes, labels and titles of ax

    Parameters
    ----------
    ax : matplotlib ax
    strip_frame : bool, optional
        Whether to remove the frame.
    strip_title: bool, optional
        Whether to remove the title
    strip_legend: bool, optional
        Whether to remove the legend

    Returns
    -------

    """
    if strip_frame:
        ax.set_frame_on(False)

    if strip_title:
        ax.set_title('')

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    if strip_legend:
        if ax.get_legend():
            ax.get_legend().remove()


def plot_tweaked_scenario(tweaked_a, tweaked_theta, tweaked_title,
                          v0x, v0y, x0, y0,
                          l, t_action, n_timesteps, obstacle,
                          scale_velocities, strip_scenario_titles,
                          arrow_width=0.45, arrow_markersize=8,
                          scale_scenario_size=1,
                          grayscale=False, agents_to_highlight=None, ax=None, show_last=False,
                          legend_inside=False, show_legend=False
                          ):
    """Function to create the make the trajectories and plot scenario for a given joint action.

    Parameters
    ----------
    arrow_markersize
    arrow_width
    show_legend
    legend_inside
    scale_scenario_size
    tweaked_a : ndarray
        Magnitudes of Acceleration of the joint action
    tweaked_theta: ndarray
        Directions of acceleration of the joint action
    tweaked_title: str
        Title for the scenario plot
    grayscale : bool
        Whether to plot scenario in grayscale
    agents_to_highlight : ndarray
        List of agents to highlight, while the other agents are grayscale
    ax
    v0x
    v0y
    x0
    y0
    l
    t_action
    n_timesteps
    obstacle
    scale_velocities
    strip_scenario_titles
    show_last

    Returns
    -------
    """

    if grayscale or agents_to_highlight:
        colour = 'lightgrey'
    else:
        colour = None

    x_tweaked, y_tweaked, t_tweaked = cfear.make_trajectories(x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                                              a=tweaked_a, theta=tweaked_theta,
                                                              t_action=t_action, n_timesteps=n_timesteps)
    trajs_boxes_tweaked = cfear.get_boxes_for_trajs(x_tweaked, y_tweaked, lx=l, ly=l)
    trajs_hulls_tweaked = cfear.get_trajs_hulls(trajs_boxes_tweaked)
    ax = plot_hulls_for_trajs(trajs_hulls_tweaked, trajs_boxes=trajs_boxes_tweaked, title=tweaked_title,
                              colour=colour, agents_to_colour=agents_to_highlight, ax=ax,
                              scale_scenario_size=scale_scenario_size, show_last=show_last,
                              legend_inside=False, show_legend=False)

    ax = plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y, ax=ax,
                                    scale=t_action / n_timesteps * scale_velocities,
                                    arrow_width=arrow_width,
                                    markersize=arrow_markersize)
    if show_legend:
        if legend_inside:
            ax.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right')
        else:
            ax.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

    plot_obstacle(obstacle, ax=ax)
    set_fontsizes(ax=ax)
    strip_axes(ax=ax, strip_legend=False, strip_title=strip_scenario_titles)
    return ax


def rescale_cbar(cbar, scale=0.7, scale_width=1.0):
    """
    Rescale the height of an existing colorbar and keep it centered.

    Parameters:
    - cbar: The colorbar object to be resized.
    - scale: A float representing the scaling factor for the height (default is 0.7).
    """

    # Get the current position of the colorbar as a tuple: (left, bottom, width, height)
    cbar_ax = cbar.ax
    pos = cbar_ax.get_position().bounds

    # Modify the height to be `scale` times the original height
    new_height = pos[3] * scale

    # Keep the colorbar centered by adjusting the bottom position
    new_bottom = pos[1] + (pos[3] - new_height) / 2

    # Set the new position: [left, bottom, width, new_height]
    cbar_ax.set_position([pos[0], new_bottom, pos[2] * scale_width, new_height])


# Helper function to determine text color (black or white) based on background
def get_contrasting_text_color(rgb):
    # Perceived luminance formula to determine brightness
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    return 'white' if luminance < 0.5 else 'black'


def plot_agent_ids(ax=None, x0=None, y0=None, zorder_lift=0, agent_colours=False, highlight_agents=[]):
    """
    Function to add labels for agents on scenario plots

    Parameters
    ----------
    highlight_agents
    agent_colours
    ax
    x0
    y0
    zorder_lift

    Returns
    -------
    ax
    """
    if ax is None:
        print('No ax passed! Aborting!')
        return False
    if x0 is None:
        print("No x0 passsed! Aborting!")
    if y0 is None:
        print("No y0 passsed! Aborting!")
    if len(x0) != len(y0):
        print("Lengths of x0 and y0 don't match! Aborting!")

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes to the graph
    n_agents = len(x0)
    for i in range(n_agents):
        G.add_node(i, pos=(x0[i], y0[i]))

    # Get positions of nodes
    pos = nx.get_node_attributes(G, 'pos')

    # Draw node labels as i + 1
    labels = {i: str(i + 1) for i in G.nodes()}

    if agent_colours:
        colours = scientific_cmap(n_colours=n_agents)
        # Determine contrasting label colors for each agent
        label_colours = {ii: get_contrasting_text_color(rgb) for ii, rgb in enumerate(colours)}
    else:
        colours = 'moccasin'
        # colours = 'lightgray'
        # label_colours = 'black'

    if highlight_agents:
        node_sizes = [500 if ii in highlight_agents else 300 for ii in range(len(x0))]
        node_edges = ['tan' if ii in highlight_agents else 'moccasin' for ii in range(len(x0))]
        # node_edges = ['tan' if ii in highlight_agents else 'lightgray' for ii in range(len(x0))]
        # node_edges = ['lightgray' if ii in highlight_agents else 'moccasin' for ii in range(len(x0))]

    else:
        node_sizes = 300
        node_edges = 'face'

    # Draw the graph
    if zorder_lift >= 0:
        nodes_drawn = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=colours,
                                             edgecolors=node_edges, linewidths=3,
                                             ax=ax, alpha=0.95)
        labels_drawn = nx.draw_networkx_labels(G, pos, labels=labels,
                                               horizontalalignment='center', verticalalignment='center_baseline')

        nodes_drawn.set_zorder(zorder_lift + 2)

        for i, (_, label_drawn) in enumerate(labels_drawn.items()):
            label_drawn.set_zorder(zorder_lift + 4)

            if agent_colours:
                label_drawn.set_color(label_colours[i])

    # Remove legend
    ax.get_legend().remove()

    return ax


def rescale_plot_size(ax=None, scale=1):
    """
    Rescales the size of an image in the provided axes by the given scale factor.

    Parameters:
    ax : matplotlib.axes.Axes, optional
        The axes containing the image. If not provided, the function will return without doing anything.
    scale : float, optional
        The factor by which the figure size will be scaled. Default is 1 (no scaling).

    """

    if ax is None:
        print('No image ax passed! Aborting!')
        return False

    # Get the figure containing the axes
    fig = ax.get_figure()

    # Get the current size of the figure in inches
    current_size = fig.get_size_inches()

    # Rescale the size by the scale factor
    new_size = current_size * scale

    # Set the new figure size
    fig.set_size_inches(new_size, forward=True)


def plot_rect_on_matrix(**kwargs):
    return PlotGWorld.plot_rect_on_matrix(**kwargs)
