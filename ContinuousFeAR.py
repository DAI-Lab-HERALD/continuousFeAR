import numpy as np
import shapely.wkt
from scipy.optimize import minimize, differential_evolution

import Plot_cFeAR as plotcf

from shapely.geometry import LinearRing
from shapely.geometry import Point
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import rotate

from numba import jit
from functools import partial
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import time
from tqdm import tqdm

import json
import pprint
import sys

import os
import pickle
import datetime

# Garbage collector
import gc

import copy

# gc.set_debug(True)
# gc.get_threshold()
LEGEND_FONTSIZE = 24

EPS = 0.000001
MULTI_PROCESSING_THRESHOLD_GET_FEAR = 8

THETA_STEP_THRESHOLD = np.pi / 16
A_STEP_THRESHOLD = 10


def main():
    pass


if __name__ == "__main__":
    main()


@jit
def make_trajectories(x0=0, y0=0, v0x=0, v0y=0,
                      a=0, theta=0,
                      t_action=10.0, n_timesteps=10):
    dt = t_action / n_timesteps
    t = np.arange(0, t_action, dt)

    ax = a * np.cos(theta)
    ay = a * np.sin(theta)

    x = x0 + v0x * t + ax * t ** 2 / 2
    y = y0 + v0y * t + ay * t ** 2 / 2

    return x, y, t


def get_a_and_theta(ax=0, ay=0):
    # Calculate magnitude of acceleration vector
    a = np.sqrt(ax ** 2 + ay ** 2)

    # Calculate angle theta using arctan2
    theta = np.arctan2(ay, ax)

    # # Ensure theta is in the range (-pi, pi]
    # if theta <= -np.pi:
    #     theta += 2 * np.pi
    # elif theta > np.pi:
    #     theta -= 2 * np.pi

    return a, theta


def generate_starting_locations(x0_mean=0, x0_std=10,
                                y0_mean=0, y0_std=10,
                                obstacle=None,
                                n=1, l=2):
    x0s = []
    y0s = []
    points = []

    while len(points) < n:
        x0 = x0_mean + x0_std * np.random.randn(1)
        y0 = y0_mean + y0_std * np.random.randn(1)

        if obstacle is not None:
            point = Point((x0, y0)).buffer(2 * l)
            # Adding a buffer 2 times the dimension of the agent as a buffer.

            if obstacle.intersects(point):
                # Skip adding point since it intersects the obstacle
                continue

        # Check the distance to existing points
        if all(np.linalg.norm(np.array([x0, y0]) - np.array(point)) >= l for point in points):
            points.append([x0, y0])
            x0s.append(x0)
            y0s.append(y0)

    return np.array(x0s), np.array(y0s)


def make_circle(x=0, y=0, radius=1):
    """
    Create a circle given its center (x, y) and radius.

    Parameters
    ----------
    x : float
        The x coordinates of the center.
    y : float
        The y coordinates of the center.
    radius : float
        The radius of the circle.

    Returns
    -------
    Polygon
        The circle polygon.
    """
    return Point((x, y)).buffer(radius)


def make_rectangle(x=0, y=0, lx=1, ly=1, theta=0):
    """
    Create a rectangle given its center (x,y) and its length in x and y directions (lx, ly)
    and rotated by theta degrees.

    Parameters
    ----------
    x : float
        The x coordinate of the center
    y : float
        The y coordinate of the center
    lx : float
        The length along the x-axis (before rotation)
    ly : float
        The length along the y-axis (before rotation)
    theta : float
        The rotation angle in degrees

    Returns
    -------
    Polygon
        The rectangle polygon.
    """
    xl = x - lx / 2
    xu = x + lx / 2
    yl = y - ly / 2
    yu = y + ly / 2

    rectangle = Polygon([(xl, yl), (xu, yl), (xu, yu), (xl, yu), (xl, yl)])
    rotated_rectangle = rotate(rectangle, theta, origin='center', use_radians=False)

    return rotated_rectangle


# Get trajectory boxes
@jit(forceobj=True)
def get_boxes_for_a_traj(x, y, lx=1, ly=1):
    # Function to get the boxes for the trajectory of one agent
    xl = x - lx / 2
    xu = x + lx / 2
    yl = y - ly / 2
    yu = y + ly / 2

    # boxes = []
    # for ii in range(len(xl)):
    #     ext = [(xl[ii], yl[ii]), (xu[ii], yl[ii]),
    #            (xu[ii], yu[ii]), (xl[ii], yu[ii]),
    #            (xl[ii], yl[ii])]
    #           int = []
    #     print(f'{ext=}')
    #     box = Polygon(ext)
    #     boxes.append(box)

    # Directly create polygons without intermediate arrays
    boxes = [Polygon([(xl[ii], yl[ii]), (xu[ii], yl[ii]), (xu[ii], yu[ii]), (xl[ii], yu[ii]), (xl[ii], yl[ii])])
             for ii in range(len(x))]

    # p1 = np.column_stack((xl, yl))
    # p2 = np.column_stack((xu, yl))
    # p3 = np.column_stack((xu, yu))
    # p4 = np.column_stack((xl, yu))
    #
    # exts = zip(p1, p2, p3, p4, p1)
    # boxes = [Polygon(ext) for ext in exts]

    return boxes


@jit(forceobj=True)
def get_boxes_for_a_traj_caching_wrapper(x, y, lx=1, ly=1):
    x = tuple(x)
    y = tuple(y)
    return get_boxes_for_a_traj_cached(x, y, lx=lx, ly=ly)


@lru_cache(maxsize=256)
def get_boxes_for_a_traj_cached(x, y, lx=1, ly=1):
    x = np.array(x)
    y = np.array(y)
    return get_boxes_for_a_traj(x=x, y=y, lx=lx, ly=ly)


@jit(forceobj=True)
def get_boxes_for_trajs(xs, ys, lx=1, ly=1, as_ndarray=True):
    # Function to get the boxes for the trajectories of nn agents

    trajs_boxes = [get_boxes_for_a_traj(x, y, lx=lx, ly=ly)
                   for x, y in zip(xs, ys)]

    # trajs_boxes = []
    # for x, y in zip(xs, ys):
    #     traj_boxes = get_boxes_for_a_traj(x, y, lx=lx, ly=ly)
    #     trajs_boxes.append(traj_boxes)

    if as_ndarray:
        trajs_boxes_ndarray = np.array(trajs_boxes)
        return trajs_boxes_ndarray
    else:
        return trajs_boxes


# get trajectory hulls for one agent (for the timesteps)
@jit(forceobj=True)
def get_traj_hulls(boxes):
    traj_hulls = [boxes[ii].union(boxes[ii + 1]).convex_hull
                  for ii in range(len(boxes) - 1)]

    # traj_hulls = []
    # for ii in range(len(boxes) - 1):
    #     union = boxes[ii].union(boxes[ii + 1])
    #     hull = union.convex_hull
    #     traj_hulls.append(hull)

    return traj_hulls


# get trajectory hulls for all agents (for the timesteps)
def get_trajs_hulls(trajs_boxes, as_ndarray=True):
    try:
        n_trajs, _ = trajs_boxes.shape
    except AttributeError:
        trajs_boxes = np.array(trajs_boxes)
        n_trajs, _ = trajs_boxes.shape

    trajs_hulls = [get_traj_hulls(trajs_boxes[ii, :])
                   for ii in range(n_trajs)]

    # trajs_hulls = []
    #
    # for ii in range(n_trajs):
    #     traj_hulls = get_traj_hulls(trajs_boxes[ii, :])
    #     trajs_hulls.append(traj_hulls)

    if as_ndarray:
        return np.array(trajs_hulls)
    else:
        return trajs_hulls


# Check Collisions for Traj Hulls
def check_hull_collisions(hulls_1, hulls_2):
    collision = False
    for ii in range(len(hulls_1)):
        hull_1 = hulls_1[ii]
        hull_2 = hulls_2[ii]
        if hull_1.intersects(hull_2):
            collision = True
            return collision
    return collision


def get_hull_of_hulls_1d(hulls1, hulls2):
    hull_of_hulls = []
    for ii in range(len(hulls1)):
        union = hulls1[ii].union(hulls2[ii])
        hull = union.convex_hull
        hull_of_hulls.append(hull)
    return hull_of_hulls


def get_hull_of_hulls_2d(hulls1, hulls2, hulls3, hulls4):
    hulls_12 = get_hull_of_hulls_1d(hulls1, hulls2)
    hulls_34 = get_hull_of_hulls_1d(hulls3, hulls4)
    hull_of_hulls = get_hull_of_hulls_1d(hulls_12, hulls_34)
    return hull_of_hulls


# @jit(forceobj=True)  #  This makes it slightly slower.
def get_hull_of_polygons(polygons=None, use_cache=True):
    if polygons is None:
        print('No polygons passed!')
        return False

    # print(f'{polygons.shape=}')

    n_trajs, _ = polygons.shape

    if use_cache:
        hull_of_polygons = [get_hull_of_polygons_of_iith_traj_caching_wrapper(ii=ii, polygons=polygons)
                            for ii in range(n_trajs)]
    else:
        hull_of_polygons = [get_hull_of_polygons_of_iith_traj(ii=ii, polygons=polygons)
                            for ii in range(n_trajs)]

    # hull_of_polygons = []
    #
    # for ii in range(n_trajs):
    #     hull = get_hull_of_polygons_of_iith_traj(ii=ii, polygons=polygons)
    #     hull_of_polygons.append(hull)

    return np.array(hull_of_polygons)


def get_hull_of_polygons_of_iith_traj(ii=None, polygons=None):
    multipolygon = MultiPolygon([jj for jj in polygons[ii, :]])
    try:
        hull = multipolygon.convex_hull
    except:
        union = multipolygon.buffer(0)
        hull = union.convex_hull
    return hull


# @jit(forceobj=True)
def get_hull_of_polygons_of_iith_traj_caching_wrapper(ii=None, polygons=None):
    # print(f'{polygons=}, {type(ii)}')
    polygons = get_tuples_of_polygons(polygons)
    # print(f'{polygons=}, {type(ii)}')
    return get_hull_of_polygons_of_iith_traj_cached(ii=ii, polygons=polygons)


def get_tuples_of_polygons(polygons=None):
    # print(f'{polygons=}')
    polygons = tuple(tuple(item for item in array) for array in polygons)
    # print(f'{polygons=}')
    return polygons


def get_polygons_from_tuples(polygons=None):
    # print(f'{polygons=}')
    polygons = np.array([[item for item in array] for array in polygons])
    # print(f'{polygons=}')
    return polygons


@lru_cache(maxsize=None)
def get_hull_of_polygons_of_iith_traj_cached(ii=None, polygons=None):
    polygons = get_polygons_from_tuples(polygons)
    return get_hull_of_polygons_of_iith_traj(ii=ii, polygons=polygons)


def intersecting_polygons_with_plots(polygons=None, plot_polygons=False, obstacle=None,
                                     plot_polygons_one_by_one=False, title=''):
    if polygons is None:
        return False

    polygons_each_against = \
        [(polygons[i], np.concatenate([polygons[:i], polygons[i + 1:]])) for i in range(len(polygons))]

    intersections = []

    square_no = 1
    for ii, not_ii in polygons_each_against:

        #  If ii intersects with the static obstacle, return True for intersection
        if obstacle is not None:
            if obstacle.intersects(ii):
                intersection = True
                other_squares_fixed = None
            else:
                intersection = False

        if not intersection:
            other_polygons = MultiPolygon([jj for jj in not_ii])
            other_squares_fixed = None

            try:
                # intersection = other_polygons.intersection(ii)
                intersection = other_polygons.intersects(ii)

            #     except TopologicalError as e:
            #         print(f"Caught a TopologicalError: {e}")
            except:
                #         print('Other squares are intersecting, so ignoring the collssion for this square for now.')
                other_squares_fixed = other_polygons.buffer(0)
                # intersection = other_squares_fixed.intersection(ii)
                intersection = other_squares_fixed.intersects(ii)

        #         intersection = False
        intersections.append(intersection)

        if plot_polygons_one_by_one:
            ax = plotcf.plot_obstacle(obstacle=obstacle, padding=0)
            plotcf.plot_intersecting_polygons_one_by_one(intersection=intersection, ii=ii, not_ii=not_ii,
                                                         other_squares_fixed=other_squares_fixed,
                                                         square_no=square_no, ax=ax)

            plotcf.plt.show()

        square_no = square_no + 1

    if plot_polygons:
        plotcf.plot_intersecting_polygons(intersections=intersections, polygons=polygons, title=title)
    return intersections


@jit(forceobj=True)
def intersecting_polygons(polygons=None, obstacle=None,
                          plot_polygons=False, check_for_agents=None,
                          plot_polygons_one_by_one=False, title=''):
    if polygons is None:
        print('polygons not passed!')
        return False

    if check_for_agents is None:
        polygons_each_against = \
            [(polygons[i], np.concatenate([polygons[:i], polygons[i + 1:]])) for i in range(len(polygons))]
    else:
        polygons_each_against = \
            [(polygons[i], np.concatenate([polygons[:i], polygons[i + 1:]])) for i in check_for_agents]

    intersections = [check_intersection_for_each_polygon(ii=ii, not_ii=not_ii, obstacle=obstacle)
                     for ii, not_ii in polygons_each_against]
    return intersections


def check_ego_collisions_as_time_rolls_with_plots(n_hulls_per_traj=None,
                                                  traj_hulls_j_not_j=None,
                                                  plot_intersections=False,
                                                  verbose_flag=False):
    # Function to check whether j collides with any other agent.
    # This function progressively checks each time instant for collisions.
    # If a collision is detected, the loop exists and returns a true value for collision.

    if n_hulls_per_traj is None:
        print('n_hulls_per_traj not passed!')
        return False

    if traj_hulls_j_not_j is None:
        print('traj_hulls_j_not_j not passed!')
        return False

    collision = False
    for tt in range(n_hulls_per_traj):
        intersections = intersecting_polygons(polygons=traj_hulls_j_not_j[:, tt], plot_polygons=False,
                                              plot_polygons_one_by_one=False, check_for_agents=[0],
                                              title=f'Timestep: {tt}')
        if verbose_flag: print(f'{intersections=}')
        if intersections[0]:
            if plot_intersections:
                plotcf.plot_intersecting_polygons(intersections=intersections,
                                                  polygons=traj_hulls_j_not_j[:, tt],
                                                  title=f'Timestep: {tt}')
            collision = True
            break

    return collision


# @jit(forceobj=True)  #  No visible improvement using @jit
def check_ego_collisions_as_time_rolls(n_hulls_per_traj=None,
                                       traj_hulls_j_not_j=None,
                                       plot_intersections=False,
                                       verbose_flag=False):
    # Function to check whether j collides with any other agent.
    # This function progressively checks each time instant for collisions.
    # If a collision is detected, the loop exists and returns a true value for collision.

    if n_hulls_per_traj is None:
        print('n_hulls_per_traj not passed!')
        return False

    if traj_hulls_j_not_j is None:
        print('traj_hulls_j_not_j not passed!')
        return False

    # The list comprehension method is slower
    # collision = any([intersecting_polygons(polygons=traj_hulls_j_not_j[:, tt], plot_polygons=False,
    #                                     plot_polygons_one_by_one=False, check_for_agents=[0])[0]
    #               for tt in range(n_hulls_per_traj)])

    # This method is faster because it skips some computations when a collision is detected.
    collision = False
    for tt in range(n_hulls_per_traj):
        intersections = intersecting_polygons(polygons=traj_hulls_j_not_j[:, tt], plot_polygons=False,
                                              plot_polygons_one_by_one=False, check_for_agents=[0])
        if intersections[0]:
            collision = True
            break

    return collision


def check_intersection_for_each_polygon(ii=None, not_ii=None, obstacle=None):
    if ii is None:
        print('ii is not passed!')
        return False

    if not_ii is None:
        print('not_ii is not passed!')
        return False

    #  If ii intersects with the static obstacle, return True for intersection
    if obstacle is not None:
        if obstacle.intersects(ii):
            return True

    other_polygons = MultiPolygon([jj for jj in not_ii])
    other_squares_fixed = None
    try:
        # intersection = other_polygons.intersection(ii)
        intersection = other_polygons.intersects(ii)

    #     except TopologicalError as e:
    #         print(f"Caught a TopologicalError: {e}")
    except:
        #         print('Other squares are intersecting, so ignoring the collision for this square for now.')
        other_squares_fixed = other_polygons.buffer(0)
        # intersection = other_squares_fixed.intersection(ii)
        intersection = other_squares_fixed.intersects(ii)

    return intersection


def get_collision_free_trajs(trajs_hulls, trajs_boxes, plot_all_timesteps=False,
                             x=None, y=None, vx=None, vy=None,
                             obstacle=None,
                             plot_polygons_one_by_one=False,
                             plot_colliding_timesteps=False,
                             save_plot_colliding_timesteps=False,
                             verbose_flag=False):
    # Function to check for colliding hulls at each timestep and
    # to resolve colliding trajectories.

    # The following code simultaneously checks whether trajs_hulls is an ndarray
    # and also provides the dimensions.
    try:
        n_traj, n_hulls_per_traj = trajs_hulls.shape
    except AttributeError:
        trajs_hulls = np.array(trajs_hulls, dtype=object)
        n_traj, n_hulls_per_traj = trajs_hulls.shape

    # The following code simultaneously checks whether trajs_boxes is an ndarray
    # and also provides the dimensions.
    try:
        n_traj_boxes, n_boxes_per_traj = trajs_boxes.shape
    except AttributeError:
        trajs_boxes = np.array(trajs_boxes, dtype=object)
        n_traj_boxes, n_boxes_per_traj = trajs_boxes.shape

    if n_traj != n_traj_boxes:
        print('There is a mismatch in the number of agents in the trajs_hulls and trajs_boxes')
        return False

    if n_hulls_per_traj != (n_boxes_per_traj - 1):
        print('There is a mismatch in the number of hulls per trajectory and number of boxes per trajectory')
        print('n_hulls_per_traj should be (n_boxes_per_traj-1)')
        print(f'Current values: {n_hulls_per_traj=}, {n_boxes_per_traj=}')
        return False

    # To make sure that the original array is not modified
    trajs_boxes = trajs_boxes.copy()
    trajs_hulls = trajs_hulls.copy()

    collision_detected = False

    for ii in range(n_hulls_per_traj):
        colliding_trajs = ['Non Empty Start']

        while len(colliding_trajs) > 0:
            if plot_polygons_one_by_one:
                intersections = intersecting_polygons_with_plots(polygons=trajs_hulls[:, ii],
                                                                 obstacle=obstacle,
                                                                 plot_polygons=plot_all_timesteps,
                                                                 plot_polygons_one_by_one=plot_polygons_one_by_one,
                                                                 title=f'Timestep: {ii}')
            else:
                intersections = intersecting_polygons(polygons=trajs_hulls[:, ii],
                                                      obstacle=obstacle,
                                                      plot_polygons=plot_all_timesteps,
                                                      title=f'Timestep: {ii}')

            colliding_trajs = [i for i, x in enumerate(intersections) if x]

            # To keep track whether any collisions were detected.
            if not collision_detected:
                if len(colliding_trajs) > 0:
                    collision_detected = True

            if verbose_flag: print(f'Timestep={ii}, {colliding_trajs=}')

            if (len(colliding_trajs) > 0) and plot_colliding_timesteps:
                ax_collision = plotcf.plot_hulls_for_trajs(trajs_hulls, colour='lightgrey')
                plotcf.plot_obstacle(obstacle=obstacle, ax=ax_collision, padding=0);
                plotcf.plot_intersecting_polygons(intersections=intersections,
                                                  ax=ax_collision,
                                                  polygons=trajs_hulls[:, ii],
                                                  title=f'$\\Delta t$: {ii+1}')

                if save_plot_colliding_timesteps:
                    plotcf.save_figure(ax=ax_collision,
                                       filename=f"collision_delta_t_{ii}")

            if (x is None) or (y is None):
                trajs_hulls, trajs_boxes = stop_colliding_trajs(
                    trajs_hulls=trajs_hulls,
                    trajs_boxes=trajs_boxes,
                    time_step=ii,
                    colliding_trajs=colliding_trajs
                )
            else:

                trajs_hulls, trajs_boxes, x, y, vx, vy = stop_colliding_trajs(
                    trajs_hulls=trajs_hulls,
                    trajs_boxes=trajs_boxes,
                    time_step=ii,
                    colliding_trajs=colliding_trajs,
                    x=x,
                    y=y,
                    vx=vx,
                    vy=vy
                )

    if verbose_flag:
        if collision_detected:
            print('Collision detected!')
        else:
            print('No collisions!')

    if (x is None) or (y is None):
        return trajs_hulls, trajs_boxes
    else:
        return trajs_hulls, trajs_boxes, x, y, vx, vy


# @jit(forceobj=True)
def stop_colliding_trajs(
        trajs_hulls=None,
        trajs_boxes=None,
        time_step=None,
        colliding_trajs=[],
        x=None, y=None,
        vx=None, vy=None):
    if trajs_hulls is None:
        print('No trajs_hulls passed!')
        return False

    if trajs_boxes is None:
        print('No trajs_boxes passed!')
        return False

    if time_step is None:
        print('No time_step passed!')
        return False

    if colliding_trajs is []:
        print('No colliding trajectories.')
        return trajs_hulls, trajs_boxes

    # To make sure that the original array is not modified
    trajs_boxes = trajs_boxes.copy()
    trajs_hulls = trajs_hulls.copy()

    for ii in colliding_trajs:
        #  Update trajs_boxes after timestep to the traj_box at timestep
        trajs_boxes[ii, time_step:] = trajs_boxes[ii, time_step]

        #  Update trajs_boxes after timestep to the traj_box at timestep
        trajs_hulls[ii, time_step:] = trajs_boxes[ii, time_step]

    if x is None or y is None:
        return trajs_hulls, trajs_boxes
    else:
        for ii in colliding_trajs:
            #  Update x after timestep to the x at timestep
            x[ii, time_step:] = x[ii, time_step]
            #  Update y after timestep to the y at timestep
            y[ii, time_step:] = y[ii, time_step]

            if vx is not None:
                #  Update vx after timestep to be zero
                vx[ii, time_step:] = 0
            if vy is not None:
                #  Update vy after timestep to be zero
                vy[ii, time_step:] = 0
        return trajs_hulls, trajs_boxes, x, y, vx, vy


def social_force_acceleration(x, y, k=1.0, threshold_distance=5.0, buffer=2, threshold_a=5.0):
    distances = np.sqrt((x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2)

    # Avoid division by zero
    distances[distances == 0] = 1e-5

    distances[distances < buffer] = buffer + EPS

    # Calculate the inverse square distance
    inv_sq_distances = 1.0 / ((distances - buffer) ** 2)

    # Mask distances greater than threshold
    inv_sq_distances[distances > threshold_distance] = 0.0

    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y

    r_x = dx / distances
    r_y = dy / distances

    # Compute social force accelerations
    a_social_x = k * r_x * inv_sq_distances
    a_social_y = k * r_y * inv_sq_distances

    a, theta = get_a_and_theta(ax=a_social_x, ay=a_social_y)

    # Clip the acceleration to the maximum threshold value
    a = np.clip(a, a_min=0, a_max=threshold_a)

    # Recompute the accelerations based on the direction theta and clipped acceleration a
    a_social_x = a * np.cos(theta)
    a_social_y = a * np.sin(theta)

    np.fill_diagonal(a_social_x, 0)
    np.fill_diagonal(a_social_y, 0)

    return np.sum(a_social_x, axis=1), np.sum(a_social_y, axis=1), distances


def generate_safe_trajectories(n, duration, timesteps, x0, y0, v0x, v0y,
                               threshold_distance=5.0, threshold_a=5.0, threshold_velocity=5.0,
                               k=1.0,
                               buffer=1,
                               restore_factor=0.1):
    """
    Function to generate collision-free trajectories for pedestrians based on a social force model.

    Parameters
    ----------
    n : int
        Number of agents
    duration : int
        Total duration of trajectory in seconds
    timesteps : int
        Number of timesteps for the whole trajectory
    x0 : ndarray
        starting x-locations of the nn agents
    y0 : ndarray
        starting y-locations of the nn agents
    v0x : ndarray
        starting x-velocities of the nn agents
    v0y : ndarray
        starting y-velocities of the nn agents
    threshold_distance : float
        distance above which the social force drops to zero
    threshold_a : float
        maximum possible acceleration of the agents
    threshold_velocity : float
        maximum possible velocity of the agents
    k : float
        stiffness parameter of the social force model
    buffer : float
        distance at which the social forces should grow to infinity
    restore_factor : float
        factor by which the new velocity should match the intial velocity of the agents

    Returns
    -------
    x: ndarray
        x data for the generated trajectories for nn agents
    y: ndarray
        y data for the generated trajectories for nn agents
    vx : ndarray
        velocities along x-axis for the generated trajectories for nn agents
    vy : ndarray
        velocities along y-axis for the generated trajectories for nn agents
    a : ndarray
        magnitude of the acceleration for nn agents for the whole trajectory
    theta: ndarray
        direction of the acceleration for nn agents for the whole trajectory

    """
    time_per_step = duration / timesteps

    x = np.zeros((n, timesteps))
    y = np.zeros((n, timesteps))
    vx = np.zeros((n, timesteps))
    vy = np.zeros((n, timesteps))
    ax = np.zeros((n, timesteps - 1))
    ay = np.zeros((n, timesteps - 1))

    x[:, 0] = x0.flatten()
    y[:, 0] = y0.flatten()

    vx[:, 0] = v0x.flatten()
    vy[:, 0] = v0y.flatten()

    for t in range(1, timesteps):
        a_social_x, a_social_y, distances = social_force_acceleration(x[:, t - 1], y[:, t - 1], buffer=buffer, k=k,
                                                                      threshold_distance=threshold_distance,
                                                                      threshold_a=threshold_a)
        # Avoid diagonal elements (distance to itself)
        np.fill_diagonal(distances, np.inf)

        # Find the minimum distance for each agent
        min_distances = np.min(distances, axis=1)

        for i in range(n):
            # Update velocities based on social accelerations
            vx[i, t] += a_social_x[i]
            vy[i, t] += a_social_y[i]

            # Calculate the component to restore the initial velocity
            restore_x = v0x[i] - vx[i, t]
            restore_y = v0y[i] - vy[i, t]

            # Apply restoring accelerations
            vx[i, t] += restore_x * restore_factor
            vy[i, t] += restore_y * restore_factor

            # Get the magnitude and direction of velocity
            v, theta_v = get_a_and_theta(ax=vx[i, t], ay=vy[i, t])

            # Set the maximum possible velocity based on the distance to the closest other agent
            #  Considering the buffer so that there is no collision
            v_max = min((min_distances[i] / 2 - buffer / 2) / time_per_step, threshold_velocity)
            v_ = np.clip(v, a_min=0, a_max=v_max)

            # Update the new safe velocity
            vx[i, t] = v_ * np.cos(theta_v)
            vy[i, t] = v_ * np.sin(theta_v)

            # Final accelerations
            ax[i, t - 1] = (vx[i, t] - vx[i, t - 1]) / time_per_step
            ay[i, t - 1] = (vy[i, t] - vy[i, t - 1]) / time_per_step

            a, theta = get_a_and_theta(ax=ax, ay=ay)

            # Update positions using velocity
            x[i, t] = x[i, t - 1] + vx[i, t] * time_per_step
            y[i, t] = y[i, t - 1] + vy[i, t] * time_per_step

    return x, y, vx, vy, a, theta


def generate_trajectories(n, duration, timesteps, x0, y0, v0x, v0y,
                          k=1.0, threshold_distance=5.0, buffer=1,
                          restore_factor=0.1):
    time_per_step = duration / timesteps

    x = np.zeros((n, timesteps))
    y = np.zeros((n, timesteps))
    vx = np.zeros((n, timesteps))
    vy = np.zeros((n, timesteps))
    ax = np.zeros((n, timesteps - 1))
    ay = np.zeros((n, timesteps - 1))

    x[:, 0] = x0.flatten()
    y[:, 0] = y0.flatten()

    vx[:, 0] = v0x.flatten()
    vy[:, 0] = v0y.flatten()

    for t in range(1, timesteps):
        a_social_x, a_social_y, _ = social_force_acceleration(x[:, t - 1], y[:, t - 1], buffer=buffer,
                                                              k=k, threshold_distance=threshold_distance)

        for i in range(n):
            # Update velocities based on social accelerations
            vx[i, t] += a_social_x[i]
            vy[i, t] += a_social_y[i]

            # Calculate the component to restore the initial velocity
            restore_x = v0x[i] - vx[i, t]
            restore_y = v0y[i] - vy[i, t]

            # Apply restoring accelerations
            vx[i, t] += restore_x * restore_factor
            vy[i, t] += restore_y * restore_factor

            # Final accelerations
            ax[i, t - 1] = (vx[i, t] - vx[i, t - 1]) / time_per_step
            ay[i, t - 1] = (vy[i, t] - vy[i, t - 1]) / time_per_step

            a, theta = get_a_and_theta(ax=ax, ay=ay)

            # Update positions using velocity
            x[i, t] = x[i, t - 1] + vx[i, t] * time_per_step
            y[i, t] = y[i, t - 1] + vy[i, t] * time_per_step

    return x, y, vx, vy, a, theta


def get_mdr(x=None, y=None, v0x=None, v0y=None, time_per_step=None,
            threshold_distance=100.0, threshold_velocity=1.0, threshold_a=2.5,
            k=1000.0, buffer=2.0, restore_factor=0.05, verbose=False
            ):
    x = x.flatten()
    y = y.flatten()
    v0x = v0x.flatten()
    v0y = v0y.flatten()

    # vx = np.zeros((nn, 1))
    # vy = np.zeros((nn, 1))
    # ax = np.zeros((nn, 1))
    # ay = np.zeros((nn, 1))

    a_social_x, a_social_y, distances = social_force_acceleration(x=x, y=y,
                                                                  buffer=buffer, k=k,
                                                                  threshold_distance=threshold_distance,
                                                                  threshold_a=threshold_a)
    if verbose:
        print(f'{a_social_x=}')
        print(f'{a_social_y=}')

    # Avoid diagonal elements (distance to itself)
    np.fill_diagonal(distances, np.inf)

    # Find the minimum distance for each agent
    min_distances = np.min(distances, axis=1)

    # for i in range(nn):
    # Update velocities based on social accelerations
    vx = a_social_x * time_per_step
    vy = a_social_y * time_per_step

    # Calculate the component to restore the initial velocity
    delta_restore_vx = v0x - vx
    delta_restore_vy = v0y - vy

    # Apply restoring accelerations
    ax = a_social_x + delta_restore_vx * restore_factor
    ay = a_social_y + delta_restore_vy * restore_factor

    vx = v0x + ax * time_per_step
    vy = v0y + ay * time_per_step

    # Get the magnitude and direction of velocity and accleration
    v, theta_v = get_a_and_theta(ax=vx, ay=vy)
    a, theta_a = get_a_and_theta(ax=ax, ay=ay)

    if verbose:
        print(f'{v=}')
        print(f'{theta_v=}')

        print(f'{a=}')
        print(f'{theta_a=}')

    # Set the maximum possible velocity based on the distance to the closest other agent
    #  Considering the buffer so that there is no collision
    v_max = np.minimum((min_distances / 2 - buffer / 2) / time_per_step, threshold_velocity)
    v_ = np.clip(v, a_min=0, a_max=v_max)

    # Update the new safe velocity
    vx = v_ * np.cos(theta_v)
    vy = v_ * np.sin(theta_v)

    # Final accelerations
    ax = (vx - v0x) / time_per_step
    ay = (vy - v0y) / time_per_step

    # a, theta = get_a_and_theta(ax=ax, ay=ay)

    a = np.sqrt(ax ** 2 + ay ** 2)  # Getting acceleration magnitude based on the thresholded velocity
    theta = theta_a  # Using the direction of acceleration before the thresholding

    if verbose:
        print(f'{a=}')
        print(f'{theta=}')
        print(f'{theta_a=}')

    return a[:, np.newaxis], theta[:, np.newaxis]


def get_mdr_target_lane_eventually(x=None, y=None, v0x=None, v0y=None, time_per_step=None,
                                   vx_desired=None, vy_desired=None, k_v=0.5,
                                   threshold_distance=100.0, threshold_velocity=1.0, threshold_a=2.5,
                                   k=1000.0, buffer=2.0, restore_factor=0.05, verbose=False,
                                   lanes=None, w_a_lane=EPS, w_lane=0.5
                                   ):
    if vx_desired is None:
        vx_desired = v0x
    if vy_desired is None:
        vy_desired = v0y

    x = x.flatten()
    y = y.flatten()
    v0x = v0x.flatten()
    v0y = v0y.flatten()
    vx_desired = vx_desired.flatten()
    vy_desired = vy_desired.flatten()

    # vx = np.zeros((nn, 1))
    # vy = np.zeros((nn, 1))
    # ax = np.zeros((nn, 1))
    # ay = np.zeros((nn, 1))

    a_social_x, a_social_y, distances = social_force_acceleration(x=x, y=y,
                                                                  buffer=buffer, k=k,
                                                                  threshold_distance=threshold_distance,
                                                                  threshold_a=threshold_a)
    if verbose:
        print(f'{a_social_x=}')
        print(f'{a_social_y=}')

    # Avoid diagonal elements (distance to itself)
    np.fill_diagonal(distances, np.inf)

    # Find the minimum distance for each agent
    min_distances = np.min(distances, axis=1)

    # for i in range(nn):
    # Update velocities based on social accelerations
    vx = a_social_x * time_per_step
    vy = a_social_y * time_per_step

    # Calculate the component to restore the initial velocity
    delta_restore_vx = v0x - vx
    delta_restore_vy = v0y - vy

    delta_desired_vx = vx_desired - v0x
    delta_desired_vy = vy_desired - v0y

    # Apply restoring accelerations
    ax = a_social_x + delta_restore_vx * restore_factor + k_v * delta_desired_vx
    ay = a_social_y + delta_restore_vy * restore_factor + k_v * delta_desired_vy

    vx = v0x + ax * time_per_step
    vy = v0y + ay * time_per_step

    # Get the magnitude and direction of velocity and accleration
    v, theta_v = get_a_and_theta(ax=vx, ay=vy)
    a, theta_a = get_a_and_theta(ax=ax, ay=ay)

    if verbose:
        print(f'{v=}')
        print(f'{theta_v=}')

        print(f'{a=}')
        print(f'{theta_a=}')

    # Set the maximum possible velocity based on the distance to the closest other agent
    #  Considering the buffer so that there is no collision
    v_max = np.minimum((min_distances / 2 - buffer / 2) / time_per_step, threshold_velocity)
    v_ = np.clip(v, a_min=0, a_max=v_max)

    # Update the new safe velocity
    vx = v_ * np.cos(theta_v)
    vy = v_ * np.sin(theta_v)

    # Final accelerations
    ax = (vx - v0x) / time_per_step
    ay = (vy - v0y) / time_per_step

    a, theta = get_a_and_theta(ax=ax, ay=ay)

    # a = np.sqrt(ax ** 2 + ay ** 2)  # Getting acceleration magnitude based on the thresholded velocity
    # theta = theta_a  # Using the direction of acceleration before the thresholding

    if verbose:
        print(f'{a=}')
        print(f'{theta=}')
        print(f'{theta_a=}')

    if lanes is not None:
        for ii in range(len(x)):
            if lanes[ii] is not None:
                print(lanes[ii])
                a_lane, theta_lane = accelerate_2_lane_eventually(x0=x[ii], y0=y[ii], v0x=v0x[ii], v0y=v0y[ii],
                                                                  w_a=w_a_lane,
                                                                  t=time_per_step, lane_polynomial_str=lanes[ii],
                                                                  bounds=[(EPS, a[ii]), (-np.pi, np.pi)],
                                                                  initial_guess=[a[ii], theta[ii]])

                a[ii] = (a[ii] + w_lane * a_lane) / (1 + w_lane)
                theta[ii] = (theta[ii] + w_lane * theta_lane) / (1 + w_lane)

                print(f'{a[ii]=}, {a_lane=}')
                print(f'{theta[ii]=}, {theta_lane=}')

        print(f'{a=}')
        print(f'{theta=}')

    return a[:, np.newaxis], theta[:, np.newaxis]


def get_mdr_target(x=None, y=None, v0x=None, v0y=None, time_per_step=None,
                   vx_desired=None, vy_desired=None, k_v=0.5,
                   threshold_distance=100.0, threshold_velocity=1.0, threshold_a=2.5,
                   k=1000.0, buffer=2.0, restore_factor=0.05, verbose=False,
                   lanes=None, k_lane=0.5, lane_plots=False
                   ):
    if vx_desired is None:
        vx_desired = v0x
    if vy_desired is None:
        vy_desired = v0y

    x = x.flatten()
    y = y.flatten()
    v0x = v0x.flatten()
    v0y = v0y.flatten()
    vx_desired = vx_desired.flatten()
    vy_desired = vy_desired.flatten()

    # vx = np.zeros((nn, 1))
    # vy = np.zeros((nn, 1))
    # ax = np.zeros((nn, 1))
    # ay = np.zeros((nn, 1))

    a_social_x, a_social_y, distances = social_force_acceleration(x=x, y=y,
                                                                  buffer=buffer, k=k,
                                                                  threshold_distance=threshold_distance,
                                                                  threshold_a=threshold_a)
    if verbose:
        print(f'{a_social_x=}')
        print(f'{a_social_y=}')

    # Avoid diagonal elements (distance to itself)
    np.fill_diagonal(distances, np.inf)

    # Find the minimum distance for each agent
    min_distances = np.min(distances, axis=1)

    # for i in range(nn):
    # Update velocities based on social accelerations
    vx = a_social_x * time_per_step
    vy = a_social_y * time_per_step

    # Calculate the component to restore the initial velocity
    delta_restore_vx = v0x - vx
    delta_restore_vy = v0y - vy

    delta_desired_vx = vx_desired - v0x
    delta_desired_vy = vy_desired - v0y

    # Apply restoring accelerations
    ax = a_social_x + delta_restore_vx * restore_factor + k_v * delta_desired_vx
    ay = a_social_y + delta_restore_vy * restore_factor + k_v * delta_desired_vy

    vx = v0x + ax * time_per_step
    vy = v0y + ay * time_per_step

    # Get the magnitude and direction of velocity and accleration
    v, theta_v = get_a_and_theta(ax=vx, ay=vy)
    a, theta_a = get_a_and_theta(ax=ax, ay=ay)

    if verbose:
        print(f'{v=}')
        print(f'{theta_v=}')

        print(f'{a=}')
        print(f'{theta_a=}')

    # Set the maximum possible velocity based on the distance to the closest other agent
    #  Considering the buffer so that there is no collision
    v_max = np.minimum((min_distances / 2 - buffer / 2) / time_per_step, threshold_velocity)
    v_ = np.clip(v, a_min=0, a_max=v_max)

    # Update the new safe velocity
    vx = v_ * np.cos(theta_v)
    vy = v_ * np.sin(theta_v)

    # Final accelerations
    ax = (vx - v0x) / time_per_step
    ay = (vy - v0y) / time_per_step

    # a, theta = get_a_and_theta(ax=ax, ay=ay)

    a = np.sqrt(ax ** 2 + ay ** 2)  # Getting acceleration magnitude based on the thresholded velocity
    theta = theta_a  # Using the direction of acceleration before the thresholding

    if verbose:
        print(f'{a=}')
        print(f'{theta=}')
        print(f'{theta_a=}')

    if lanes is not None:
        for ii in range(len(x)):
            if lanes[ii] is not None:
                print(lanes[ii])
                a_theta_lane = accelerate_2_lane(x0=x[ii], y0=y[ii], v0x=v0x[ii], v0y=v0y[ii], plots=lane_plots,
                                                 t=time_per_step, lane_polynomial_str=lanes[ii])
                if a_theta_lane:
                    # accelerate_2_lane returns False if there is no lane_polynomial_str
                    a_lane, theta_lane = a_theta_lane

                    ax_ii = a[ii] * np.cos(theta[ii])
                    ay_ii = a[ii] * np.sin(theta[ii])

                    ax_lane = a_lane * np.cos(theta_lane)
                    ay_lane = a_lane * np.sin(theta_lane)

                    ax_ii = (ax_ii + k_lane * ax_lane) / (1 + k_lane)
                    ay_ii = (ay_ii + k_lane * ay_lane) / (1 + k_lane)

                    a[ii], theta[ii] = get_a_and_theta(ax=ax_ii, ay=ay_ii)

                print(f'{a[ii]=}, {a_lane=}')
                print(f'{theta[ii]=}, {theta_lane=}')

        print(f'{a=}')
        print(f'{theta=}')

    return a[:, np.newaxis], theta[:, np.newaxis]


def swap_traj_boxes_in_trajs(
        trajs_hulls=None,
        trajs_boxes=None,
        ii=None,
        new_traj_boxes=[]):
    if trajs_hulls is None:
        print('No trajs_hulls passed!')
        return False

    if trajs_boxes is None:
        print('No trajs_boxes passed!')
        return False

    if ii is None:
        print('No agent id passed!')
        return False

    if new_traj_boxes is []:
        print('No new trajectory passed!')

    # To make sure that the original array is not modified
    trajs_boxes = trajs_boxes.copy()
    trajs_hulls = trajs_hulls.copy()

    old_traj_shape = trajs_boxes[ii, :].shape
    new_traj_shape = new_traj_boxes.shape

    if old_traj_shape != new_traj_shape:
        print('Dimensions of the new trajectory do not match the old trajectory.')
        print(f'{old_traj_shape=}')
        print(f'{new_traj_shape=}')
        return False
    else:
        trajs_boxes[ii, :] = new_traj_boxes
        new_traj_hulls = np.array(get_traj_hulls(new_traj_boxes))
        trajs_hulls[ii, :] = new_traj_hulls

    return trajs_hulls, trajs_boxes


# @jit(forceobj=True)
def swap_agent_traj_in_trajs(
        trajs_hulls=None, trajs_boxes=None,
        x0=None, y0=None, v0x=None, v0y=None,
        traj2swap=None, a2swap=None, theta2swap=None,
        t_action=None, n_timesteps=None, l=None,
        plot_new_traj=False, ax=None):
    if trajs_hulls is None:
        print('No trajs_hulls passed!')
        return False

    if trajs_boxes is None:
        print('No trajs_boxes passed!')
        return False

    if traj2swap is None:
        print('No agent id passed!')
        return False

    if x0 is None:
        print('No x0 (initial locations x) passed!')
        return False

    if y0 is None:
        print('No x0 (initial locations y) passed!')
        return False

    if v0x is None:
        print('No v0x (initial velocities x) passed!')
        return False

    if v0y is None:
        print('No v0y (initial velocities y) passed!')
        return False

    if a2swap is None:
        print('No acceleration passed!')
        return False

    if theta2swap is None:
        print('No theta passed!')
        return False

    if t_action is None:
        print('No t_action passed!')
        return False

    if n_timesteps is None:
        print('No n_timesteps passed!')
        return False

    if l is None:
        print('No l (dimensions of the agent) passed!')
        return False

    # To make sure that the original array is not modified
    trajs_boxes = trajs_boxes.copy()
    trajs_hulls = trajs_hulls.copy()

    # a2swap.reshape((-1,1))
    # theta2swap.reshape((-1,1))

    # print("x0:", x0[traj2swap].shape)
    # print("y0:", y0[traj2swap].shape)
    # print("v0x:", v0x[traj2swap].shape)
    # print("v0y:", v0y[traj2swap].shape)
    # print("a2swap:", a2swap.shape)
    # print("theta2swap:", theta2swap.shape)

    x_, y_, _ = make_trajectories(x0=x0[traj2swap], y0=y0[traj2swap],
                                  v0x=v0x[traj2swap], v0y=v0y[traj2swap],
                                  a=a2swap, theta=theta2swap,
                                  t_action=t_action, n_timesteps=n_timesteps)

    if plot_new_traj:
        plotcf.plot_traj_boxes_from_xy(x=x_, y=y_, lx=1, ly=1, ax=ax)

    if isinstance(traj2swap, int):
        # print(f'Int: {traj2swap=}')
        new_traj_boxes = get_boxes_for_trajs(x_, y_, lx=l, ly=l).squeeze()

        trajs_hulls, trajs_boxes = swap_traj_boxes_in_trajs(trajs_hulls=trajs_hulls,
                                                            trajs_boxes=trajs_boxes,
                                                            ii=traj2swap, new_traj_boxes=new_traj_boxes)

    else:  # If there traj2swap is an array or a list of mulitiple agents
        # print(f'List : {traj2swap=}')
        new_traj_boxes = get_boxes_for_trajs(x_, y_, lx=l, ly=l)

        for ii in range(len(traj2swap)):
            trajs_hulls, trajs_boxes = swap_traj_boxes_in_trajs(trajs_hulls=trajs_hulls,
                                                                trajs_boxes=trajs_boxes,
                                                                ii=traj2swap[ii], new_traj_boxes=new_traj_boxes[ii, :])
    return trajs_hulls, trajs_boxes


@lru_cache(maxsize=256)
def get_other_agents(n=None, ii=None):
    if n is None:
        print('Number of agents (nn) not passed!')
        return

    if ii is None:
        print('Agent of interest (ii) not passed!')
        return

    agents = range(n)
    others = np.concatenate([agents[:ii], agents[ii + 1:]]).astype(int)

    return others


def get_feasible_action_space(
        actor=None, move=None, affected=None, a_min=-1, a_max=1, a_num=1,
        theta_min=-np.pi / 2, theta_max=np.pi / 2, theta_num=8,
        trajs_hulls=None, trajs_boxes=None, obstacle=None,
        x0=None, y0=None, v0x=None, v0y=None,
        t_action=None, n_timesteps=None, l=None, verbose_flag=False,
        plot_subspaces=False, plot_feasible_action_space=False, strip_feasible_action_space_plots=False,
        show_legend=True, legend_inside=False, finer=True,
        return_feasibility=False,
        plot_intersections=False):
    # ------------------------ Checks ------------------------------ #

    if actor is None:
        print('No actor agent passed!')
        return False

    if affected is None:
        print('No affected agent passed!')

    if trajs_hulls is None:
        print('No trajs_hulls passed!')
        return False

    if trajs_boxes is None:
        print('No trajs_boxes passed!')
        return False

    if x0 is None:
        print('No x0 (initial locations x) passed!')
        return False

    if y0 is None:
        print('No x0 (initial locations y) passed!')
        return False

    if v0x is None:
        print('No v0x (initial velocities x) passed!')
        return False

    if v0y is None:
        print('No v0y (initial velocities y) passed!')
        return False

    if t_action is None:
        print('No t_action passed!')
        return False

    if n_timesteps is None:
        print('No n_timesteps passed!')
        return False

    if l is None:
        print('No l (dimensions of the agent) passed!')
        return False

    # The following code simultaneously checks whether trajs_hulls is an ndarray
    # and also provides the dimensions.
    try:
        n_traj, n_hulls_per_traj = trajs_hulls.shape
    except AttributeError:
        trajs_hulls = np.array(trajs_hulls, dtype=object)
        n_traj, n_hulls_per_traj = trajs_hulls.shape

    # The following code simultaneously checks whether trajs_boxes is an ndarray
    # and also provides the dimensions.
    try:
        n_traj_boxes, n_boxes_per_traj = trajs_boxes.shape
    except AttributeError:
        trajs_boxes = np.array(trajs_boxes, dtype=object)
        n_traj_boxes, n_boxes_per_traj = trajs_boxes.shape

    if n_traj != n_traj_boxes:
        print('There is a mismatch in the number of agents in the trajs_hulls and trajs_boxes')
        return False

    if n_hulls_per_traj != (n_boxes_per_traj - 1):
        print('There is a mismatch in the number of hulls per trajectory and number of boxes per trajectory')
        print('n_hulls_per_traj should be (n_boxes_per_traj-1)')
        print(f'Current values: {n_hulls_per_traj=}, {n_boxes_per_traj=}')
        return False

    # ----------------------------------------------------------- #

    # Swap actor action to move passed to get new trajs_boxes and trajs_hulls
    if move is not None:
        mdr_string = ' MdR | '
        trajs_hulls, trajs_boxes = swap_agent_traj_in_trajs(trajs_hulls=trajs_hulls, trajs_boxes=trajs_boxes,
                                                            x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                                            traj2swap=actor,
                                                            a2swap=move[:, 0].reshape((-1, 1)),
                                                            theta2swap=move[:, 1].reshape((-1, 1)),
                                                            t_action=t_action, n_timesteps=n_timesteps,
                                                            l=l, plot_new_traj=False)

    else:
        mdr_string = ' MdR | '
        # To make sure that the original array is not modified
        trajs_boxes = trajs_boxes.copy()
        trajs_hulls = trajs_hulls.copy()

    # Get list of agents other than the affected
    not_affected = get_other_agents(n=n_traj, ii=affected)

    # Get traj_hulls and trajs_boxes for not_affected
    trajs_hulls_not_j = trajs_hulls[not_affected]
    trajs_boxes_not_j = trajs_boxes[not_affected]

    # # Resolve collisions if actors move was updated.
    # if move is not None:
    #     trajs_hulls_not_j, trajs_boxes_not_j = get_collision_free_trajs(trajs_hulls_not_j, trajs_boxes_not_j,
    #                                                                     obstacle=obstacle,
    #                                                                     plot_polygons_one_by_one=False)

    # Resolve collisions of all agents except the affected
    trajs_hulls_not_j, trajs_boxes_not_j = get_collision_free_trajs(trajs_hulls_not_j, trajs_boxes_not_j,
                                                                    obstacle=obstacle,
                                                                    plot_polygons_one_by_one=False)

    a_js = np.linspace(a_min, a_max, a_num + 1)
    theta_js = np.linspace(theta_min, theta_max, theta_num + 1)

    x0_j = x0[affected]
    y0_j = y0[affected]
    v0x_j = v0x[affected]
    v0y_j = v0y[affected]

    if plot_feasible_action_space:
        fig, configuration_space_ax = plotcf.get_new_fig(122)
        action_space_ax = fig.add_subplot(121)
        fig.tight_layout(pad=5.0)

        _, action_space_ax_stripped = plotcf.get_new_fig()
        _, configuration_space_ax_stripped = plotcf.get_new_fig()

    else:
        configuration_space_ax = None
        action_space_ax = None

        action_space_ax_stripped = None
        configuration_space_ax_stripped = None

    count = 0

    aa_s = np.array(range(len(a_js) - 1))
    thetaa_s = np.array(range(len(theta_js) - 1))

    # feasibility = [get_feasibility_of_subspace_faster(aa=aa, thetaa=thetaa,
    #                                                   a_js=a_js, theta_js=theta_js,
    #                                                   actor=actor,
    #                                                   affected=affected,
    #                                                   not_affected=not_affected,
    #                                                   x0_j=x0_j, y0_j=y0_j, v0x_j=v0x_j, v0y_j=v0y_j,
    #                                                   n_hulls_per_traj=n_hulls_per_traj,
    #                                                   trajs_hulls_not_j=trajs_hulls_not_j,
    #                                                   n_timesteps=n_timesteps, t_action=t_action, l=l,
    #                                                   action_space_ax=action_space_ax,
    #                                                   configuration_space_ax=configuration_space_ax,
    #                                                   plot_feasible_action_space=plot_feasible_action_space,
    #                                                   plot_intersections=plot_intersections,
    #                                                   plot_subspaces=plot_subspaces,
    #                                                   verbose_flag=verbose_flag)
    #                for aa in aa_s
    #                for thetaa in thetaa_s]

    feasibility = [get_feasibility_of_subspace_faster(aa=aa, thetaa=thetaa,
                                                      a_js=a_js, theta_js=theta_js,
                                                      actor=actor,
                                                      affected=affected,
                                                      not_affected=not_affected,
                                                      x0_j=x0_j, y0_j=y0_j, v0x_j=v0x_j, v0y_j=v0y_j,
                                                      n_hulls_per_traj=n_hulls_per_traj,
                                                      trajs_hulls_not_j=trajs_hulls_not_j,
                                                      obstacle=obstacle,
                                                      n_timesteps=n_timesteps, t_action=t_action, l=l,
                                                      action_space_ax=action_space_ax,
                                                      configuration_space_ax=configuration_space_ax,
                                                      action_space_ax_stripped=action_space_ax_stripped,
                                                      configuration_space_ax_stripped=configuration_space_ax_stripped,
                                                      plot_feasible_action_space=False,
                                                      plot_intersections=plot_intersections,
                                                      plot_subspaces=plot_subspaces,
                                                      verbose_flag=verbose_flag)
                   for aa in aa_s
                   for thetaa in thetaa_s]

    # aa_thetaa_list = [(aa, thetaa)
    #                   for aa in aa_s
    #                   for thetaa in thetaa_s]

    count = feasibility.count(True)

    if plot_feasible_action_space:
        plot_feasible_action_and_configuration_space(a_min=a_min, a_max=a_max, a_num=a_num,
                                                     theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                                                     feasibility=feasibility,
                                                     action_space_ax=action_space_ax,
                                                     action_space_ax_stripped=action_space_ax_stripped,
                                                     configuration_space_ax=configuration_space_ax,
                                                     configuration_space_ax_stripped=configuration_space_ax_stripped,
                                                     strip_feasible_action_space_plots=strip_feasible_action_space_plots,
                                                     show_legend=show_legend, legend_inside=legend_inside, finer=finer,
                                                     l=l, n_timesteps=n_timesteps,
                                                     obstacle=obstacle,
                                                     t_action=t_action,
                                                     actor=actor, affected=affected,
                                                     not_affected=not_affected,
                                                     trajs_boxes=trajs_boxes,
                                                     trajs_boxes_not_j=trajs_boxes_not_j,
                                                     trajs_hulls_not_j=trajs_hulls_not_j,
                                                     v0x_j=v0x_j, v0y_j=v0y_j, x0_j=x0_j, y0_j=y0_j,
                                                     mdr_string=mdr_string)
    if return_feasibility:
        # return count, feasibility, aa_thetaa_list
        return count, feasibility

    return count


def plot_feasible_action_space_for_actor(feasibility=None, actor=None,
                                         a_min=None, a_max=None, a_num=None,
                                         theta_min=None, theta_max=None, theta_num=None,
                                         mdr_a=None, mdr_theta=None,
                                         n_timesteps=None, t_action=None, nn=None, l=None,
                                         trajs_boxes=None, trajs_hulls=None, obstacle=None,
                                         v0x=None, v0y=None, x0=None, y0=None,
                                         save_figs=False, feasible_space_filename='',
                                         show_legend=True, legend_inside=True, finer=False,
                                         ):
    """
    Plot the Feasible Action-Space Reduction values of one actor agent on all other affected agents
    within a spatial interaction over a time-window.

    Parameters
    ----------
    feasibility : dict
        The dict with feasibility information of the subspaces.
    actor : int
        The actor agent identifier.
    a_min : float
        Lower limit of the magnitude of acceleration for the affected agent.
    a_max : float
        Upper limit of the magnitude of acceleration for the affected agent.
    a_num : int
        Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
    theta_min : float
        Lower limit of the angle for direction of acceleration for the affected agent.
    theta_max : float
        Upper limit of the angle for direction of acceleration for the affected agent.
    theta_num : int
        Number of intervals/subdivisions for direction of acceleration for the affected agent.
    mdr_a : ndarray
        Magnitude components of acceleration for the agents' Move de Rigueurs.
    mdr_theta : ndarray
        Direction component of acceleration for the agents' Move de Rigueurs.
    n_timesteps : int
        Number of timesteps to consider within the time window.
    t_action : float
        Time window to be considered.
    nn : int
        Number of agents.
    l : float
        Length of an agent along one axis (assuming same length in x and y directions).
    trajs_boxes : list
        Trajectory boxes for agents for each time instant.
    trajs_hulls : list
        Trajectory hulls for agents for each time interval (convex hulls of trajectory boxes).
    obstacle : MultiPolygon (optional)
        Static obstacles as a MultiPolygon
    v0x : ndarray
        Starting velocity in the x-direction for agents.
    v0y : ndarray
        Starting velocity in the y-direction for agents.
    x0 : ndarray
        Starting x locations for agents.
    y0 : ndarray
        Starting y locations for agents.
    save_figs : Bool, default=False
        Whether to save the feasible action space plots
    feasible_space_filename : str
        String to name the plots to be saved.
    show_legend: bool, optional
        Whether to show the legend
    legend_inside: bool, optional
        Whether to place the legend outside the figure axis or inside.
    finer: bool, optional
        If true, the fonts wont be resized to be bigger.


    Returns
    -------

    """

    others = get_other_agents(n=nn, ii=actor)
    print('actor=', str(actor), '\n others=', str(others))
    mdr = np.column_stack([mdr_a[actor].squeeze(), mdr_theta[actor].squeeze()])

    for affected in range(nn):
        if affected in others:
            plot_feasible_action_space_actor_affected(
                feasibility=feasibility[affected],
                actor=actor, affected=affected,
                a_min=a_min, a_max=a_max, a_num=a_num,
                theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                mdr=mdr, n_timesteps=n_timesteps, t_action=t_action, l=l,
                trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls, obstacle=obstacle,
                v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                show_legend=show_legend, legend_inside=legend_inside, finer=finer,
                save_figs=save_figs, feasible_space_filename=feasible_space_filename)
        else:
            plot_feasible_action_space_affected_affected(
                feasibility=feasibility[affected],
                affected=affected, mdr_a=mdr_a, mdr_theta=mdr_theta,
                a_min=a_min, a_max=a_max, a_num=a_num,
                theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                n_timesteps=n_timesteps, t_action=t_action, l=l,
                trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls, obstacle=obstacle,
                v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                show_legend=show_legend, legend_inside=legend_inside, finer=finer,
                save_figs=save_figs, feasible_space_filename=feasible_space_filename)

        # Re-enable garbage collection
        gc.enable()

        # Disable garbage collection
        gc.disable()


# @jit(forceobj=True)
def plot_feasible_action_space_for_all(nn=None,
                                       feasibility=None,
                                       mdr_a=None, mdr_theta=None,
                                       a_min=None, a_max=None, a_num=None,
                                       theta_min=None, theta_max=None, theta_num=None,
                                       trajs_boxes=None,
                                       trajs_hulls=None,
                                       obstacle=None,
                                       x0=None, y0=None, v0x=None, v0y=None,
                                       t_action=None, n_timesteps=None,
                                       l=None,
                                       save_figs=False, feasible_space_filename='',
                                       show_legend=True, legend_inside=True, finer=False
                                       ):
    for actor in range(nn):
        plot_feasible_action_space_for_actor(feasibility=feasibility[actor], actor=actor,
                                             a_min=a_min, a_max=a_max, a_num=a_num,
                                             theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                                             mdr_a=mdr_a, mdr_theta=mdr_theta,
                                             n_timesteps=n_timesteps, t_action=t_action, nn=nn, l=l,
                                             trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls, obstacle=obstacle,
                                             v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                             save_figs=save_figs, feasible_space_filename=feasible_space_filename,
                                             show_legend=show_legend, legend_inside=legend_inside, finer=finer,
                                             )
        if not save_figs:
            plotcf.plt.show()


@jit(forceobj=True)
def plot_feasible_action_space_actor_affected(
        feasibility=None,
        actor=None, affected=None,
        a_min=None, a_max=None, a_num=None,
        theta_min=None, theta_max=None, theta_num=None,
        mdr=None, n_timesteps=None, t_action=None, l=None,
        trajs_boxes=None, trajs_hulls=None, obstacle=None,
        v0x=None, v0y=None, x0=None, y0=None,
        save_figs=False, feasible_space_filename='',
        show_legend=True, legend_inside=True, finer=False):
    """
   Draw the Feasible Action-Space Reduction values of one actor agent on another affected agent
    within a spatial interaction over a time-window.

    Parameters
    ----------
    feasibility : dict
        The dict with feasibility information of the subspaces.
    actor : int
        The actor agent identifier.
    affected : int
        The affected agent identifier.
    a_min : float
        Lower limit of the magnitude of acceleration for the affected agent.
    a_max : float
        Upper limit of the magnitude of acceleration for the affected agent.
    a_num : int
        Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
    theta_min : float
        Lower limit of the angle for direction of acceleration for the affected agent.
    theta_max : float
        Upper limit of the angle for direction of acceleration for the affected agent.
    theta_num : int
        Number of intervals/subdivisions for direction of acceleration for the affected agent.
    mdr : list
        The move de rigueur of the actor agent (contains [magnitude, direction] of the constant acceleration move).
    n_timesteps : int
        Number of timesteps to consider within the time window.
    t_action : float
        Time window to be considered.
    l : float
        Length of an agent along one axis (assuming same length in x and y directions).
    trajs_boxes : list
        Trajectory boxes for agents for each time instant.
    trajs_hulls : list
        Trajectory hulls for agents for each time interval (convex hulls of trajectory boxes).
    obstacle : MultiPolygon (optional)
        Static obstacles as a MultiPolygon
    v0x : ndarray
        Starting velocity in the x-direction for agents.
    v0y : ndarray
        Starting velocity in the y-direction for agents.
    x0 : ndarray
        Starting x locations for agents.
    y0 : ndarray
        Starting y locations for agents.
    save_figs : Bool, default=False
        Whether to save the feasible action space plots
    feasible_space_filename : str
        String to name the plots to be saved.
    show_legend: bool, optional
        Whether to show the legend
    legend_inside: bool, optional
        Whether to place the legend outside the figure axis or inside.
    finer: bool, optional
        If true, the fonts wont be resized to be bigger.

    Returns
    -------

    """

    plot_feasible_action_space_with_agent_trajs(
        feasibility=feasibility['feasibility_move'],
        actor=actor,
        affected=affected,
        move=None,
        a_min=a_min, a_max=a_max, a_num=a_num,
        theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
        trajs_hulls=trajs_hulls,
        trajs_boxes=trajs_boxes,
        obstacle=obstacle,
        x0=x0, y0=y0, v0x=v0x, v0y=v0y,
        t_action=t_action,
        n_timesteps=n_timesteps,
        l=l,
        save_figs=save_figs,
        feasible_space_filename=feasible_space_filename + '_move',
        show_legend=show_legend, legend_inside=legend_inside, finer=finer

    )

    plot_feasible_action_space_with_agent_trajs(
        feasibility=feasibility['feasibility_mdr'],
        actor=actor,
        affected=affected,
        move=mdr,
        a_min=a_min, a_max=a_max, a_num=a_num,
        theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
        trajs_hulls=trajs_hulls,
        trajs_boxes=trajs_boxes,
        obstacle=obstacle,
        x0=x0, y0=y0, v0x=v0x, v0y=v0y,
        t_action=t_action,
        n_timesteps=n_timesteps,
        l=l,
        save_figs=save_figs,
        feasible_space_filename=feasible_space_filename + '_mdr',
        show_legend=show_legend, legend_inside=legend_inside, finer=finer
    )


@jit(forceobj=True)
def plot_feasible_action_space_affected_affected(
        feasibility=None,
        affected=None, nn=None, mdr_a=None, mdr_theta=None,
        a_min=None, a_max=None, a_num=None,
        theta_min=None, theta_max=None, theta_num=None,
        n_timesteps=None, t_action=None, l=None,
        trajs_boxes=None, trajs_hulls=None, obstacle=None,
        v0x=None, v0y=None, x0=None, y0=None,
        save_figs=False, feasible_space_filename='',
        show_legend=True, legend_inside=True, finer=False):
    """
    Draw the Feasible Action-Space Reduction values of one actor agent on another affected agent
    within a spatial interaction over a time-window.

    Parameters
    ----------
    feasibility : dict
        The dict with feasibility information of the subspaces.
    affected : int
        The affected agent identifier.
    a_min : float
        Lower limit of the magnitude of acceleration for the affected agent.
    a_max : float
        Upper limit of the magnitude of acceleration for the affected agent.
    a_num : int
        Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
    theta_min : float
        Lower limit of the angle for direction of acceleration for the affected agent.
    theta_max : float
        Upper limit of the angle for direction of acceleration for the affected agent.
    theta_num : int
        Number of intervals/subdivisions for direction of acceleration for the affected agent.
    mdr : list
        The move de rigueur of the actor agent (contains [magnitude, direction] of the constant acceleration move).
    n_timesteps : int
        Number of timesteps to consider within the time window.
    t_action : float
        Time window to be considered.
    l : float
        Length of an agent along one axis (assuming same length in x and y directions).
    trajs_boxes : list
        Trajectory boxes for agents for each time instant.
    trajs_hulls : list
        Trajectory hulls for agents for each time interval (convex hulls of trajectory boxes).
    obstacle : MultiPolygon (optional)
        Static obstacles as a MultiPolygon
    v0x : ndarray
        Starting velocity in the x-direction for agents.
    v0y : ndarray
        Starting velocity in the y-direction for agents.
    x0 : ndarray
        Starting x locations for agents.
    y0 : ndarray
        Starting y locations for agents.
    save_figs : Bool, default=False
        Whether to save the feasible action space plots
    feasible_space_filename : str
        String to name the plots to be saved.
    show_legend: bool, optional
        Whether to show the legend
    legend_inside: bool, optional
        Whether to place the legend outside the figure axis or inside.
    finer: bool, optional
        If true, the fonts wont be resized to be bigger.

    Returns
    -------
    fear_actor_to_affected : ndarray
        Feasible Action-Space Reduction values of the actor on the affected agent.
    """

    if nn is None:
        nn, _ = x0.shape

    actor = get_other_agents(n=nn, ii=affected)
    mdr = np.column_stack([mdr_a[actor].squeeze(), mdr_theta[actor].squeeze()])

    plot_feasible_action_space_with_agent_trajs(
        feasibility=feasibility['feasibility_move'],
        actor=actor,
        affected=affected,
        move=None,
        a_min=a_min, a_max=a_max, a_num=a_num,
        theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
        trajs_hulls=trajs_hulls,
        trajs_boxes=trajs_boxes,
        obstacle=obstacle,
        x0=x0, y0=y0, v0x=v0x, v0y=v0y,
        t_action=t_action,
        n_timesteps=n_timesteps,
        l=l,
        save_figs=save_figs,
        feasible_space_filename=feasible_space_filename + '_move',
        show_legend=show_legend, legend_inside=legend_inside, finer=finer
    )

    plot_feasible_action_space_with_agent_trajs(
        feasibility=feasibility['feasibility_mdr'],
        actor=actor,
        affected=affected,
        move=mdr,
        a_min=a_min, a_max=a_max, a_num=a_num,
        theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
        trajs_hulls=trajs_hulls,
        trajs_boxes=trajs_boxes,
        obstacle=obstacle,
        x0=x0, y0=y0, v0x=v0x, v0y=v0y,
        t_action=t_action,
        n_timesteps=n_timesteps,
        l=l,
        save_figs=save_figs,
        feasible_space_filename=feasible_space_filename + '_mdr',
        show_legend=show_legend, legend_inside=legend_inside, finer=finer
    )


def plot_feasible_action_space_with_agent_trajs(
        feasibility=None,
        actor=None,
        affected=None,
        move=None,
        a_min=-1, a_max=1, a_num=1,
        theta_min=-np.pi / 2, theta_max=np.pi / 2, theta_num=8,
        trajs_hulls=None, trajs_boxes=None, obstacle=None,
        x0=None, y0=None, v0x=None, v0y=None,
        t_action=None, n_timesteps=None, l=None,
        strip_feasible_action_space_plots=True,
        show_legend=True, legend_inside=True, finer=False,
        save_figs=False, feasible_space_filename=''
):
    # ------------------------ Checks ------------------------------ #

    if feasibility is None:
        print('Feasibility not passed!')
        return False

    if actor is None:
        print('No actor agent passed!')
        return False

    if affected is None:
        print('No affected agent passed!')

    if trajs_hulls is None:
        print('No trajs_hulls passed!')
        return False

    if trajs_boxes is None:
        print('No trajs_boxes passed!')
        return False

    if x0 is None:
        print('No x0 (initial locations x) passed!')
        return False

    if y0 is None:
        print('No x0 (initial locations y) passed!')
        return False

    if v0x is None:
        print('No v0x (initial velocities x) passed!')
        return False

    if v0y is None:
        print('No v0y (initial velocities y) passed!')
        return False

    if t_action is None:
        print('No t_action passed!')
        return False

    if n_timesteps is None:
        print('No n_timesteps passed!')
        return False

    if l is None:
        print('No l (dimensions of the agent) passed!')
        return False

    # The following code simultaneously checks whether trajs_hulls is an ndarray
    # and also provides the dimensions.
    try:
        n_traj, n_hulls_per_traj = trajs_hulls.shape
    except AttributeError:
        trajs_hulls = np.array(trajs_hulls, dtype=object)
        n_traj, n_hulls_per_traj = trajs_hulls.shape

    # The following code simultaneously checks whether trajs_boxes is an ndarray
    # and also provides the dimensions.
    try:
        n_traj_boxes, n_boxes_per_traj = trajs_boxes.shape
    except AttributeError:
        trajs_boxes = np.array(trajs_boxes, dtype=object)
        n_traj_boxes, n_boxes_per_traj = trajs_boxes.shape

    if n_traj != n_traj_boxes:
        print('There is a mismatch in the number of agents in the trajs_hulls and trajs_boxes')
        return False

    if n_hulls_per_traj != (n_boxes_per_traj - 1):
        print('There is a mismatch in the number of hulls per trajectory and number of boxes per trajectory')
        print('n_hulls_per_traj should be (n_boxes_per_traj-1)')
        print(f'Current values: {n_hulls_per_traj=}, {n_boxes_per_traj=}')
        return False

    # ----------------------------------------------------------- #

    # Swap actor action to move passed to get new trajs_boxes and trajs_hulls
    if move is not None:
        mdr_string = ' MdR | '
        trajs_hulls, trajs_boxes = swap_agent_traj_in_trajs(trajs_hulls=trajs_hulls, trajs_boxes=trajs_boxes,
                                                            x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                                            traj2swap=actor,
                                                            a2swap=move[:, 0].reshape((-1, 1)),
                                                            theta2swap=move[:, 1].reshape((-1, 1)),
                                                            t_action=t_action, n_timesteps=n_timesteps,
                                                            l=l, plot_new_traj=False)
    else:
        mdr_string = ' Move | '
        # To make sure that the original array is not modified
        trajs_boxes = trajs_boxes.copy()
        trajs_hulls = trajs_hulls.copy()

    # Get list of agents other than the affected
    not_affected = get_other_agents(n=n_traj, ii=affected)

    # Get traj_hulls and trajs_boxes for not_affected
    trajs_hulls_not_j = trajs_hulls[not_affected]
    trajs_boxes_not_j = trajs_boxes[not_affected]

    # Resolve collisions if actors move was updated.
    if move is not None:
        trajs_hulls_not_j, trajs_boxes_not_j = get_collision_free_trajs(trajs_hulls_not_j, trajs_boxes_not_j,
                                                                        obstacle=obstacle,
                                                                        plot_polygons_one_by_one=False)

    x0_j = x0[affected]
    y0_j = y0[affected]
    v0x_j = v0x[affected]
    v0y_j = v0y[affected]

    fig, configuration_space_ax = plotcf.get_new_fig(121)
    action_space_ax = fig.add_subplot(122)
    fig.tight_layout(pad=5.0)

    _, action_space_ax_stripped = plotcf.get_new_fig()
    _, configuration_space_ax_stripped = plotcf.get_new_fig()

    plot_feasible_action_and_configuration_space(a_min=a_min, a_max=a_max, a_num=a_num,
                                                 theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                                                 feasibility=feasibility,
                                                 action_space_ax=action_space_ax,
                                                 action_space_ax_stripped=action_space_ax_stripped,
                                                 configuration_space_ax=configuration_space_ax,
                                                 configuration_space_ax_stripped=configuration_space_ax_stripped,
                                                 strip_feasible_action_space_plots=strip_feasible_action_space_plots,
                                                 l=l, n_timesteps=n_timesteps,
                                                 obstacle=obstacle,
                                                 t_action=t_action,
                                                 show_legend=show_legend, legend_inside=legend_inside, finer=finer,
                                                 actor=actor, affected=affected,
                                                 not_affected=not_affected,
                                                 trajs_boxes=trajs_boxes,
                                                 trajs_boxes_not_j=trajs_boxes_not_j,
                                                 trajs_hulls_not_j=trajs_hulls_not_j,
                                                 v0x_j=v0x_j, v0y_j=v0y_j, x0_j=x0_j, y0_j=y0_j,
                                                 mdr_string=mdr_string,
                                                 save_figs=save_figs,
                                                 feasible_space_filename=feasible_space_filename)


def plot_feasible_action_and_configuration_space(a_min, a_max, a_num, action_space_ax, action_space_ax_stripped,
                                                 actor, affected, configuration_space_ax,
                                                 configuration_space_ax_stripped, feasibility, l, n_timesteps,
                                                 not_affected, obstacle, t_action,
                                                 theta_min, theta_max, theta_num,
                                                 trajs_boxes, trajs_boxes_not_j, trajs_hulls_not_j,
                                                 v0x_j, v0y_j, x0_j, y0_j, mdr_string='',
                                                 strip_feasible_action_space_plots=True,
                                                 show_legend=True, legend_inside=True,
                                                 save_figs=False, feasible_space_filename='',
                                                 verbose_title=False,
                                                 finer=False):
    a_js = np.linspace(a_min, a_max, a_num + 1)
    theta_js = np.linspace(theta_min, theta_max, theta_num + 1)

    aa_s = np.array(range(len(a_js) - 1))
    thetaa_s = np.array(range(len(theta_js) - 1))

    aa_thetaa_list = [(aa, thetaa)
                      for aa in aa_s
                      for thetaa in thetaa_s]

    plot_feasible_action_and_configuration_subspaces(feasibility=feasibility, aa_thetaa_list=aa_thetaa_list,
                                                     a_js=a_js, theta_js=theta_js,
                                                     x0_j=x0_j, y0_j=y0_j, v0x_j=v0x_j, v0y_j=v0y_j,
                                                     n_timesteps=n_timesteps, t_action=t_action, l=l,
                                                     action_space_ax=action_space_ax,
                                                     configuration_space_ax=configuration_space_ax,
                                                     action_space_ax_stripped=action_space_ax_stripped,
                                                     configuration_space_ax_stripped=configuration_space_ax_stripped
                                                     )

    plotcf.plot_polygon_outline(trajs_boxes[affected][0], color=plotcf.BLACK, zorder=5,
                                ax=configuration_space_ax, linewidth=1)

    # plotcf.plot_velocities_for_agents(x=x0_j, y=y0_j, vx=v0x_j, vy=v0y_j,
    #                                   scale=2, markersize=.05,
    #                                   ax=configuration_space_ax, colour=plotcf.BLACK)

    if verbose_title:
        action_space_title = f'Feasible Action Space\n{mdr_string}' \
                             f'Actor: {actor + 1}, Affected : {affected + 1}'
        configuration_space_title = f'Feasible Configuration Space\n{mdr_string}' \
                                    f'Actor: {actor + 1}, Affected : {affected + 1}'
    else:
        action_space_title = 'Action Space'
        configuration_space_title = 'Configuration Space'

    colours = plotcf.scientific_cmap(n_colours=(len(not_affected) + 1))
    colours_not_affected = [colours[ii] for ii in not_affected]
    # print(f'{colours_not_affected=}')

    plotcf.plot_hulls_for_trajs(trajs_hulls_not_j, trajs_boxes=trajs_boxes_not_j, ax=configuration_space_ax,
                                labels=not_affected, show_legend=show_legend, legend_inside=legend_inside,
                                title=configuration_space_title,
                                colour=colours_not_affected
                                # colour=plotcf.YELLOW
                                )
    plotcf.plot_obstacle(obstacle=obstacle, ax=configuration_space_ax)
    action_space_ax.set_title(action_space_title, fontsize=12)
    # action_space_ax.set_xlabel('Acceleration', fontsize=12)
    # action_space_ax.set_ylabel('Theta', fontsize=12)
    action_space_ax.set_xlabel('$a$', fontsize=12)
    action_space_ax.set_ylabel('$\\theta$', fontsize=12)
    # action_space_ax.set_aspect(1 / np.pi)

    # configuration_space_ax_stripped.set_title(f'Feasible Configuration Space\n{mdr_string}'
    #                                           f'Actor: {actor + 1}, Affected : {affected + 1}', fontsize=12)
    # action_space_ax_stripped.set_title(f'Feasible Action Space\n{mdr_string}'
    #                                    f'Actor: {actor + 1}, Affected : {affected + 1}', fontsize=12)
    configuration_space_ax_stripped.set_title('')
    action_space_ax_stripped.set_title('')

    # action_space_ax_stripped.set_xlabel('Acceleration', fontsize=12)
    # action_space_ax_stripped.set_ylabel('Theta', fontsize=12)
    action_space_ax_stripped.set_xlabel('$a$', fontsize=12)
    action_space_ax_stripped.set_ylabel('$\\theta$', fontsize=12)

    # plotcf.make_axes_gray(action_space_ax)
    # plotcf.make_axes_gray(action_space_ax_stripped)

    match_ax1_size_2_ax2(ax1=action_space_ax, ax2=configuration_space_ax)
    match_ax1_size_2_ax2(ax1=action_space_ax_stripped, ax2=configuration_space_ax_stripped)
    plotcf.strip_axes(ax=configuration_space_ax, strip_title=False, strip_legend=False)

    if not finer:
        plotcf.set_figure_fontsizes(obj=action_space_ax, title_fontsize=16, other_fontsize=16)
        plotcf.set_figure_fontsizes(obj=action_space_ax_stripped, title_fontsize=16, other_fontsize=16)
        plotcf.set_figure_fontsizes(obj=configuration_space_ax_stripped, title_fontsize=16, other_fontsize=16)

        # plotcf.rescale_plot_size(ax=action_space_ax, scale=0.5)
        plotcf.rescale_plot_size(ax=action_space_ax_stripped, scale=0.5)
        plotcf.rescale_plot_size(ax=action_space_ax_stripped, scale=0.5)

    if save_figs:
        plotcf.save_figure(ax=action_space_ax,
                           filename=f"{feasible_space_filename}_actionConfigSpace_{actor + 1}-{affected + 1}")

    if save_figs:
        # Save the plots before stripping
        plotcf.save_figure(ax=action_space_ax_stripped, close_after=not strip_feasible_action_space_plots,
                           filename=f"{feasible_space_filename}_actionSpace_{actor + 1}-{affected + 1}")
        plotcf.save_figure(ax=configuration_space_ax_stripped, close_after=not strip_feasible_action_space_plots,
                           filename=f"{feasible_space_filename}_configSpace_{actor + 1}-{affected + 1}")

    if strip_feasible_action_space_plots:
        plotcf.strip_axes(ax=action_space_ax_stripped)
        plotcf.strip_axes(ax=configuration_space_ax_stripped)
        if save_figs:
            plotcf.save_figure(ax=action_space_ax_stripped,
                               filename=f"{feasible_space_filename}_actionSpace_{actor + 1}-{affected + 1}_stripped")
            plotcf.save_figure(ax=configuration_space_ax_stripped,
                               filename=f"{feasible_space_filename}_configSpace_{actor + 1}-{affected + 1}_stripped")


def read_and_plot_results(results_file_path=None, plot_feasible_action_and_cofiguration_spaces_=False,
                          plot_scenario=False,
                          plot_scenario_gray=False,
                          plot_boxes_and_trajs=False,
                          plot_fear_graph=True,
                          save_figs=False,
                          strip_scenario_titles=True,
                          scale_velocities=2,
                          scale_scenario_size=1,
                          match_scenario_lims=True,
                          fear_threshold=None,
                          fear_bar_plot=False,
                          fear_bar_plot_indices=None,
                          fear_bar_plot_panel_text=None,
                          fear_cbar=None,
                          highlight_agents=None,
                          place_scenario_legend=None,
                          arrow_width=0.45,
                          arrow_markersize=8,
                          animate=False
                          ):
    """

    Parameters
    ----------
    arrow_markersize
    arrow_width
    results_file_path
    plot_feasible_action_and_cofiguration_spaces_
    plot_scenario
    plot_scenario_gray
    plot_boxes_and_trajs
    plot_fear_graph
    save_figs
    strip_scenario_titles
    scale_velocities
    scale_scenario_size
    match_scenario_lims
    fear_threshold
    fear_bar_plot
    fear_bar_plot_indices
    fear_cbar
    highlight_agents
    place_scenario_legend : str
        Choose from ['right_out', 'right_in', 'left_out', 'left_in', 'skip', 'extend_right_out', 'up']
        to choose how to place legends on scenario plots
    fear_bar_plot_panel_text : array
        Array with strings for panel text for fear_bar_plots. Must be the same size as the number of values.

    Returns
    -------

    """
    if results_file_path is None:
        print('No results_file_path passed in!')
        return False

    results_file_name, _ = os.path.splitext(os.path.basename(results_file_path))

    # Open the file in binary mode
    with open(results_file_path, 'rb') as file:
        print('\n------------------------------------------------------------------------------\n')
        print(f'\nLoading file: {results_file_path}\n')
        print('------------------------------------------------------------------------------\n')

        # Call load method to deserialze
        results = pickle.load(file)

    values = results['values']
    base_params = results['base_params']
    hyperparameter = results['hyperparameter']
    scenario_name = results['scenario_name']
    scenarios_json_file = results['scenarios_json_file']
    fears = results['fear_values']
    feasibilities = results['feasibilities']
    if 'mdrs' in results.keys():
        mdrs = results['mdrs']
    else:
        mdrs = None

    for vv in range(len(values)):
        params = base_params.copy()

        # Set the value of the hyperparameter
        params[hyperparameter] = values[vv]

        # Extracting the parameter values to variables
        t_action = params['t_action']
        n_timesteps = params['n_timesteps']
        collision_free_trajs = params['collision_free_trajs']
        compute_mdrs = params['compute_mdrs']
        a_min = params['a_min']
        a_max = params['a_max']
        a_num = params['a_num']
        theta_min = params['theta_min']
        theta_max = params['theta_max']
        theta_num = params['theta_num']

        scenarios = Scenario.load_scenarios(scenarios_json_file)
        scenario = scenarios[scenario_name]

        # Populating variables with scenario data
        N = scenario.n_agents()
        l = scenario.l
        x0 = scenario.x0
        y0 = scenario.y0
        v0x = scenario.v0x
        v0y = scenario.v0y
        a = scenario.a
        theta = scenario.theta
        mdr_a_scenario = scenario.mdr_a
        mdr_theta_scenario = scenario.mdr_theta

        if scenario.obstacle != "None":
            obstacle = scenario.obstacle
        else:
            obstacle = None

        fear_bar_plot_panel_text_vv = None
        if fear_bar_plot_panel_text:
            if len(fear_bar_plot_panel_text) == len(values):
                fear_bar_plot_panel_text_vv = fear_bar_plot_panel_text[vv]
            else:
                print('Mismatch in the number of panel texts. Skipping.')

        # -------------------------------------------------------------------------------------------------------------

        #  Creating the trajectories for the initial conditions

        x_, y_, t_ = make_trajectories(x0=x0, y0=y0, v0x=v0x, v0y=v0y, a=a, theta=theta,
                                       t_action=t_action, n_timesteps=n_timesteps)
        if plot_boxes_and_trajs:
            ax = plotcf.plot_trajectories(x=x_, y=y_, obstacle=obstacle)
            plotcf.set_fontsizes(ax=ax)

            ax = plotcf.plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y, ax=ax,
                                                   scale=t_action / n_timesteps * scale_velocities,
                                                   arrow_width=arrow_width,
                                                   markersize=arrow_markersize)
            plotcf.set_fontsizes(ax=ax)
            plotcf.strip_axes(ax=ax, strip_legend=False, strip_title=strip_scenario_titles)

            if save_figs:
                plotcf.save_figure(ax=ax, filename=f"{results_file_name}_scenario_{vv}_trajs")

        trajs_boxes = get_boxes_for_trajs(x_, y_, lx=l, ly=l)

        if plot_boxes_and_trajs:
            ax = plotcf.plot_boxes_for_trajs(trajs_boxes, scale_scenario_size=scale_scenario_size * 0.7)
            ax = plotcf.plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y, ax=ax,
                                                   scale=t_action / n_timesteps * scale_velocities,
                                                   arrow_width=arrow_width,
                                                   markersize=arrow_markersize)
            plotcf.plot_obstacle(obstacle, ax=ax)
            plotcf.set_fontsizes(ax=ax)
            plotcf.strip_axes(ax=ax, strip_legend=True, strip_title=strip_scenario_titles)

            if save_figs:
                plotcf.save_figure(ax=ax, filename=f"{results_file_name}_scenario_{vv}_boxes")

        trajs_hulls = get_trajs_hulls(trajs_boxes)

        if plot_boxes_and_trajs:  # Plot scenario - but not scaled to match the MdR plot
            if place_scenario_legend:
                show_scenario_legend = False
            else:
                show_scenario_legend = True

            ax_scenario_unscaled = plotcf.plot_hulls_for_trajs(trajs_hulls, trajs_boxes=trajs_boxes,
                                                               legend_inside=False, show_legend=show_scenario_legend,
                                                               scale_scenario_size=scale_scenario_size * 0.7,
                                                               title='Scenario')
            ax_scenario_unscaled = plotcf.plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y,
                                                                     ax=ax_scenario_unscaled,
                                                                     scale=t_action / n_timesteps * scale_velocities,
                                                                     arrow_width=arrow_width,
                                                                     markersize=arrow_markersize)
            plotcf.plot_obstacle(obstacle, ax=ax_scenario_unscaled)

            if place_scenario_legend == 'right_out':
                ax_scenario_unscaled.legend(bbox_to_anchor=(1.0, 1.0), loc='lower right')
            elif place_scenario_legend == 'right_in':
                ax_scenario_unscaled.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right')
            elif place_scenario_legend == 'left_out':
                ax_scenario_unscaled.legend(bbox_to_anchor=(0.0, 1.0), loc='lower left')
            elif place_scenario_legend == 'left_in':
                ax_scenario_unscaled.legend(bbox_to_anchor=(0.0, .98), loc='upper left')
            elif place_scenario_legend == 'extend_right_out':
                ax_scenario_unscaled.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            elif place_scenario_legend == 'up':
                ax_scenario_unscaled.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', ncols=min(N, 3))

            plotcf.set_fontsizes(ax=ax_scenario_unscaled)
            plotcf.strip_axes(ax=ax_scenario_unscaled, strip_legend=False, strip_title=strip_scenario_titles)
            if save_figs:
                plotcf.save_figure(ax=ax_scenario_unscaled, filename=f"{results_file_name}_scenario_{vv}_unscaled")

        if plot_scenario:
            if place_scenario_legend:
                show_scenario_legend = False
            else:
                show_scenario_legend = True

            ax_scenario = plotcf.plot_hulls_for_trajs(trajs_hulls, trajs_boxes=trajs_boxes,
                                                      legend_inside=False, show_legend=show_scenario_legend,
                                                      scale_scenario_size=scale_scenario_size,
                                                      title='Scenario')
            if N >= 5:
                ax_scenario = plotcf.plot_agent_ids(ax=ax_scenario, x0=x0, y0=y0, zorder_lift=5,
                                                    highlight_agents=highlight_agents)
                show_scenario_legend = False

            ax_scenario = plotcf.plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y, ax=ax_scenario,
                                                            scale=t_action / n_timesteps * scale_velocities,
                                                            arrow_width=arrow_width,
                                                            markersize=arrow_markersize)
            plotcf.plot_obstacle(obstacle, ax=ax_scenario)

            if place_scenario_legend == 'right_out':
                ax_scenario.legend(bbox_to_anchor=(1.0, 1.0), loc='lower right')
            elif place_scenario_legend == 'right_in':
                ax_scenario.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right')
            elif place_scenario_legend == 'left_out':
                ax_scenario.legend(bbox_to_anchor=(0.0, 1.0), loc='lower left')
            elif place_scenario_legend == 'left_in':
                ax_scenario.legend(bbox_to_anchor=(0.0, .98), loc='upper left')
            elif place_scenario_legend == 'extend_right_out':
                ax_scenario.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            elif place_scenario_legend == 'up':
                ax_scenario.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', ncols=min(N, 3))

            if not (place_scenario_legend == 'skip'):
                plotcf.set_fontsizes(ax=ax_scenario, legend_fontsize=LEGEND_FONTSIZE)
            plotcf.strip_axes(ax=ax_scenario, strip_legend=False, strip_title=strip_scenario_titles)

        if plot_scenario_gray:
            ax = plotcf.plot_hulls_for_trajs(trajs_hulls, trajs_boxes=trajs_boxes, colour='lightgrey',
                                             scale_scenario_size=scale_scenario_size,
                                             title='Scenario')
            ax = plotcf.plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y, ax=ax,
                                                   scale=t_action / n_timesteps * scale_velocities,
                                                   arrow_width=arrow_width,
                                                   markersize=arrow_markersize)
            plotcf.plot_obstacle(obstacle, ax=ax)
            plotcf.set_fontsizes(ax=ax)
            plotcf.strip_axes(ax=ax, strip_legend=False, strip_title=strip_scenario_titles)
            # plotcf.match_xylims_to_fit(ax1=ax_scenario, ax2=ax)
            if save_figs:
                plotcf.save_figure(ax=ax, filename=f"{results_file_name}_scenario_{vv}_gray")

        if animate:
            for tt in range(n_timesteps-1):
                ax = plotcf.plot_hulls_for_trajs(trajs_hulls, trajs_boxes=trajs_boxes, colour='gainsboro',
                                                 show_legend=False,
                                                 scale_scenario_size=scale_scenario_size)
                trajs_hulls_tt = trajs_hulls[:, tt].reshape(-1, 1)
                ax = plotcf.plot_hulls_for_trajs(trajs_hulls_tt, ax=ax,
                                                 scale_scenario_size=scale_scenario_size,
                                                 show_legend=False,
                                                 show_first=False,
                                                 title=f'Scenario: Timestep:{tt}')
                # ax = plotcf.plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y, ax=ax,
                #                                        scale=t_action / n_timesteps * scale_velocities,
                #                                        arrow_width=arrow_width,
                #                                        markersize=arrow_markersize)
                plotcf.plot_obstacle(obstacle, ax=ax)
                plotcf.set_fontsizes(ax=ax)
                plotcf.strip_axes(ax=ax, strip_legend=False, strip_title=strip_scenario_titles)
                # plotcf.match_xylims_to_fit(ax1=ax_scenario, ax2=ax)
                if save_figs:
                    plotcf.save_figure(ax=ax, filename=f"{results_file_name}_scenario_{vv}_animation_{tt}")

        # -------------------------------------------------------------------------------------------------------------

        # Check for collisions and rectify collisions

        if collision_free_trajs:
            trajs_hulls, trajs_boxes = get_collision_free_trajs(trajs_hulls, trajs_boxes, obstacle=obstacle,
                                                                plot_polygons_one_by_one=False)

            ax = plotcf.plot_boxes_for_trajs(trajs_boxes, obstacle=obstacle, scale_scenario_size=scale_scenario_size)
            ax = plotcf.plot_hulls_for_trajs(trajs_hulls, obstacle=obstacle, scale_scenario_size=scale_scenario_size)

        # -------------------------------------------------------------------------------------------------------------
        # Moves de Rigueur !

        if mdrs is not None:  # Check if MdRs saved in results
            mdr_a = mdrs[vv]['mdr_a']
            mdr_theta = mdrs[vv]['mdr_theta']

            # Plot scenario with MdR
            if plot_scenario:
                tweaked_a = mdr_a
                tweaked_theta = mdr_theta
                tweaked_title = 'Moves de Rigueur'

                ax_mdr = plotcf.plot_tweaked_scenario(tweaked_a=tweaked_a, tweaked_theta=tweaked_theta,
                                                      tweaked_title=tweaked_title,
                                                      v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                      l=l, n_timesteps=n_timesteps, obstacle=obstacle,
                                                      scale_velocities=scale_velocities,
                                                      arrow_width=arrow_width,
                                                      arrow_markersize=arrow_markersize,
                                                      scale_scenario_size=scale_scenario_size,
                                                      strip_scenario_titles=strip_scenario_titles, t_action=t_action,
                                                      agents_to_highlight=[], show_legend=show_scenario_legend,
                                                      )
                ax_mdr.get_figure().set_facecolor('whitesmoke')

                if place_scenario_legend == 'right_out':
                    ax_mdr.legend(bbox_to_anchor=(1.0, 1.0), loc='lower right')
                elif place_scenario_legend == 'right_in':
                    ax_mdr.legend(bbox_to_anchor=(0.98, 0.98), loc='upper right')
                elif place_scenario_legend == 'left_out':
                    ax_mdr.legend(bbox_to_anchor=(0.0, 1.0), loc='lower left')
                elif place_scenario_legend == 'left_in':
                    ax_mdr.legend(bbox_to_anchor=(0.0, .98), loc='upper left')
                elif place_scenario_legend == 'extend_right_out':
                    ax_mdr.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
                elif place_scenario_legend == 'up':
                    ax_mdr.legend(bbox_to_anchor=(0.5, 1.15), loc='lower center', ncols=min(N, 3))

                if not (place_scenario_legend == 'skip'):
                    plotcf.set_fontsizes(ax=ax_mdr, legend_fontsize=LEGEND_FONTSIZE)
                # plotcf.set_fontsizes(ax=ax_mdr, legend_fontsize=LEGEND_FONTSIZE)

                if match_scenario_lims:
                    plotcf.match_xylims_to_fit(ax1=ax_scenario, ax2=ax_mdr)
                if save_figs:
                    plotcf.save_figure(ax=ax_mdr, filename=f"{results_file_name}_scenario_{vv}_mdr")
                    plotcf.save_figure(ax=ax_scenario, filename=f"{results_file_name}_scenario_{vv}")

        # Compute MdRs
        elif compute_mdrs:
            mdr_a, mdr_theta = get_mdr(x=x0, y=y0, v0x=v0x, v0y=v0y, time_per_step=t_action,
                                       buffer=2 * l)
        else:
            mdr_a = mdr_a_scenario
            mdr_theta = mdr_theta_scenario

        print(f'mdr_a=\n{mdr_a}')
        print(f'mdr_theta=\n{mdr_theta}')
        # print(f'{a=}')
        # print(f'{theta=}')
        # print(f'{(mdr_a==a)=}')
        # print(f'{(mdr_theta==theta)=}')

        # -------------------------------------------------------------------------------------------------------------

        # load FeAR

        fear = fears[vv]
        feasibility = feasibilities[vv]

        if plot_feasible_action_and_cofiguration_spaces_:
            feasible_space_filename = f"{results_file_name}_scenario_{vv}"

            legend_inside = N < 5
            show_legend = N < 10
            finer = N > 8

            plot_feasible_action_space_for_all(nn=N, feasibility=feasibility,
                                               mdr_a=mdr_a, mdr_theta=mdr_theta,
                                               a_min=a_min, a_max=a_max, a_num=a_num,
                                               theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                                               trajs_boxes=trajs_boxes,
                                               trajs_hulls=trajs_hulls,
                                               obstacle=obstacle,
                                               x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                               t_action=t_action, n_timesteps=n_timesteps,
                                               l=l,
                                               save_figs=save_figs,
                                               show_legend=show_legend, legend_inside=legend_inside, finer=finer,
                                               feasible_space_filename=feasible_space_filename)

        print(f'fear = \n{np.array2string(fear, separator=", ", max_line_width=np.inf)}')
        ax = plotcf.plot_fear(fear=fear, cbar=fear_cbar)

        # if highlight_agents == [2, 3, 4, 5]:
        if check_array_consecutive_ints(arr=highlight_agents):
            np_highlight_agents = np.array(highlight_agents)
            highlight_start = np_highlight_agents.min()
            highlight_len = len(np_highlight_agents)
            ax = plotcf.plot_rect_on_matrix(x=highlight_start, y=highlight_start,
                                            x_len=highlight_len, y_len=highlight_len, offset=0.07,
                                            ax=ax, color='tan', linewidth=4)
            ax = plotcf.plot_rect_on_matrix(x=highlight_start, y=highlight_start,
                                            x_len=highlight_len, y_len=highlight_len, offset=0.01,
                                            ax=ax, color='white', linewidth=1)

        # plotcf.set_fontsizes(ax=ax)
        if save_figs:
            plotcf.save_figure(ax=ax, filename=f"{results_file_name}_scenario_{vv}_fear")

        if fear_bar_plot:
            # tweak_ticks = (vv == 0)
            tweak_ticks = False

            ax = plotcf.plot_fear_barplot(fear=fear, indices_to_plot=fear_bar_plot_indices, tweak_ticks=tweak_ticks)
            if save_figs:
                plotcf.save_figure(ax=ax, filename=f"{results_file_name}_scenario_{vv}_fear_bars")

            if fear_bar_plot_panel_text_vv:
                ax = plotcf.plot_fear_barplot(fear=fear, indices_to_plot=fear_bar_plot_indices, tweak_ticks=tweak_ticks)
                ax.text(0.01, 0.99, fear_bar_plot_panel_text_vv, transform=ax.transAxes,
                        ha='left', va='top', fontsize=30)
                if save_figs:
                    plotcf.save_figure(ax=ax, filename=f"{results_file_name}_scenario_{vv}_fear_bars_panel")

        # Dynamically setting the threshold for the fear values based on the number of agents
        fear_threshold_percentile = 0 + (N - 2) / N * (100 - 1)  # so that when as nn->inf, this -> 99

        if plot_fear_graph:
            ax = plotcf.plot_fear_graph_on_trajs(fear=fear, obstacle=obstacle, trajs_hulls=trajs_hulls,
                                                 trajs_boxes=trajs_boxes, x0=x0, y0=y0, fear_threshold=fear_threshold,
                                                 fear_threshold_percentile=fear_threshold_percentile)
            # ax = plotcf.plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y, ax=ax, scale=t_action / n_timesteps*5)
            plotcf.set_fontsizes(ax=ax)
            plotcf.strip_axes(ax=ax)
            if save_figs:
                plotcf.save_figure(ax=ax, filename=f"{results_file_name}_fearGraph_{vv}")

        plotcf.plt.show()

    print('\n\n------------------------------------------------------------------------------')
    print(f'\nSummary : {hyperparameter=}, {values=}\n')
    print('------------------------------------------------------------------------------\n')

    for fear in fears:
        # Print the array with each row on a separate line
        print(f'fear = \n{np.array2string(fear, separator=", ", max_line_width=np.inf)}')
        print('                             --------------- ')

    print('\n\n------------------------------------------------------------------------------\n')

    ax = plotcf.plot_hyper_fears(results=results)
    plotcf.set_figure_fontsizes(obj=ax, title_fontsize=12, other_fontsize=14)
    if save_figs:
        plotcf.save_figure(ax=ax, filename=f"{results_file_name}_hyper_fears")

    # ------------------------------------------------------------------------------------------ #

    # Plot results for grid search for fear by actors if present.

    if 'fear_grid_search' in results.keys():
        fear_grid_search = results['fear_grid_search']

        plotcf.plot_fear_grid_search(fear_grid_search, save_figs=save_figs, results_file_name=results_file_name)

        for actor_string in fear_grid_search.keys():
            actor = int(actor_string)

            ax = plotcf.plot_tweaked_scenario(tweaked_a=a, tweaked_theta=theta,
                                              tweaked_title='',
                                              v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                              l=l, n_timesteps=n_timesteps, obstacle=obstacle,
                                              scale_velocities=scale_velocities,
                                              arrow_width=arrow_width,
                                              arrow_markersize=arrow_markersize,
                                              strip_scenario_titles=strip_scenario_titles, t_action=t_action,
                                              agents_to_highlight=[actor]
                                              )
            if save_figs:
                plotcf.save_figure(ax=ax, filename=f"{results_file_name}_actor_{actor}")
            else:
                plotcf.plt.show()

            fears_by_actor = fear_grid_search[actor_string]['fears_by_actor']
            actor_as = fear_grid_search[actor_string]['actor_as']
            actor_thetas = fear_grid_search[actor_string]['actor_thetas']
            if 'collisions_4_actor' in fear_grid_search[actor_string].keys():
                collisions_4_actor = fear_grid_search[actor_string]['collisions_4_actor']
            else:
                collisions_4_actor = None

            others = get_other_agents(n=N, ii=actor)
            fear_on_others = fears_by_actor[:, :, others]

            tweaked_title = 'Optimal actions for Mean FeAR'
            mean_fear_by_actor = np.mean(fear_on_others, axis=2)
            ax = plotcf.plot_fearless_actions_for_actor(fear_by_actor=mean_fear_by_actor,
                                                        tweaked_title=tweaked_title,
                                                        actor_as=actor_as, actor_thetas=actor_thetas,
                                                        actor=actor, a=a, theta=theta,
                                                        v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                        l=l, n_timesteps=n_timesteps,
                                                        obstacle=obstacle, scale_velocities=scale_velocities,
                                                        arrow_width=arrow_width,
                                                        arrow_markersize=arrow_markersize,
                                                        strip_scenario_titles=strip_scenario_titles, t_action=t_action,
                                                        collisions_4_actor=collisions_4_actor
                                                        )
            if save_figs:
                plotcf.save_figure(ax=ax, filename=f"{results_file_name}_actor_{actor}_mean_fearless_actions")

            tweaked_title = 'Optimal actions for Min FeAR'
            min_fear_by_actor = np.min(fear_on_others, axis=2)
            ax = plotcf.plot_fearless_actions_for_actor(fear_by_actor=min_fear_by_actor,
                                                        tweaked_title=tweaked_title,
                                                        actor_as=actor_as, actor_thetas=actor_thetas,
                                                        actor=actor, a=a, theta=theta,
                                                        v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                        l=l, n_timesteps=n_timesteps,
                                                        obstacle=obstacle, scale_velocities=scale_velocities,
                                                        arrow_width=arrow_width,
                                                        arrow_markersize=arrow_markersize,
                                                        strip_scenario_titles=strip_scenario_titles, t_action=t_action,
                                                        collisions_4_actor=collisions_4_actor
                                                        )
            if save_figs:
                plotcf.save_figure(ax=ax, filename=f"{results_file_name}_actor_{actor}_min_fearless_actions")

            tweaked_title = 'Optimal actions for Max FeAR'
            max_fear_by_actor = np.max(fear_on_others, axis=2)
            ax = plotcf.plot_fearless_actions_for_actor(fear_by_actor=max_fear_by_actor,
                                                        tweaked_title=tweaked_title,
                                                        actor_as=actor_as, actor_thetas=actor_thetas,
                                                        actor=actor, a=a, theta=theta,
                                                        v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                        l=l, n_timesteps=n_timesteps,
                                                        obstacle=obstacle, scale_velocities=scale_velocities,
                                                        arrow_width=arrow_width,
                                                        arrow_markersize=arrow_markersize,
                                                        strip_scenario_titles=strip_scenario_titles, t_action=t_action,
                                                        collisions_4_actor=collisions_4_actor
                                                        )
            if save_figs:
                plotcf.save_figure(ax=ax, filename=f"{results_file_name}_actor_{actor}_max_fearless_actions")

            # ---------------------------- More involved metrics ------------------------------------------ #

            tweaked_title = 'Optimal actions for being courteous to most'
            number_negative_fear_by_actor = - np.sum(fear_on_others < 0, axis=2)
            ax = plotcf.plot_fearless_actions_for_actor(fear_by_actor=number_negative_fear_by_actor,
                                                        tweaked_title=tweaked_title,
                                                        actor_as=actor_as, actor_thetas=actor_thetas,
                                                        actor=actor, a=a, theta=theta,
                                                        v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                        l=l, n_timesteps=n_timesteps,
                                                        obstacle=obstacle, scale_velocities=scale_velocities,
                                                        arrow_width=arrow_width,
                                                        arrow_markersize=arrow_markersize,
                                                        strip_scenario_titles=strip_scenario_titles, t_action=t_action,
                                                        collisions_4_actor=collisions_4_actor
                                                        )
            if save_figs:
                plotcf.save_figure(ax=ax,
                                   filename=f"{results_file_name}_actor_{actor}_courteous2most_fearless_actions")

            tweaked_title = 'Optimal actions for being assertive to least'
            number_positive_fear_by_actor = np.sum(fear_on_others > 0, axis=2)
            ax = plotcf.plot_fearless_actions_for_actor(fear_by_actor=number_positive_fear_by_actor,
                                                        tweaked_title=tweaked_title,
                                                        actor_as=actor_as, actor_thetas=actor_thetas,
                                                        actor=actor, a=a, theta=theta,
                                                        v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                        l=l, n_timesteps=n_timesteps,
                                                        obstacle=obstacle, scale_velocities=scale_velocities,
                                                        arrow_width=arrow_width,
                                                        arrow_markersize=arrow_markersize,
                                                        strip_scenario_titles=strip_scenario_titles, t_action=t_action,
                                                        collisions_4_actor=collisions_4_actor
                                                        )
            if save_figs:
                plotcf.save_figure(ax=ax,
                                   filename=f"{results_file_name}_actor_{actor}_assertive2least_fearless_actions")

            tweaked_title = 'Optimal actions for (assertive - courteous) affected'
            number_assertive_minus_courteous = number_positive_fear_by_actor + number_negative_fear_by_actor
            ax = plotcf.plot_fearless_actions_for_actor(fear_by_actor=number_assertive_minus_courteous,
                                                        tweaked_title=tweaked_title,
                                                        actor_as=actor_as, actor_thetas=actor_thetas,
                                                        actor=actor, a=a, theta=theta,
                                                        v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                        l=l, n_timesteps=n_timesteps,
                                                        obstacle=obstacle, scale_velocities=scale_velocities,
                                                        arrow_width=arrow_width,
                                                        arrow_markersize=arrow_markersize,
                                                        strip_scenario_titles=strip_scenario_titles, t_action=t_action,
                                                        collisions_4_actor=collisions_4_actor
                                                        )
            if save_figs:
                plotcf.save_figure(ax=ax,
                                   filename=f"{results_file_name}_actor_{actor}_assertive-courteous_fearless_actions")


# Very slow - DO NOT USE
def get_feasible_action_space_multiprocessing(
        actor=None, move=None, affected=None, a_min=-1, a_max=1, a_num=1,
        theta_min=-np.pi / 2, theta_max=np.pi / 2, theta_num=8,
        trajs_hulls=None, trajs_boxes=None,
        x0=None, y0=None, v0x=None, v0y=None,
        t_action=None, n_timesteps=None, l=None, verbose_flag=False,
        plot_subspaces=False, plot_feasible_action_space=False, plot_intersections=False):
    # ------------------------ Checks ------------------------------ #

    if actor is None:
        print('No actor agent passed!')
        return False

    if affected is None:
        print('No affected agent passed!')

    if trajs_hulls is None:
        print('No trajs_hulls passed!')
        return False

    if trajs_boxes is None:
        print('No trajs_boxes passed!')
        return False

    if x0 is None:
        print('No x0 (initial locations x) passed!')
        return False

    if y0 is None:
        print('No x0 (initial locations y) passed!')
        return False

    if v0x is None:
        print('No v0x (initial velocities x) passed!')
        return False

    if v0y is None:
        print('No v0y (initial velocities y) passed!')
        return False

    if t_action is None:
        print('No t_action passed!')
        return False

    if n_timesteps is None:
        print('No n_timesteps passed!')
        return False

    if l is None:
        print('No l (dimensions of the agent) passed!')
        return False

    # The following code simultaneously checks whether trajs_hulls is an ndarray
    # and also provides the dimensions.
    try:
        n_traj, n_hulls_per_traj = trajs_hulls.shape
    except AttributeError:
        trajs_hulls = np.array(trajs_hulls, dtype=object)
        n_traj, n_hulls_per_traj = trajs_hulls.shape

    # The following code simultaneously checks whether trajs_boxes is an ndarray
    # and also provides the dimensions.
    try:
        n_traj_boxes, n_boxes_per_traj = trajs_boxes.shape
    except AttributeError:
        trajs_boxes = np.array(trajs_boxes, dtype=object)
        n_traj_boxes, n_boxes_per_traj = trajs_boxes.shape

    if n_traj != n_traj_boxes:
        print('There is a mismatch in the number of agents in the trajs_hulls and trajs_boxes')
        return False

    if n_hulls_per_traj != (n_boxes_per_traj - 1):
        print('There is a mismatch in the number of hulls per trajectory and number of boxes per trajectory')
        print('n_hulls_per_traj should be (n_boxes_per_traj-1)')
        print(f'Current values: {n_hulls_per_traj=}, {n_boxes_per_traj=}')
        return False

    # ----------------------------------------------------------- #

    # To make sure that the original array is not modified
    trajs_boxes = trajs_boxes.copy()
    trajs_hulls = trajs_hulls.copy()

    # Swap actor action to move passed to get new trajs_boxes and trajs_hulls
    if move is not None:
        trajs_hulls, trajs_boxes = swap_agent_traj_in_trajs(trajs_hulls=trajs_hulls, trajs_boxes=trajs_boxes,
                                                            x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                                            traj2swap=actor, a2swap=move[0], theta2swap=move[1],
                                                            t_action=t_action, n_timesteps=n_timesteps,
                                                            l=l, plot_new_traj=False)

    # Get list of agents other than the affected
    not_affected = get_other_agents(n=n_traj, ii=affected)

    # Get traj_hulls and trajs_boxes for not_affected
    trajs_hulls_not_j = trajs_hulls[not_affected]
    trajs_boxes_not_j = trajs_boxes[not_affected]

    # Resolve collisions if actors move was updated.
    if move is not None:
        trajs_hulls_not_j, trajs_boxes_not_j = get_collision_free_trajs(trajs_hulls_not_j, trajs_boxes_not_j,
                                                                        plot_polygons_one_by_one=False)

    a_js = np.linspace(a_min, a_max, a_num)
    theta_js = np.linspace(theta_min, theta_max, theta_num)

    x0_j = x0[affected]
    y0_j = y0[affected]
    v0x_j = v0x[affected]
    v0y_j = v0y[affected]

    if plot_feasible_action_space:
        fig, configuration_space_ax = plotcf.get_new_fig(122)
        action_space_ax = fig.add_subplot(121)
        fig.tight_layout(pad=5.0)
    else:
        configuration_space_ax = None
        action_space_ax = None

    # count = 0

    aa_s = np.array(range(len(a_js) - 1))
    thetaa_s = np.array(range(len(theta_js) - 1))

    aa_, thertaa_ = np.meshgrid(aa_s, thetaa_s)
    aa_thetaa_s = list(zip(aa_.ravel(), thertaa_.ravel()))

    get_feasibility_of_subspace_partial = partial(get_feasibility_of_subspace_faster_aa_thetaa_tuple,
                                                  a_js=a_js, theta_js=theta_js,
                                                  actor=actor,
                                                  affected=affected,
                                                  not_affected=not_affected,
                                                  x0_j=x0_j, y0_j=y0_j, v0x_j=v0x_j, v0y_j=v0y_j,
                                                  n_hulls_per_traj=n_hulls_per_traj,
                                                  trajs_hulls_not_j=trajs_hulls_not_j,
                                                  n_timesteps=n_timesteps, t_action=t_action, l=l,
                                                  action_space_ax=action_space_ax,
                                                  configuration_space_ax=configuration_space_ax,
                                                  plot_feasible_action_space=
                                                  plot_feasible_action_space,
                                                  plot_intersections=plot_intersections,
                                                  plot_subspaces=plot_subspaces,
                                                  verbose_flag=verbose_flag)
    num_processes = cpu_count() - 2

    with Pool(processes=num_processes) as pool:  # Adjust the number of processes as needed
        # Use pool.map to apply the wrapper function to each actor in parallel
        feasibility = pool.map(get_feasibility_of_subspace_partial, aa_thetaa_s)

    count = feasibility.count(True)

    if plot_feasible_action_space:
        plotcf.plot_hulls_for_trajs(trajs_hulls_not_j, ax=configuration_space_ax, labels=not_affected,
                                    title=f'Feasible Configuration Space |\n'
                                          f'Actor: {actor}, Affected : {affected}',
                                    colour=plotcf.YELLOW)
        action_space_ax.set_title('Feasible Action Space', fontsize=12)
        action_space_ax.set_xlabel('Acceleration', fontsize=12)
        action_space_ax.set_ylabel('Theta', fontsize=12)

        # action_space_ax.set_aspect(1 / np.pi)

        match_ax1_size_2_ax2(ax1=action_space_ax, ax2=configuration_space_ax)

        plotcf.make_axes_gray(action_space_ax)

    return count


def match_ax1_size_2_ax2(ax1=None, ax2=None):
    """ Function to resize one ax to match the size of another ax to it side
    Only works for axs which are side by side.

    Parameters
    ----------
    ax1 : matplotlib ax
        ax which is to be resized
    ax2 : matplotlib ax
        ax to whose size the ax1 is to be resized to

    Returns
    -------

    """

    if ax1 is None:
        print('ax1 not passed!')
        return False

    if ax2 is None:
        print('ax2 not passed!')
        return False

    # Matching the size of the action space plot to that of the config space
    ax2_box = ax2.get_position()
    ax1_box = ax1.get_position()

    box_left = ax1_box.x0
    box_bottom = ax2_box.y0

    box_width = ax2_box.width
    box_height = ax2_box.height
    ax1.set_position([box_left, box_bottom, box_width, box_height])


def get_feasibility_of_subspace(a_js=None, aa=None,
                                theta_js=None, thetaa=None,
                                actor=None,
                                affected=None,
                                not_affected=None,
                                x0_j=None, y0_j=None, v0x_j=None, v0y_j=None,
                                n_hulls_per_traj=None,
                                trajs_hulls_not_j=None,
                                n_timesteps=None,
                                t_action=None,
                                l=None,
                                action_space_ax=None,
                                configuration_space_ax=None,
                                plot_feasible_action_space=False,
                                plot_intersections=False,
                                plot_subspaces=False,
                                verbose_flag=False):
    a_j_l = a_js[aa]
    a_j_u = a_js[aa + 1]
    theta_j_l = theta_js[thetaa]
    theta_j_u = theta_js[thetaa + 1]

    if plot_feasible_action_space:
        ext = [(a_j_l, theta_j_l), (a_j_u, theta_j_l),
               (a_j_u, theta_j_u), (a_j_l, theta_j_u),
               (a_j_l, theta_j_l)]
        a_theta_box = Polygon(ext)

    x_j_ll, y_j_ll, _ = make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
                                          t_action=t_action, n_timesteps=n_timesteps,
                                          a=a_j_l, theta=theta_j_l)
    x_j_lu, y_j_lu, _ = make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
                                          t_action=t_action, n_timesteps=n_timesteps,
                                          a=a_j_l, theta=theta_j_u)
    x_j_ul, y_j_ul, _ = make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
                                          t_action=t_action, n_timesteps=n_timesteps,
                                          a=a_j_u, theta=theta_j_l)
    x_j_uu, y_j_uu, _ = make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
                                          t_action=t_action, n_timesteps=n_timesteps,
                                          a=a_j_u, theta=theta_j_u)
    # ---------------------------------------------------------------------------------------
    # Get traj_boxes
    boxes_j_ll = get_boxes_for_a_traj(x=x_j_ll, y=y_j_ll, lx=l, ly=l)
    boxes_j_lu = get_boxes_for_a_traj(x=x_j_lu, y=y_j_lu, lx=l, ly=l)
    boxes_j_ul = get_boxes_for_a_traj(x=x_j_ul, y=y_j_ul, lx=l, ly=l)
    boxes_j_uu = get_boxes_for_a_traj(x=x_j_uu, y=y_j_uu, lx=l, ly=l)
    # ---------------------------------------------------------------------------------------
    # Get traj_hulls
    traj_hulls_j_ll = get_traj_hulls(boxes_j_ll)
    traj_hulls_j_lu = get_traj_hulls(boxes_j_lu)
    traj_hulls_j_ul = get_traj_hulls(boxes_j_ul)
    traj_hulls_j_uu = get_traj_hulls(boxes_j_uu)
    traj_hulls_j = zip(traj_hulls_j_ll, traj_hulls_j_lu,
                       traj_hulls_j_ul, traj_hulls_j_uu)
    traj_hulls_j = np.array([ii for ii in traj_hulls_j])
    last_boxes_j = np.array([boxes_j_ll[-1], boxes_j_lu[-1], boxes_j_ul[-1], boxes_j_uu[-1]]).reshape((1, 4))
    if verbose_flag: print(f'{last_boxes_j.shape=}')
    last_hull_j = get_hull_of_polygons(last_boxes_j)
    traj_hulls_of_hulls_j = get_hull_of_polygons(traj_hulls_j)
    # -------------------------------------------------------- #
    if plot_subspaces:
        fig, ax = plotcf.get_new_fig()

        # plotcf.plot_hulls_for_trajs(trajs_hulls_not_j, ax=ax, colour=plotcf.BLUE, labels=not_affected)
        plotcf.plot_hulls_for_trajs(trajs_hulls_not_j, ax=ax, labels=not_affected,
                                    title=f'Actor: {actor}, Affected : {affected}')
        plotcf.plot_traj_hulls(traj_hulls_j_ll, ax=ax, colour=plotcf.GRAY)
        plotcf.plot_traj_hulls(traj_hulls_j_lu, ax=ax, colour=plotcf.GRAY)
        plotcf.plot_traj_hulls(traj_hulls_j_ul, ax=ax, colour=plotcf.GRAY)
        plotcf.plot_traj_hulls(traj_hulls_j_uu, ax=ax, colour=plotcf.GRAY)
        plotcf.plot_traj_hulls(traj_hulls_of_hulls_j, ax=ax, colour=plotcf.RED, show_plot=True)
    # -------------------------------------------------------- #
    # Collision Checks
    traj_hulls_of_hulls_j = traj_hulls_of_hulls_j.reshape((1, n_hulls_per_traj))
    if verbose_flag: print(f'{trajs_hulls_not_j.shape=}, {traj_hulls_of_hulls_j.shape=}')
    traj_hulls_j_not_j = np.concatenate((traj_hulls_of_hulls_j, trajs_hulls_not_j))
    # The affected agent j would be under index 0 in this array.
    if verbose_flag: print(f'{traj_hulls_j_not_j.shape=}')
    collision = check_ego_collisions_as_time_rolls(n_hulls_per_traj=n_hulls_per_traj,
                                                   traj_hulls_j_not_j=traj_hulls_j_not_j,
                                                   plot_intersections=plot_intersections,
                                                   verbose_flag=verbose_flag)
    if verbose_flag: print(f'{collision=}')
    # if not collision:
    #     count = count + 1
    if verbose_flag: print(f'{traj_hulls_of_hulls_j.shape=}')
    if verbose_flag: print(f'{traj_hulls_of_hulls_j[0][-1]=}')
    if plot_feasible_action_space:
        if not collision:
            plotcf.plot_polygon(a_theta_box, ax=action_space_ax, add_points=False, color=plotcf.BLUE)
            # plotcf.plot_traj_hulls([traj_hulls_of_hulls_j[0][-1]], ax=configuration_space_ax,
            #                        colour=plotcf.BLUE, show_first=False)
            plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax,
                                   colour=plotcf.BLUE, show_first=False)
        else:
            plotcf.plot_polygon(a_theta_box, ax=action_space_ax, add_points=False, color=plotcf.RED)
            # plotcf.plot_traj_hulls([traj_hulls_of_hulls_j[0][-1]], ax=configuration_space_ax,
            #                        colour=plotcf.RED, show_first=False)
            plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax,
                                   colour=plotcf.RED, show_first=False)
    return not collision


@jit(forceobj=True)
def get_feasibility_of_subspace_faster_aa_thetaa_tuple(aa_thetaa=None,
                                                       a_js=None, theta_js=None,
                                                       actor=None,
                                                       affected=None,
                                                       not_affected=None,
                                                       x0_j=None, y0_j=None, v0x_j=None, v0y_j=None,
                                                       n_hulls_per_traj=None,
                                                       trajs_hulls_not_j=None,
                                                       n_timesteps=None,
                                                       t_action=None,
                                                       l=None,
                                                       action_space_ax=None,
                                                       configuration_space_ax=None,
                                                       plot_feasible_action_space=False,
                                                       plot_intersections=False,
                                                       plot_subspaces=False,
                                                       verbose_flag=False):
    aa, thetaa = aa_thetaa
    return get_feasibility_of_subspace_faster(aa=aa, thetaa=thetaa,
                                              a_js=a_js, theta_js=theta_js,
                                              actor=actor,
                                              affected=affected,
                                              not_affected=not_affected,
                                              x0_j=x0_j, y0_j=y0_j, v0x_j=v0x_j, v0y_j=v0y_j,
                                              n_hulls_per_traj=n_hulls_per_traj,
                                              trajs_hulls_not_j=trajs_hulls_not_j,
                                              n_timesteps=n_timesteps, t_action=t_action, l=l,
                                              action_space_ax=action_space_ax,
                                              configuration_space_ax=configuration_space_ax,
                                              plot_feasible_action_space=plot_feasible_action_space,
                                              plot_intersections=plot_intersections,
                                              plot_subspaces=plot_subspaces,
                                              verbose_flag=verbose_flag)


# @jit(forceobj=True)
# def get_feasibility_of_subspace_faster(a_js=None, aa=None,
#                                        theta_js=None, thetaa=None,
#                                        actor=None,
#                                        affected=None,
#                                        not_affected=None,
#                                        x0_j=None, y0_j=None, v0x_j=None, v0y_j=None,
#                                        n_hulls_per_traj=None,
#                                        trajs_hulls_not_j=None,
#                                        n_timesteps=None,
#                                        t_action=None,
#                                        l=None,
#                                        use_cache=True,
#                                        action_space_ax=None,
#                                        configuration_space_ax=None,
#                                        plot_feasible_action_space=False,
#                                        plot_intersections=False,
#                                        plot_subspaces=False,
#                                        verbose_flag=False):
#     a_j_l = a_js[aa]
#     a_j_u = a_js[aa + 1]
#     theta_j_l = theta_js[thetaa]
#     theta_j_u = theta_js[thetaa + 1]
#
#
#     x_j_ll, y_j_ll, _ = make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
#                                           t_action=t_action, n_timesteps=n_timesteps,
#                                           a=a_j_l, theta=theta_j_l)
#     x_j_lu, y_j_lu, _ = make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
#                                           t_action=t_action, n_timesteps=n_timesteps,
#                                           a=a_j_l, theta=theta_j_u)
#     x_j_ul, y_j_ul, _ = make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
#                                           t_action=t_action, n_timesteps=n_timesteps,
#                                           a=a_j_u, theta=theta_j_l)
#     x_j_uu, y_j_uu, _ = make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
#                                           t_action=t_action, n_timesteps=n_timesteps,
#                                           a=a_j_u, theta=theta_j_u)
#     # ---------------------------------------------------------------------------------------
#     # Get traj_boxes
#     if use_cache:
#         # print('Using Cache!')
#         boxes_j_ll = get_boxes_for_a_traj_caching_wrapper(x=x_j_ll, y=y_j_ll, lx=l, ly=l)
#         boxes_j_lu = get_boxes_for_a_traj_caching_wrapper(x=x_j_lu, y=y_j_lu, lx=l, ly=l)
#         boxes_j_ul = get_boxes_for_a_traj_caching_wrapper(x=x_j_ul, y=y_j_ul, lx=l, ly=l)
#         boxes_j_uu = get_boxes_for_a_traj_caching_wrapper(x=x_j_uu, y=y_j_uu, lx=l, ly=l)
#     else:
#         boxes_j_ll = get_boxes_for_a_traj(x=x_j_ll, y=y_j_ll, lx=l, ly=l)
#         boxes_j_lu = get_boxes_for_a_traj(x=x_j_lu, y=y_j_lu, lx=l, ly=l)
#         boxes_j_ul = get_boxes_for_a_traj(x=x_j_ul, y=y_j_ul, lx=l, ly=l)
#         boxes_j_uu = get_boxes_for_a_traj(x=x_j_uu, y=y_j_uu, lx=l, ly=l)
#     # --------------------------------------------------------------------------
#     # ---------------------------------------------------------------------------------------
#     # Get traj_hulls
#     traj_hulls_j_ll = get_traj_hulls(boxes_j_ll)
#     traj_hulls_j_lu = get_traj_hulls(boxes_j_lu)
#     traj_hulls_j_ul = get_traj_hulls(boxes_j_ul)
#     traj_hulls_j_uu = get_traj_hulls(boxes_j_uu)
#     traj_hulls_j = zip(traj_hulls_j_ll, traj_hulls_j_lu,
#                        traj_hulls_j_ul, traj_hulls_j_uu)
#     traj_hulls_j = np.array([ii for ii in traj_hulls_j])
#     last_boxes_j = np.array([boxes_j_ll[-1], boxes_j_lu[-1], boxes_j_ul[-1], boxes_j_uu[-1]]).reshape((1, 4))
#
#     last_hull_j = get_hull_of_polygons(last_boxes_j)
#     traj_hulls_of_hulls_j = get_hull_of_polygons(traj_hulls_j)
#
#
#     # Collision Checks
#     traj_hulls_of_hulls_j = traj_hulls_of_hulls_j.reshape((1, n_hulls_per_traj))
#     # if verbose_flag: print(f'{trajs_hulls_not_j.shape=}, {traj_hulls_of_hulls_j.shape=}')
#     traj_hulls_j_not_j = np.concatenate((traj_hulls_of_hulls_j, trajs_hulls_not_j))
#     # The affected agent j would be under index 0 in this array.
#     # if verbose_flag: print(f'{traj_hulls_j_not_j.shape=}')
#     collision = check_ego_collisions_as_time_rolls(n_hulls_per_traj=n_hulls_per_traj,
#                                                    traj_hulls_j_not_j=traj_hulls_j_not_j,
#                                                    plot_intersections=False,
#                                                    verbose_flag=False)
#
#     return collision


@jit(forceobj=True)
def get_feasibility_of_subspace_faster(a_js=None, aa=None,
                                       theta_js=None, thetaa=None,
                                       actor=None,
                                       affected=None,
                                       not_affected=None,
                                       x0_j=None, y0_j=None, v0x_j=None, v0y_j=None,
                                       n_hulls_per_traj=None,
                                       trajs_hulls_not_j=None,
                                       obstacle=None,
                                       n_timesteps=None,
                                       t_action=None,
                                       l=None,
                                       use_cache=True,
                                       action_space_ax=None,
                                       configuration_space_ax=None,
                                       action_space_ax_stripped=None,
                                       configuration_space_ax_stripped=None,
                                       plot_feasible_action_space=False,
                                       plot_intersections=False,
                                       plot_subspaces=False,
                                       verbose_flag=False):
    a_j_l = a_js[aa]
    a_j_u = a_js[aa + 1]
    theta_j_l = theta_js[thetaa]
    theta_j_u = theta_js[thetaa + 1]

    if plot_feasible_action_space:
        ext = [(a_j_l, theta_j_l), (a_j_u, theta_j_l),
               (a_j_u, theta_j_u), (a_j_l, theta_j_u),
               (a_j_l, theta_j_l)]
        a_theta_box = Polygon(ext)

    if abs(a_j_u - a_j_l) <= A_STEP_THRESHOLD:
        a_j_subs = np.array([a_j_l, a_j_u])
        n_a_subs = 2
    else:
        n_a_subs = np.ceil((a_j_u - a_j_l) / A_STEP_THRESHOLD).astype(int)
        a_j_subs = np.linspace(a_j_l, a_j_u, num=n_a_subs)
    # print('a_j_subs: ', str(a_j_subs))

    if abs(theta_j_u - theta_j_l) <= THETA_STEP_THRESHOLD:
        theta_j_subs = np.array([theta_j_l, theta_j_u])
        n_theta_subs = 2
    else:
        n_theta_subs = np.ceil((theta_j_u - theta_j_l) / THETA_STEP_THRESHOLD).astype(int)
        theta_j_subs = np.linspace(theta_j_l, theta_j_u, num=n_theta_subs)

    n_subs = n_a_subs * n_theta_subs

    # ---------------------------------------------------------------------------------------
    # ----------------------Minimum resolution in subspaces----------------------------------
    xy_j_s = [make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
                                t_action=t_action, n_timesteps=n_timesteps,
                                a=a_, theta=theta_)
              for a_ in a_j_subs
              for theta_ in theta_j_subs]

    # Get traj_boxes
    if use_cache:
        boxes_j = [get_boxes_for_a_traj_caching_wrapper(x=x_j_, y=y_j_, lx=l, ly=l)
                   for x_j_, y_j_, _ in xy_j_s]
    else:
        boxes_j = [get_boxes_for_a_traj(x=x_j_, y=y_j_, lx=l, ly=l)
                   for x_j_, y_j_, _ in xy_j_s]

    # Get traj_hulls
    traj_hulls_j_ = [get_traj_hulls(boxes) for boxes in boxes_j]
    traj_hulls_j = np.array(traj_hulls_j_).transpose()

    last_boxes_j = np.array([boxes[-1] for boxes in boxes_j]).reshape((1, n_subs))
    last_hull_j = get_hull_of_polygons(last_boxes_j)
    traj_hulls_of_hulls_j = get_hull_of_polygons(traj_hulls_j)

    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------

    if plot_subspaces:
        fig, ax = plotcf.get_new_fig()

        plotcf.plot_hulls_for_trajs(trajs_hulls_not_j, ax=ax, labels=not_affected,
                                    title='Actor: ' + str(actor) + ', Affected : ' + str(affected))

        for traj_hulls_ in traj_hulls_j_:
            plotcf.plot_traj_hulls(traj_hulls_, ax=ax, colour=plotcf.GRAY)

        plotcf.plot_traj_hulls(traj_hulls_of_hulls_j, ax=ax, colour=plotcf.RED, show_plot=True)

    # # -------------------------------------------------------- #
    # Collision Checks
    traj_hulls_of_hulls_j = traj_hulls_of_hulls_j.reshape((1, n_hulls_per_traj))
    traj_hulls_j_not_j = np.concatenate((traj_hulls_of_hulls_j, trajs_hulls_not_j))

    # collision = check_collision_with_static_obstacle(obstacle, traj_hulls_of_hulls_j)

    collision = False

    if obstacle is not None:
        hull_over_time_j = get_hull_of_polygons(traj_hulls_of_hulls_j)
        collision = obstacle.intersects(hull_over_time_j)

        # # Checking each hull separately is slower that generating the hull and chehcking whether it collides
        # for hull in np.flip(traj_hulls_of_hulls_j.squeeze()):
        #     collision = obstacle.intersects(hull)
        #     if collision:
        #         break

    if not collision:  # Only do this if there is not collision with static obstacle
        # The affected agent j would be under index 0 in this array.
        collision = check_ego_collisions_as_time_rolls(n_hulls_per_traj=n_hulls_per_traj,
                                                       traj_hulls_j_not_j=traj_hulls_j_not_j,
                                                       plot_intersections=False,
                                                       verbose_flag=False)

    if plot_feasible_action_space:
        if not collision:
            plotcf.plot_polygon(a_theta_box, ax=action_space_ax, add_points=False, color=plotcf.BLUE)
            # plotcf.plot_traj_hulls([traj_hulls_of_hulls_j[0][-1]], ax=configuration_space_ax,
            #                        colour=plotcf.BLUE, show_first=False)
            plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax,
                                   colour=plotcf.BLUE, show_first=False)
            if action_space_ax_stripped:
                plotcf.plot_polygon(a_theta_box, ax=action_space_ax_stripped, add_points=False, color=plotcf.BLUE)
            if configuration_space_ax_stripped:
                plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax_stripped,
                                       colour=plotcf.BLUE, show_first=False)

        else:
            plotcf.plot_polygon(a_theta_box, ax=action_space_ax, add_points=False, color=plotcf.RED)
            # plotcf.plot_traj_hulls([traj_hulls_of_hulls_j[0][-1]], ax=configuration_space_ax,
            #                        colour=plotcf.RED, show_first=False)
            plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax,
                                   colour=plotcf.RED, show_first=False)
            if action_space_ax_stripped:
                plotcf.plot_polygon(a_theta_box, ax=action_space_ax_stripped, add_points=False, color=plotcf.RED)
            if configuration_space_ax_stripped:
                plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax_stripped,
                                       colour=plotcf.RED, show_first=False)

    return not collision


# @jit(forceobj=True)
# def check_collision_with_static_obstacle(obstacle, traj_hulls_of_hulls_j):
#     # Check whether the subspace hull intersects with static obstacles
#     if obstacle is not None:
#         # hull_over_time_j = get_hull_of_polygons(traj_hulls_of_hulls_j)
#         # collision = obstacle.intersects(hull_over_time_j)
#
#         # Checking each hull separately is slower that generating the hull and chehcking whether it collides
#         for hull in np.flip(traj_hulls_of_hulls_j.squeeze()):
#             collision = obstacle.intersects(hull)
#             if collision:
#                 break
#     return collision


@jit(forceobj=True)
def plot_feasible_action_and_configuration_subspaces(feasibility=None, aa_thetaa_list=None,
                                                     a_js=None,
                                                     theta_js=None,
                                                     x0_j=None, y0_j=None, v0x_j=None, v0y_j=None,
                                                     n_timesteps=None,
                                                     t_action=None,
                                                     l=None,
                                                     use_cache=True,
                                                     action_space_ax=None,
                                                     configuration_space_ax=None,
                                                     action_space_ax_stripped=None,
                                                     configuration_space_ax_stripped=None,
                                                     highlight_ss=[92],
                                                     unify_action_spaces_in_stripped=True
                                                     ):
    if unify_action_spaces_in_stripped:
        action_space_ax_stripped_ = None
    else:
        action_space_ax_stripped_ = action_space_ax_stripped

    for ss in range(len(aa_thetaa_list)):
        # Set linewidth for non-highlighted cases
        # linewidth = None
        linewidth = 0.5

        plot_each_subspace_in_action_config_space_plots(a_js=a_js,
                                                        aa_thetaa_list=aa_thetaa_list,
                                                        action_space_ax=action_space_ax,
                                                        action_space_ax_stripped=action_space_ax_stripped_,
                                                        configuration_space_ax=configuration_space_ax,
                                                        configuration_space_ax_stripped=configuration_space_ax_stripped,
                                                        feasibility=feasibility,
                                                        l=l,
                                                        linewidth=linewidth,
                                                        n_timesteps=n_timesteps,
                                                        ss=ss,
                                                        t_action=t_action,
                                                        theta_js=theta_js,
                                                        use_cache=use_cache,
                                                        v0x_j=v0x_j,
                                                        v0y_j=v0y_j,
                                                        x0_j=x0_j,
                                                        y0_j=y0_j)

    for ss in highlight_ss:

        # aa, thetaa = aa_thetaa_list[ss]
        # a_j_l = a_js[aa]
        # a_j_u = a_js[aa + 1]
        # theta_j_l = theta_js[thetaa]
        # theta_j_u = theta_js[thetaa + 1]
        #
        # print('a_j_l=', a_j_l)
        # print('a_j_u=', a_j_u)
        # print('theta_j_l=', theta_j_l)
        # print('theta_j_u=', theta_j_u)

        # Set linewidth for highlighted cases
        linewidth = 2

        plot_each_subspace_in_action_config_space_plots(a_js=a_js,
                                                        aa_thetaa_list=aa_thetaa_list,
                                                        action_space_ax=action_space_ax,
                                                        configuration_space_ax=configuration_space_ax,
                                                        feasibility=feasibility,
                                                        l=l,
                                                        linewidth=linewidth,
                                                        n_timesteps=n_timesteps,
                                                        ss=ss,
                                                        t_action=t_action,
                                                        theta_js=theta_js,
                                                        use_cache=use_cache,
                                                        v0x_j=v0x_j,
                                                        v0y_j=v0y_j,
                                                        x0_j=x0_j,
                                                        y0_j=y0_j,
                                                        zorder=10,
                                                        highlight=True)

        for _ in range(4):
            plot_each_subspace_in_action_config_space_plots(a_js=a_js,
                                                            aa_thetaa_list=aa_thetaa_list,
                                                            action_space_ax=action_space_ax,
                                                            configuration_space_ax=configuration_space_ax,
                                                            feasibility=feasibility,
                                                            l=l,
                                                            linewidth=linewidth,
                                                            n_timesteps=n_timesteps,
                                                            ss=ss,
                                                            t_action=t_action,
                                                            theta_js=theta_js,
                                                            use_cache=use_cache,
                                                            v0x_j=v0x_j,
                                                            v0y_j=v0y_j,
                                                            x0_j=x0_j,
                                                            y0_j=y0_j,
                                                            zorder=10)

    if unify_action_spaces_in_stripped and action_space_ax_stripped:
        plot_unified_feasible_action_space(a_js, aa_thetaa_list, action_space_ax_stripped, feasibility, linewidth,
                                           theta_js)

    plotcf.make_axes_gray(action_space_ax)
    if action_space_ax_stripped:
        plotcf.make_axes_gray(action_space_ax_stripped)


def plot_unified_feasible_action_space(a_js, aa_thetaa_list, action_space_ax_stripped, feasibility, linewidth, theta_js,
                                       zorder=2):
    ss_feasible = []
    ss_collides = []
    for ss in range(len(aa_thetaa_list)):
        aa, thetaa = aa_thetaa_list[ss]
        collision = not feasibility[ss]
        a_j_l = a_js[aa]
        a_j_u = a_js[aa + 1]
        theta_j_l = theta_js[thetaa]
        theta_j_u = theta_js[thetaa + 1]
        ext = [(a_j_l, theta_j_l), (a_j_u, theta_j_l),
               (a_j_u, theta_j_u), (a_j_l, theta_j_u),
               (a_j_l, theta_j_l), (a_j_u, theta_j_l)]
        a_theta_box = Polygon(ext)

        if collision:
            ss_collides.append(a_theta_box)
        else:
            ss_feasible.append(a_theta_box)
    action_space_collides = MultiPolygon(ss_collides).buffer(0.0001)
    action_space_feasible = MultiPolygon(ss_feasible).buffer(0.0001)

    feasible_facecolor = list(plotcf.mcolors.to_rgba(plotcf.BLUE))
    feasible_facecolor[-1] = 0.8
    feasible_facecolor = tuple(feasible_facecolor)

    plotcf.plot_polygon(action_space_feasible, ax=action_space_ax_stripped, add_points=False,
                        linewidth=linewidth, facecolor=feasible_facecolor,
                        color=plotcf.BLUE, zorder=zorder + 5)
    plotcf.plot_polygon(action_space_collides, ax=action_space_ax_stripped, add_points=False,
                        zorder=zorder,
                        linewidth=linewidth,
                        color=plotcf.RED, hatch='///')


def plot_each_subspace_in_action_config_space_plots(a_js, aa_thetaa_list, action_space_ax,
                                                    configuration_space_ax, feasibility,
                                                    l, linewidth, n_timesteps, ss, t_action, theta_js, use_cache, v0x_j,
                                                    v0y_j, x0_j, y0_j,
                                                    highlight=False,
                                                    zorder=2,
                                                    configuration_space_ax_stripped=None,
                                                    custom_feasible_facecolor=True,
                                                    action_space_ax_stripped=None):
    aa, thetaa = aa_thetaa_list[ss]
    collision = not feasibility[ss]
    a_j_l = a_js[aa]
    a_j_u = a_js[aa + 1]
    theta_j_l = theta_js[thetaa]
    theta_j_u = theta_js[thetaa + 1]
    ext = [(a_j_l, theta_j_l), (a_j_u, theta_j_l),
           (a_j_u, theta_j_u), (a_j_l, theta_j_u),
           (a_j_l, theta_j_l), (a_j_u, theta_j_l)]
    a_theta_box = Polygon(ext)
    if abs(a_j_u - a_j_l) <= A_STEP_THRESHOLD:
        a_j_subs = np.array([a_j_l, a_j_u])
        n_a_subs = 2
    else:
        n_a_subs = np.ceil((a_j_u - a_j_l) / A_STEP_THRESHOLD).astype(int)
        a_j_subs = np.linspace(a_j_l, a_j_u, num=n_a_subs)
    # print('a_j_subs: ', str(a_j_subs))
    if abs(theta_j_u - theta_j_l) <= THETA_STEP_THRESHOLD:
        theta_j_subs = np.array([theta_j_l, theta_j_u])
        n_theta_subs = 2
    else:
        n_theta_subs = np.ceil((theta_j_u - theta_j_l) / THETA_STEP_THRESHOLD).astype(int)
        theta_j_subs = np.linspace(theta_j_l, theta_j_u, num=n_theta_subs)
    n_subs = n_a_subs * n_theta_subs
    # ---------------------------------------------------------------------------------------
    # ----------------------Minimum resolution in subspaces----------------------------------
    xy_j_s = [make_trajectories(x0=x0_j, y0=y0_j, v0x=v0x_j, v0y=v0y_j,
                                t_action=t_action, n_timesteps=n_timesteps,
                                a=a_, theta=theta_)
              for a_ in a_j_subs
              for theta_ in theta_j_subs]
    # Get traj_boxes
    if use_cache:
        boxes_j = [get_boxes_for_a_traj_caching_wrapper(x=x_j_, y=y_j_, lx=l, ly=l)
                   for x_j_, y_j_, _ in xy_j_s]
    else:
        boxes_j = [get_boxes_for_a_traj(x=x_j_, y=y_j_, lx=l, ly=l)
                   for x_j_, y_j_, _ in xy_j_s]
    # Get traj_hulls
    traj_hulls_j_ = [get_traj_hulls(boxes) for boxes in boxes_j]
    traj_hulls_j = np.array(traj_hulls_j_).transpose()
    last_boxes_j = np.array([boxes[-1] for boxes in boxes_j]).reshape((1, n_subs))
    last_hull_j = get_hull_of_polygons(last_boxes_j)
    # traj_hulls_of_hulls_j = get_hull_of_polygons(traj_hulls_j)
    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------
    if custom_feasible_facecolor:
        feasible_facecolor = list(plotcf.mcolors.to_rgba(plotcf.BLUE))
        feasible_facecolor[-1] = 0.8
        feasible_facecolor = tuple(feasible_facecolor)
    else:
        feasible_facecolor = None
    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------

    if highlight:
        plotcf.plot_polygon(a_theta_box, ax=action_space_ax, add_points=False, color=plotcf.WHITE,
                            zorder=zorder,
                            linewidth=linewidth + 2)

        last_hull_j_buffed = np.array([last_hull_j[0].buffer(0.1)])

        plotcf.plot_traj_hulls(last_hull_j_buffed, ax=configuration_space_ax,
                               colour=plotcf.WHITE, show_first=False,
                               linewidth=linewidth + 2)

        if action_space_ax_stripped:
            plotcf.plot_polygon(a_theta_box, ax=action_space_ax_stripped, add_points=False,
                                zorder=zorder,
                                linewidth=linewidth + 2,
                                color=plotcf.WHITE)
        if configuration_space_ax_stripped:
            plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax_stripped,
                                   linewidth=linewidth + 2,
                                   colour=plotcf.WHITE, show_first=False)

    if not collision:
        plotcf.plot_polygon(a_theta_box, ax=action_space_ax, add_points=False, color=plotcf.BLUE, zorder=zorder + 5,
                            linewidth=linewidth, facecolor=feasible_facecolor)

        plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax,
                               colour=plotcf.BLUE, facecolor=feasible_facecolor, show_first=False,
                               linewidth=linewidth)
        if action_space_ax_stripped:
            plotcf.plot_polygon(a_theta_box, ax=action_space_ax_stripped, add_points=False,
                                linewidth=linewidth,
                                color=plotcf.BLUE, facecolor=feasible_facecolor, zorder=zorder + 5)
        if configuration_space_ax_stripped:
            plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax_stripped,
                                   linewidth=linewidth,
                                   colour=plotcf.BLUE, facecolor=feasible_facecolor, show_first=False)

    else:
        plotcf.plot_polygon(a_theta_box, ax=action_space_ax, add_points=False, color=plotcf.RED,
                            zorder=zorder,
                            hatch='///', linewidth=linewidth)

        plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax,
                               colour=plotcf.RED, hatch='///', show_first=False,
                               linewidth=linewidth)

        if action_space_ax_stripped:
            plotcf.plot_polygon(a_theta_box, ax=action_space_ax_stripped, add_points=False,
                                zorder=zorder,
                                linewidth=linewidth,
                                color=plotcf.RED, hatch='///')
        if configuration_space_ax_stripped:
            plotcf.plot_traj_hulls(last_hull_j, ax=configuration_space_ax_stripped,
                                   linewidth=linewidth,
                                   colour=plotcf.RED, hatch='///', show_first=False)


# @jit(forceobj=True)
def get_fear(nn=None, mdr_a=None, mdr_theta=None,
             a_min=None, a_max=None, a_num=None,
             theta_min=None, theta_max=None, theta_num=None,
             trajs_boxes=None,
             trajs_hulls=None,
             obstacle=None,
             x0=None, y0=None, v0x=None, v0y=None,
             t_action=None, n_timesteps=None,
             verbose=True,
             l=None, plot_subspaces=False, plot_feasible_action_space=False,
             return_feasibility=False):
    """
    Calculate the Feasible Action-Space Reduction values for all agents within a spatial interaction over a time-window.

    Parameters
    ----------
    nn : int
        Number of agents.
    mdr_a : ndarray
        Magnitude components of acceleration for the agents' Move de Rigueurs.
    mdr_theta : ndarray
        Direction component of acceleration for the agents' Move de Rigueurs.
    a_min : float
        Lower limit of the magnitude of acceleration for the affected agent.
    a_max : float
        Upper limit of the magnitude of acceleration for the affected agent.
    a_num : int
        Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
    theta_min : float
        Lower limit of the angle for direction of acceleration for the affected agent.
    theta_max : float
        Upper limit of the angle for direction of acceleration for the affected agent.
    theta_num : int
        Number of intervals/subdivisions for direction of acceleration for the affected agent.
    trajs_boxes : list
        Trajectory boxes for agents for each time instant.
    trajs_hulls : list
        Trajectory hulls for agents for each time interval (convex hulls of trajectory boxes).
    obstacle : MultiPolygon (optional)
        Static obstacles as a MultiPolygon
    x0 : ndarray
        Starting x locations for agents.
    y0 : ndarray
        Starting y locations for agents.
    v0x : ndarray
        Starting velocity in the x-direction for agents.
    v0y : ndarray
        Starting velocity in the y-direction for agents.
    t_action : float
        Time window to be considered.
    n_timesteps : int
        Number of timesteps to consider within the time window.
    l : float
        Length of an agent along one axis (assuming same length in x and y directions).
    plot_subspaces : bool, optional
        Whether to plot subspaces.
    plot_feasible_action_space : bool, optional
        Whether to plot feasible action space.
    return_feasibility : bool, optional
        Whether to return the dict with feasibility information of the subspaces.

    Returns
    -------
    fear : ndarray
        Array containing Feasible Action-Space Reduction values.
    """
    # Disable garbage collection
    gc.disable()

    # fear = np.array([get_fear_for_actor(actor=actor,
    #                                     a_max=a_max, a_min=a_min, a_num=a_num,
    #                                     theta_max=theta_max, theta_min=theta_min, theta_num=theta_num,
    #                                     mdr_a=mdr_a, mdr_theta=mdr_theta,
    #                                     n_timesteps=n_timesteps, t_action=t_action, nn=nn, l=l,
    #                                     trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls,
    #                                     v0x=v0x, v0y=v0y, x0=x0, y0=y0,
    #                                     plot_feasible_action_space=plot_feasible_action_space,
    #                                     plot_subspaces=plot_subspaces).squeeze()
    #                  for actor in range(nn)])

    get_fear_for_actor_partial = partial(get_fear_for_actor,
                                         a_max=a_max, a_min=a_min, a_num=a_num,
                                         theta_max=theta_max, theta_min=theta_min, theta_num=theta_num,
                                         mdr_a=mdr_a, mdr_theta=mdr_theta,
                                         n_timesteps=n_timesteps, t_action=t_action, nn=nn, l=l,
                                         trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls,
                                         obstacle=obstacle,
                                         v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                         plot_feasible_action_space=plot_feasible_action_space,
                                         plot_subspaces=plot_subspaces,
                                         return_feasibility=return_feasibility
                                         )

    # num_processes = cpu_count() - 2
    num_processes = cpu_count() // 2
    # num_processes = cpu_count()

    if nn > MULTI_PROCESSING_THRESHOLD_GET_FEAR:
        print(f'{num_processes=}')

        actors = range(nn)
        with Pool(processes=num_processes) as pool:  # Adjust the number of processes as needed
            # Use pool.map to apply the wrapper function to each actor in parallel
            fears = pool.map(get_fear_for_actor_partial, actors)

        if return_feasibility:
            fears, feasibilities = zip(*fears)
            feasibility = np.array(feasibilities)

        fear = np.array(fears)

    else:  # Do not use multiprocessing if nn is small
        print('No multiprocessing.')
        if return_feasibility:
            fear_feasibility = [get_fear_for_actor_partial(actor=actor) for actor in range(nn)]
            fear, feasibility = zip(*fear_feasibility)

            # Converting to numpy arrays
            fear = np.array(fear)
            feasibility = np.array(feasibility)

        else:
            fear = np.array([get_fear_for_actor_partial(actor=actor) for actor in range(nn)])

    # fear = np.zeros((nn, nn))
    #
    # for actor in range(nn):
    #     fear_actor = get_fear_for_actor(actor=actor,
    #                                     a_max=a_max, a_min=a_min, a_num=a_num,
    #                                     theta_max=theta_max, theta_min=theta_min, theta_num=theta_num,
    #                                     mdr_a=mdr_a, mdr_theta=mdr_theta,
    #                                     n_timesteps=n_timesteps, t_action=t_action, nn=nn, l=l,
    #                                     trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls,
    #                                     v0x=v0x, v0y=v0y, x0=x0, y0=y0,
    #                                     plot_feasible_action_space=plot_feasible_action_space,
    #                                     plot_subspaces=plot_subspaces)
    #
    #     fear[actor, :] = fear_actor

    if verbose:
        print(get_boxes_for_a_traj_cached.cache_info())
        print(get_hull_of_polygons_of_iith_traj_cached.cache_info())
        print(get_other_agents.cache_info())

    # Re-enable garbage collection
    gc.enable()

    if return_feasibility:
        return fear, feasibility

    return fear


# @jit(forceobj=True)  # Does not make it faster.
def get_fear_for_actor(actor=None,
                       a_min=None, a_max=None, a_num=None,
                       theta_min=None, theta_max=None, theta_num=None,
                       mdr_a=None, mdr_theta=None,
                       n_timesteps=None, t_action=None, nn=None, l=None,
                       trajs_boxes=None, trajs_hulls=None, obstacle=None,
                       v0x=None, v0y=None, x0=None, y0=None,
                       plot_feasible_action_space=False,
                       plot_subspaces=False,
                       return_feasibility=False):
    """
    Calculate the Feasible Action-Space Reduction values of one actor agent on all other affected agents
     within a spatial interaction over a time-window.

    Parameters
    ----------
    actor : int
        The actor agent identifier.
    a_min : float
        Lower limit of the magnitude of acceleration for the affected agent.
    a_max : float
        Upper limit of the magnitude of acceleration for the affected agent.
    a_num : int
        Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
    theta_min : float
        Lower limit of the angle for direction of acceleration for the affected agent.
    theta_max : float
        Upper limit of the angle for direction of acceleration for the affected agent.
    theta_num : int
        Number of intervals/subdivisions for direction of acceleration for the affected agent.
    mdr_a : ndarray
        Magnitude components of acceleration for the agents' Move de Rigueurs.
    mdr_theta : ndarray
        Direction component of acceleration for the agents' Move de Rigueurs.
    n_timesteps : int
        Number of timesteps to consider within the time window.
    t_action : float
        Time window to be considered.
    nn : int
        Number of agents.
    l : float
        Length of an agent along one axis (assuming same length in x and y directions).
    trajs_boxes : list
        Trajectory boxes for agents for each time instant.
    trajs_hulls : list
        Trajectory hulls for agents for each time interval (convex hulls of trajectory boxes).
    obstacle : MultiPolygon (optional)
        Static obstacles as a MultiPolygon
    v0x : ndarray
        Starting velocity in the x-direction for agents.
    v0y : ndarray
        Starting velocity in the y-direction for agents.
    x0 : ndarray
        Starting x locations for agents.
    y0 : ndarray
        Starting y locations for agents.
    plot_feasible_action_space : bool, optional
        Whether to plot the feasible action space.
    plot_subspaces : bool, optional
        Whether to plot the subspaces.

    Returns
    -------
    fear_for_actor : ndarray
        Array containing Feasible Action-Space Reduction values for the specified actor.
    """
    others = get_other_agents(n=nn, ii=actor)

    # print(f'{actor=}, \nn{others=}')
    print('actor=', str(actor), '\n others=', str(others))

    # mdr = np.array([mdr_a[actor], mdr_theta[actor]]).squeeze()
    # mdr = np.array([mdr_a[actor], mdr_theta[actor]]).squeeze()
    mdr = np.column_stack([mdr_a[actor].squeeze(), mdr_theta[actor].squeeze()])

    if return_feasibility:
        fear_feasibility_actor = [get_fear_actor_affected(actor=actor, affected=affected,
                                                          a_max=a_max, a_min=a_min, a_num=a_num,
                                                          mdr=mdr, n_timesteps=n_timesteps, t_action=t_action, l=l,
                                                          theta_max=theta_max, theta_min=theta_min, theta_num=theta_num,
                                                          trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls,
                                                          obstacle=obstacle,
                                                          v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                          plot_feasible_action_space=plot_feasible_action_space,
                                                          plot_subspaces=plot_subspaces,
                                                          return_feasibility=return_feasibility)
                                  if affected in others
                                  else get_feal(affected=affected, nn=nn, mdr_a=mdr_a, mdr_theta=mdr_theta,
                                                a_max=a_max, a_min=a_min, a_num=a_num,
                                                n_timesteps=n_timesteps, t_action=t_action, l=l,
                                                theta_max=theta_max, theta_min=theta_min, theta_num=theta_num,
                                                trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls,
                                                obstacle=obstacle,
                                                v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                plot_feasible_action_space=plot_feasible_action_space,
                                                plot_subspaces=plot_subspaces,
                                                return_feasibility=return_feasibility)
                                  for affected in range(nn)]

        fear_actor, feasibility_actor = zip(*fear_feasibility_actor)
        fear_actor = np.array(fear_actor)
        feasibility_actor = np.array(feasibility_actor)

        # Re-enable garbage collection
        gc.enable()

        # Disable garbage collection
        gc.disable()

        return fear_actor, feasibility_actor

    # If return_feasibility is False, only return the fear_actor
    fear_actor = np.array([get_fear_actor_affected(actor=actor, affected=affected,
                                                   a_max=a_max, a_min=a_min, a_num=a_num,
                                                   mdr=mdr, n_timesteps=n_timesteps, t_action=t_action, l=l,
                                                   theta_max=theta_max, theta_min=theta_min, theta_num=theta_num,
                                                   trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls,
                                                   obstacle=obstacle,
                                                   v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                                   plot_feasible_action_space=plot_feasible_action_space,
                                                   plot_subspaces=plot_subspaces)
                           if affected in others
                           else get_feal(affected=affected, nn=nn, mdr_a=mdr_a, mdr_theta=mdr_theta,
                                         a_max=a_max, a_min=a_min, a_num=a_num,
                                         n_timesteps=n_timesteps, t_action=t_action, l=l,
                                         theta_max=theta_max, theta_min=theta_min, theta_num=theta_num,
                                         trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls,
                                         obstacle=obstacle,
                                         v0x=v0x, v0y=v0y, x0=x0, y0=y0,
                                         plot_feasible_action_space=plot_feasible_action_space,
                                         plot_subspaces=plot_subspaces)
                           for affected in range(nn)])

    # Re-enable garbage collection
    gc.enable()

    # Disable garbage collection
    gc.disable()

    return fear_actor


@jit(forceobj=True)
def get_feal(affected=None, nn=None, mdr_a=None, mdr_theta=None,
             a_min=None, a_max=None, a_num=None,
             theta_min=None, theta_max=None, theta_num=None,
             n_timesteps=None, t_action=None, l=None,
             trajs_boxes=None, trajs_hulls=None, obstacle=None,
             v0x=None, v0y=None, x0=None, y0=None,
             return_feasibility=False,
             plot_feasible_action_space=False,
             plot_subspaces=False):
    """
    Calculate the Feasible Action-Space Left values of an affected agent
    within a spatial interaction over a time-window.

    Parameters
    ----------
    affected : int
        The affected agent identifier.
    a_min : float
        Lower limit of the magnitude of acceleration for the affected agent.
    a_max : float
        Upper limit of the magnitude of acceleration for the affected agent.
    a_num : int
        Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
    theta_min : float
        Lower limit of the angle for direction of acceleration for the affected agent.
    theta_max : float
        Upper limit of the angle for direction of acceleration for the affected agent.
    theta_num : int
        Number of intervals/subdivisions for direction of acceleration for the affected agent.
    mdr : list
        The move de rigueur of the actor agent (contains [magnitude, direction] of the constant acceleration move).
    n_timesteps : int
        Number of timesteps to consider within the time window.
    t_action : float
        Time window to be considered.
    l : float
        Length of an agent along one axis (assuming same length in x and y directions).
    trajs_boxes : list
        Trajectory boxes for agents for each time instant.
    trajs_hulls : list
        Trajectory hulls for agents for each time interval (convex hulls of trajectory boxes).
    obstacle : MultiPolygon (optional)
        Static obstacles as a MultiPolygon
    v0x : ndarray
        Starting velocity in the x-direction for agents.
    v0y : ndarray
        Starting velocity in the y-direction for agents.
    x0 : ndarray
        Starting x locations for agents.
    y0 : ndarray
        Starting y locations for agents.
    plot_feasible_action_space : bool, optional
        Whether to plot the feasible action space.
    plot_subspaces : bool, optional
        Whether to plot the subspaces
    return_feasibility : bool, optional
        Whether to return the dict with feasibility information of the subspaces.

    Returns
    -------
    feal_actor_to_affected : ndarray
        Feasible Action-Space Remaining/Left values of the affected agent.
    """

    if affected is None:
        print('No affected agent ID passed!')
        return False

    if nn is None:
        nn, _ = x0.shape

    actor = get_other_agents(n=nn, ii=affected)
    mdr = np.column_stack([mdr_a[actor].squeeze(), mdr_theta[actor].squeeze()])

    t0 = time.perf_counter()
    count_move = get_feasible_action_space(actor=actor, move=None, affected=affected,
                                           a_min=a_min, a_max=a_max, a_num=a_num,
                                           theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                                           trajs_hulls=trajs_hulls,
                                           trajs_boxes=trajs_boxes,
                                           obstacle=obstacle,
                                           x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                           t_action=t_action, n_timesteps=n_timesteps,
                                           l=l, plot_subspaces=plot_subspaces,
                                           plot_feasible_action_space=plot_feasible_action_space,
                                           return_feasibility=return_feasibility)
    t1 = time.perf_counter()
    count_mdr = get_feasible_action_space(actor=actor, move=mdr, affected=affected,
                                          a_min=a_min, a_max=a_max, a_num=a_num,
                                          theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                                          trajs_hulls=trajs_hulls,
                                          trajs_boxes=trajs_boxes,
                                          obstacle=obstacle,
                                          x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                          t_action=t_action, n_timesteps=n_timesteps,
                                          l=l, plot_subspaces=plot_subspaces,
                                          plot_feasible_action_space=plot_feasible_action_space,
                                          return_feasibility=return_feasibility)
    t2 = time.perf_counter()

    if return_feasibility:
        # count_move, feasibility_move, aa_thetaa_list_move = count_move
        # count_mdr, feasibility_mdr, aa_thetaa_list_mdr = count_mdr

        count_move, feasibility_move = count_move
        count_mdr, feasibility_mdr = count_mdr

        # feasibility = {
        #     'feasibility_move': feasibility_move,
        #     'feasibility_mdr': feasibility_mdr,
        #     'aa_thetaa_list_move': aa_thetaa_list_move,
        #     'aa_thetaa_list_mdr': aa_thetaa_list_mdr
        # }
        feasibility = {
            'feasibility_move': feasibility_move,
            'feasibility_mdr': feasibility_mdr
        }

    time_move = t1 - t0
    time_mdr = t2 - t1
    print('time_move=', time_move, ' ,time_mdr=', time_mdr)
    feal_actor_affected = np.clip(count_move / (count_mdr + EPS), -1, 1)

    if return_feasibility:
        return feal_actor_affected, feasibility

    return feal_actor_affected


@jit(forceobj=True)
def get_fear_actor_affected(actor=None, affected=None,
                            a_min=None, a_max=None, a_num=None,
                            theta_min=None, theta_max=None, theta_num=None,
                            mdr=None, n_timesteps=None, t_action=None, l=None,
                            trajs_boxes=None, trajs_hulls=None, obstacle=None,
                            v0x=None, v0y=None, x0=None, y0=None,
                            plot_feasible_action_space=False,
                            plot_subspaces=False,
                            return_feasibility=False):
    """
    Calculate the Feasible Action-Space Reduction values of one actor agent on another affected agent
    within a spatial interaction over a time-window.

    Parameters
    ----------
    actor : int
        The actor agent identifier.
    affected : int
        The affected agent identifier.
    a_min : float
        Lower limit of the magnitude of acceleration for the affected agent.
    a_max : float
        Upper limit of the magnitude of acceleration for the affected agent.
    a_num : int
        Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
    theta_min : float
        Lower limit of the angle for direction of acceleration for the affected agent.
    theta_max : float
        Upper limit of the angle for direction of acceleration for the affected agent.
    theta_num : int
        Number of intervals/subdivisions for direction of acceleration for the affected agent.
    mdr : list
        The move de rigueur of the actor agent (contains [magnitude, direction] of the constant acceleration move).
    n_timesteps : int
        Number of timesteps to consider within the time window.
    t_action : float
        Time window to be considered.
    l : float
        Length of an agent along one axis (assuming same length in x and y directions).
    trajs_boxes : list
        Trajectory boxes for agents for each time instant.
    trajs_hulls : list
        Trajectory hulls for agents for each time interval (convex hulls of trajectory boxes).
    obstacle : MultiPolygon (optional)
        Static obstacles as a MultiPolygon
    v0x : ndarray
        Starting velocity in the x-direction for agents.
    v0y : ndarray
        Starting velocity in the y-direction for agents.
    x0 : ndarray
        Starting x locations for agents.
    y0 : ndarray
        Starting y locations for agents.
    plot_feasible_action_space : bool, optional
        Whether to plot the feasible action space.
    plot_subspaces : bool, optional
        Whether to plot the subspaces
    return_feasibility : bool, optional
        Whether to return the dict with feasibility information of the subspaces.

    Returns
    -------
    fear_actor_to_affected : ndarray
        Feasible Action-Space Reduction values of the actor on the affected agent.
    """
    t0 = time.perf_counter()
    count_move = get_feasible_action_space(actor=actor, move=None, affected=affected,
                                           a_min=a_min, a_max=a_max, a_num=a_num,
                                           theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                                           trajs_hulls=trajs_hulls,
                                           trajs_boxes=trajs_boxes,
                                           obstacle=obstacle,
                                           x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                           t_action=t_action, n_timesteps=n_timesteps,
                                           l=l, plot_subspaces=plot_subspaces,
                                           plot_feasible_action_space=plot_feasible_action_space,
                                           return_feasibility=return_feasibility)
    t1 = time.perf_counter()
    count_mdr = get_feasible_action_space(actor=actor, move=mdr, affected=affected,
                                          a_min=a_min, a_max=a_max, a_num=a_num,
                                          theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                                          trajs_hulls=trajs_hulls,
                                          trajs_boxes=trajs_boxes,
                                          obstacle=obstacle,
                                          x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                          t_action=t_action, n_timesteps=n_timesteps,
                                          l=l, plot_subspaces=plot_subspaces,
                                          plot_feasible_action_space=plot_feasible_action_space,
                                          return_feasibility=return_feasibility)
    if return_feasibility:
        # count_move, feasibility_move, aa_thetaa_list_move = count_move
        # count_mdr, feasibility_mdr, aa_thetaa_list_mdr = count_mdr

        count_move, feasibility_move = count_move
        count_mdr, feasibility_mdr = count_mdr

        # feasibility = {
        #     'feasibility_move': feasibility_move,
        #     'feasibility_mdr': feasibility_mdr,
        #     'aa_thetaa_list_move': aa_thetaa_list_move,
        #     'aa_thetaa_list_mdr': aa_thetaa_list_mdr
        # }
        feasibility = {
            'feasibility_move': feasibility_move,
            'feasibility_mdr': feasibility_mdr
        }

    t2 = time.perf_counter()
    time_move = t1 - t0
    time_mdr = t2 - t1
    # print(f'{time_move=} , {time_mdr=}')
    print('time_move=', time_move, ' ,time_mdr=', time_mdr)
    fear_actor_affected = np.clip((count_mdr - count_move) / (count_mdr + EPS), -1, 1)

    if return_feasibility:
        return fear_actor_affected, feasibility

    return fear_actor_affected


def get_fears_actual_trajectory(timesteps_per_window=None, duration=None, timesteps=None, nn=None,
                                mdr_a=None, mdr_theta=None, a_min=None, a_max=None, a_num=None,
                                theta_min=None, theta_max=None, theta_num=None,
                                trajs_hulls=None, trajs_boxes=None, obstacle=None, x=None,
                                y=None, vx=None, vy=None, l=None,
                                show_plots=True,
                                plot_subspaces=False, plot_feasible_action_space=False):
    """
    Calculate FeAR over a sequence of actions based on a moving window over actual trajectory

    Parameters
    ----------
    timesteps_per_window
    duration
    timesteps
    nn
    mdr_a
    mdr_theta
    a_min
    a_max
    a_num
    theta_min
    theta_max
    theta_num
    trajs_hulls
    trajs_boxes
    obstacle
    x
    y
    vx
    vy
    l
    plot_subspaces
    plot_feasible_action_space

    Returns
    -------

    """

    if timesteps_per_window is None:
        print('No timesteps_per_window passed !')
        return False

    if duration is None:
        print('No duration passed !')
        return False

    if x is None:
        print('No x passed !')
        return False
    if y is None:
        print('No y passed !')
        return False
    if vx is None:
        print('No vx passed !')
        return False
    if vy is None:
        print('No vy passed !')
        return False

    t_action = duration / timesteps * timesteps_per_window

    num_windows = timesteps - timesteps_per_window + 1

    fears = np.zeros((num_windows, nn, nn))

    for ii in range(num_windows):
        window = range(ii, ii + timesteps_per_window)
        print(f'{window=}')

        xx0 = x[:, ii]
        yy0 = y[:, ii]
        vxx0 = vx[:, ii]
        vyy0 = vy[:, ii]

        fear = get_fear(nn=nn, mdr_a=mdr_a, mdr_theta=mdr_theta, a_min=a_min, a_max=a_max, a_num=a_num,
                        theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                        trajs_hulls=trajs_hulls[:, ii:ii + timesteps_per_window - 1],
                        trajs_boxes=trajs_boxes[:, ii:ii + timesteps_per_window], obstacle=obstacle, x0=xx0,
                        y0=yy0, v0x=vxx0, v0y=vyy0, t_action=t_action, n_timesteps=timesteps_per_window, l=l,
                        verbose=False,
                        plot_subspaces=plot_subspaces, plot_feasible_action_space=plot_feasible_action_space)

        if show_plots:
            ax = plotcf.plot_hulls_for_trajs(trajs_hulls, colour='lightgrey')
            ax = plotcf.plot_hulls_for_trajs(trajs_hulls[:, ii:ii + timesteps_per_window - 1],
                                             trajs_boxes=trajs_boxes[:, ii:ii + timesteps_per_window], ax=ax)
            plotcf.plot_velocities_for_agents(x=xx0, y=yy0, vx=vxx0, vy=vyy0, arrow_width=0.5, scale=10, markersize=1,
                                              ax=ax)

            plotcf.plot_fear_graph_on_trajs(fear=fear, obstacle=obstacle,
                                            trajs_hulls=trajs_hulls[:, ii:ii + timesteps_per_window], x0=xx0, y0=yy0,
                                            fear_threshold_percentile=0)

            plotcf.plt.show()

        fears[ii, :, :] = fear

    return fears


def get_fears_constant_acceleration(timesteps_per_window=None, duration=None, timesteps=None, nn=None,
                                    mdr_a=None, mdr_theta=None, a_min=None, a_max=None, a_num=None,
                                    theta_min=None, theta_max=None, theta_num=None,
                                    trajs_hulls=None, trajs_boxes=None, obstacle=None, x=None,
                                    y=None, vx=None, vy=None, l=None, a=None, theta=None,
                                    show_plots=True, plot_on_grid=False, compute_mdrs=False, return_mdrs=False,
                                    plot_subspaces=False, plot_feasible_action_space=False):
    """
    Calculate FeAR over a sequence of actions based on assuming constant acceleration within time windows

    Parameters
    ----------
    plot_on_grid
    timesteps_per_window
    duration
    timesteps
    nn
    mdr_a
    mdr_theta
    a_min
    a_max
    a_num
    theta_min
    theta_max
    theta_num
    trajs_hulls
    trajs_boxes
    obstacle
    x
    y
    vx
    vy
    l
    plot_subspaces
    plot_feasible_action_space

    Returns
    -------

    """

    if timesteps_per_window is None:
        print('No timesteps_per_window passed !')
        return False

    if duration is None:
        print('No duration passed !')
        return False

    if x is None:
        print('No x passed !')
        return False
    if y is None:
        print('No y passed !')
        return False
    if vx is None:
        print('No vx passed !')
        return False
    if vy is None:
        print('No vy passed !')
        return False
    if a is None:
        print('No a passed !')
        return False
    if theta is None:
        print('No theta passed !')
        return False

    if not compute_mdrs:
        if mdr_a is None:
            print('No mdr_a passed !')
            return False
        if mdr_theta is None:
            print('No mdr_theta passed !')
            return False

    time_per_window = duration / timesteps * timesteps_per_window

    # Constant acceleration windows till the last observed timestep
    # num_windows = timesteps - timesteps_per_window + 1

    # Extrapolating constant acceleration beyond the last observed timestep
    num_windows = timesteps - 1

    fears = np.zeros((num_windows, nn, nn))
    if return_mdrs:
        mdrs_a = np.zeros((nn, num_windows))
        mdrs_theta = np.zeros((nn, num_windows))

    for ii in range(num_windows):
        window = range(ii, ii + timesteps_per_window)
        print(f'{window=}')

        xx0 = x[:, ii].reshape(-1, 1)
        yy0 = y[:, ii].reshape(-1, 1)
        vxx0 = vx[:, ii].reshape(-1, 1)
        vyy0 = vy[:, ii].reshape(-1, 1)
        a0 = a[:, ii].reshape(-1, 1)
        theta0 = theta[:, ii].reshape(-1, 1)

        x_, y_, t_ = make_trajectories(x0=xx0, y0=yy0, v0x=vxx0, v0y=vyy0, a=a0, theta=theta0,
                                       t_action=time_per_window, n_timesteps=timesteps_per_window)

        trajs_boxes_ = get_boxes_for_trajs(x_, y_, lx=l, ly=l)

        trajs_hulls_ = get_trajs_hulls(trajs_boxes_)

        if compute_mdrs:
            mdr_a, mdr_theta = get_mdr(x=xx0, y=yy0, v0x=vxx0, v0y=vyy0,
                                       time_per_step=time_per_window / timesteps_per_window)
            if return_mdrs:
                mdrs_a[:, ii:] = mdr_a
                mdrs_theta[:, ii:] = mdr_theta

        fear = get_fear(nn=nn, mdr_a=mdr_a, mdr_theta=mdr_theta, a_min=a_min, a_max=a_max, a_num=a_num,
                        theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                        trajs_hulls=trajs_hulls_, trajs_boxes=trajs_boxes_, obstacle=obstacle,
                        x0=xx0, y0=yy0, v0x=vxx0, v0y=vyy0,
                        t_action=time_per_window, n_timesteps=timesteps_per_window,
                        verbose=False,
                        l=l, plot_subspaces=plot_subspaces, plot_feasible_action_space=plot_feasible_action_space)

        if show_plots:
            # if plot_on_grid and nn == 2:
            #     fig = plotcf.plt.figure(figsize=(16, 9), dpi=300)
            #     # Define a GridSpec layout: 2 rows, 2 columns
            #     gs = plotcf.gridspec.GridSpec(2, 2, width_ratios=[2, 1])
            #
            #     scene_ax = fig.add_subplot(gs[:, 0])
            #     fears12_ax = fig.add_subplot(gs[0, 1])
            #     fears21_ax = fig.add_subplot(gs[1, 1])
            # else:
            ax = plotcf.plot_hulls_for_trajs(trajs_hulls, trajs_boxes=trajs_boxes, colour='lightgrey')
            scene_xlim = ax.get_xlim()
            scene_ylim = ax.get_ylim()

            ax = plotcf.plot_hulls_for_trajs(trajs_hulls_, trajs_boxes=trajs_boxes_, ax=ax)

            plotcf.plot_velocities_for_agents(x=xx0, y=yy0, vx=vxx0, vy=vyy0,
                                              arrow_width=0.5, scale=10, markersize=1, ax=ax)

            fear_ax = plotcf.plot_fear_graph_on_trajs(fear=fear, obstacle=obstacle,
                                                      trajs_hulls=trajs_hulls_, x0=xx0, y0=yy0,
                                                      fear_threshold_percentile=0)
            fear_ax.set_xlim(scene_xlim)
            fear_ax.set_ylim(scene_ylim)

            plotcf.plt.show()

        fears[ii, :, :] = fear

    if return_mdrs:
        return fears, mdrs_a, mdrs_theta
    return fears


def get_fear_for_scenario(scenarios_json_file=None, scenario_name=None,
                          t_action=None, n_timesteps=None,
                          collision_free_trajs=False, compute_mdrs=False,
                          a_min=None, a_max=None, a_num=None,
                          theta_min=None, theta_max=None, theta_num=None,
                          return_feasibility=False,
                          return_mdr=False,
                          plot_feasible_action_space=False,
                          plot_trajs_and_boxes=False,
                          plot_scenario=True
                          ):
    """
        Calculate the Feasible Action-Space Reduction values for all agents within a given scenario over a time-window.

        Parameters
        ----------
        scenarios_json_file : str
            Path to the JSON file containing the scenario data.
        scenario_name : str
            The name of the specific scenario within the JSON file to be used.
        t_action : float
            Time window to be considered.
        n_timesteps : int
            Number of timesteps to consider within the time window.
        collision_free_trajs : bool, optional
            If True, only consider collision-free trajectories.
        compute_mdrs : bool, optional
            If True, compute the Move de Rigueurs (MDRs) for the agents.
            Optionally, if a dict is passed in with the mdr_a and mdr_theta arrays,
             these will be used to overide the mdrs
        a_min : float
            Lower limit of the magnitude of acceleration for the affected agent.
        a_max : float
            Upper limit of the magnitude of acceleration for the affected agent.
        a_num : int
            Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
        theta_min : float
            Lower limit of the angle for direction of acceleration for the affected agent.
        theta_max : float
            Upper limit of the angle for direction of acceleration for the affected agent.
        theta_num : int
            Number of intervals/subdivisions for direction of acceleration for the affected agent.
        return_feasibility : bool, optional
            Whether to return the dict with feasibility information of the subspaces.
        return_mdr : bool, optional
            Whether to return the dict with computed MdRs
        plot_feasible_action_space: bool, optional
            Whether to make the plots for the feasible action space
        plot_trajs_and_boxes: bool
            Whether to make the plots for the trajectories and traj_boxes
        plot_scenario : bool
            Whether to make the plots for the scenario

        Returns
        -------
        fear : ndarray
            Array containing Feasible Action-Space Reduction values for the specified scenario.

        Notes
        -----
        This function loads the scenario data from a JSON file and prepares the necessary parameters
        to call the `get_fear` function. It extracts the trajectory data, initial positions, velocities,
        and other relevant parameters from the specified scenario within the JSON file.
        """

    if scenarios_json_file is None:
        print('Missing json file with scenarios !')
        return False
    if scenario_name is None:
        print('No scenario_name passed in!')
        return False

    if t_action is None:
        print('No t_action passed!')
        return False
    if n_timesteps is None:
        print('No n_timesteps passed!')
        return False
    if a_min is None:
        print('No a_min passed!')
        return False
    if a_max is None:
        print('No a_max passed!')
        return False
    if a_num is None:
        print('No a_num passed!')
        return False
    if theta_min is None:
        print('No theta_min passed!')
        return False
    if theta_max is None:
        print('No theta_max passed!')
        return False
    if theta_num is None:
        print('No theta_num passed!')
        return False
    scenarios = Scenario.load_scenarios(scenarios_json_file)
    scenario = scenarios[scenario_name]

    # Populating variables with scenario data
    N = scenario.n_agents()
    l = scenario.l
    x0 = scenario.x0
    y0 = scenario.y0
    v0x = scenario.v0x
    v0y = scenario.v0y
    a = scenario.a
    theta = scenario.theta
    mdr_a = scenario.mdr_a
    mdr_theta = scenario.mdr_theta

    if scenario.obstacle != "None":
        obstacle = scenario.obstacle
    else:
        obstacle = None

    # -------------------------------------------------------------------------------------------------------------

    #  Creating the trajectories for the initial conditions

    x_, y_, t_ = make_trajectories(x0=x0, y0=y0, v0x=v0x, v0y=v0y, a=a, theta=theta,
                                   t_action=t_action, n_timesteps=n_timesteps)
    if plot_trajs_and_boxes:
        ax = plotcf.plot_trajectories(x=x_, y=y_, obstacle=obstacle)

    trajs_boxes = get_boxes_for_trajs(x_, y_, lx=l, ly=l)
    if plot_trajs_and_boxes:
        ax = plotcf.plot_boxes_for_trajs(trajs_boxes)
        plotcf.plot_obstacle(obstacle, ax=ax)

    trajs_hulls = get_trajs_hulls(trajs_boxes)
    if plot_trajs_and_boxes:
        ax = plotcf.plot_hulls_for_trajs(trajs_hulls, trajs_boxes=trajs_boxes)
        ax = plotcf.plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y, ax=ax, scale=t_action / n_timesteps)
        plotcf.plot_obstacle(obstacle, ax=ax)

    if plot_scenario:
        ax = plotcf.plot_hulls_for_trajs(trajs_hulls, trajs_boxes=trajs_boxes, colour='lightgrey')
        plotcf.plot_obstacle(obstacle, ax=ax)

        ax = plotcf.plot_velocities_for_agents(x=x0, y=y0, vx=v0x, vy=v0y, ax=ax, scale=t_action / n_timesteps)

    # -------------------------------------------------------------------------------------------------------------

    # Check for collisions and rectify collisions

    if collision_free_trajs:
        trajs_hulls, trajs_boxes = get_collision_free_trajs(trajs_hulls, trajs_boxes, obstacle=obstacle,
                                                            plot_polygons_one_by_one=False)

        ax = plotcf.plot_boxes_for_trajs(trajs_boxes, obstacle=obstacle)
        ax = plotcf.plot_hulls_for_trajs(trajs_hulls, obstacle=obstacle)

    # -------------------------------------------------------------------------------------------------------------

    # Compute MdRs

    # Use the values in the dict if compute_mdrs is a dict
    if isinstance(compute_mdrs, dict):
        mdr_a = compute_mdrs['mdr_a']
        mdr_theta = compute_mdrs['mdr_theta']

    # Else, compute the mdrs if compute_mdrs is true
    elif compute_mdrs:
        mdr_a, mdr_theta = get_mdr(x=x0, y=y0, v0x=v0x, v0y=v0y, time_per_step=t_action,
                                   buffer=2 * l)
        print(f'{mdr_a=}')
        print(f'{mdr_theta=}')

    # -------------------------------------------------------------------------------------------------------------

    # If feasibility is returned, make the plots after all the calculations.
    if return_feasibility:
        plot_feasible_action_space_to_pass = False
    else:
        plot_feasible_action_space_to_pass = plot_feasible_action_space

    # Compute FeAR

    fear = get_fear(nn=N, mdr_a=mdr_a, mdr_theta=mdr_theta,
                    a_min=a_min, a_max=a_max, a_num=a_num,
                    theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                    trajs_hulls=trajs_hulls,
                    trajs_boxes=trajs_boxes,
                    obstacle=obstacle,
                    x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                    t_action=t_action, n_timesteps=n_timesteps,
                    l=l, plot_subspaces=False,
                    plot_feasible_action_space=plot_feasible_action_space_to_pass,
                    return_feasibility=return_feasibility)

    if return_mdr:
        mdr = {'mdr_a': mdr_a,
               'mdr_theta': mdr_theta}

    if return_feasibility:
        fear, feasibility = fear
        if plot_feasible_action_space:
            plot_feasible_action_space_for_all(nn=N, feasibility=feasibility,
                                               mdr_a=mdr_a, mdr_theta=mdr_theta,
                                               a_min=a_min, a_max=a_max, a_num=a_num,
                                               theta_min=theta_min, theta_max=theta_max, theta_num=theta_num,
                                               trajs_boxes=trajs_boxes,
                                               trajs_hulls=trajs_hulls,
                                               obstacle=obstacle,
                                               x0=x0, y0=y0, v0x=v0x, v0y=v0y,
                                               t_action=t_action, n_timesteps=n_timesteps,
                                               show_legend=True, legend_inside=False, finer=True,
                                               l=l)

        if return_mdr:
            return fear, feasibility, mdr

        return fear, feasibility

    if return_mdr:
        return fear, mdr
    return fear


def get_fear_by_actor_for_scenario(actor=None, scenarios_json_file=None, scenario_name=None,
                                   t_action=None, n_timesteps=None,
                                   collision_free_trajs=False, compute_mdrs=False,
                                   a_min=None, a_max=None, a_num=None,
                                   theta_min=None, theta_max=None, theta_num=None
                                   ):
    """
        Calculate the Feasible Action-Space Reduction values for an actor agent within a given scenario
        over a time-window.

        Parameters
        ----------
        actor : int
            actor agent ID
        scenarios_json_file : str
            Path to the JSON file containing the scenario data.
        scenario_name : str
            The name of the specific scenario within the JSON file to be used.
        t_action : float
            Time window to be considered.
        n_timesteps : int
            Number of timesteps to consider within the time window.
        collision_free_trajs : bool, optional
            If True, only consider collision-free trajectories.
        compute_mdrs : bool, optional
            If True, compute the Move de Rigueurs (MDRs) for the agents.
            Optionally, if a dict is passed in with the mdr_a and mdr_theta arrays,
             these will be used to overide the mdrs
        a_min : float
            Lower limit of the magnitude of acceleration for the affected agent.
        a_max : float
            Upper limit of the magnitude of acceleration for the affected agent.
        a_num : int
            Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
        theta_min : float
            Lower limit of the angle for direction of acceleration for the affected agent.
        theta_max : float
            Upper limit of the angle for direction of acceleration for the affected agent.
        theta_num : int
            Number of intervals/subdivisions for direction of acceleration for the affected agent.

        Returns
        -------
        fear : ndarray
            Array containing Feasible Action-Space Reduction values for the actor in the specified scenario.

        Notes
        -----
        This function loads the scenario data from a JSON file and prepares the necessary parameters
        to call the `get_fear` function. It extracts the trajectory data, initial positions, velocities,
        and other relevant parameters from the specified scenario within the JSON file.
        """

    if actor is None:
        print('Actor agent ID not passed!')
        return False
    if scenarios_json_file is None:
        print('Missing json file with scenarios !')
        return False
    if scenario_name is None:
        print('No scenario_name passed in!')
        return False

    if t_action is None:
        print('No t_action passed!')
        return False
    if n_timesteps is None:
        print('No n_timesteps passed!')
        return False
    if a_min is None:
        print('No a_min passed!')
        return False
    if a_max is None:
        print('No a_max passed!')
        return False
    if a_num is None:
        print('No a_num passed!')
        return False
    if theta_min is None:
        print('No theta_min passed!')
        return False
    if theta_max is None:
        print('No theta_max passed!')
        return False
    if theta_num is None:
        print('No theta_num passed!')
        return False

    # -------------------------------------------------------------------------------------------------------------

    # Load Scenario

    scenarios = Scenario.load_scenarios(scenarios_json_file)
    scenario = scenarios[scenario_name]

    # Populating variables with scenario data
    N = scenario.n_agents()
    l = scenario.l
    x0 = scenario.x0
    y0 = scenario.y0
    v0x = scenario.v0x
    v0y = scenario.v0y
    a = scenario.a
    theta = scenario.theta
    mdr_a = scenario.mdr_a
    mdr_theta = scenario.mdr_theta

    if scenario.obstacle != "None":
        obstacle = scenario.obstacle
    else:
        obstacle = None

    # -------------------------------------------------------------------------------------------------------------

    # Compute MdRs

    # Use the values in the dict if compute_mdrs is a dict
    if isinstance(compute_mdrs, dict):
        mdr_a = compute_mdrs['mdr_a']
        mdr_theta = compute_mdrs['mdr_theta']

    # Else, compute the mdrs if compute_mdrs is true
    elif compute_mdrs:
        mdr_a, mdr_theta = get_mdr(x=x0, y=y0, v0x=v0x, v0y=v0y, time_per_step=t_action,
                                   buffer=2 * l)
        print(f'{mdr_a=}')
        print(f'{mdr_theta=}')

    actor_a = a[actor]
    actor_theta = theta[actor]

    fear_by_actor = swap_actor_action_and_get_fear_by_actor(actor=actor, actor_a=actor_a, actor_theta=actor_theta,
                                                            a=a, theta=theta,
                                                            a_max=a_max, a_min=a_min, a_num=a_num,
                                                            theta_max=theta_max, theta_min=theta_min,
                                                            theta_num=theta_num,
                                                            nn=N, collision_free_trajs=collision_free_trajs, l=l,
                                                            mdr_a=mdr_a, mdr_theta=mdr_theta, n_timesteps=n_timesteps,
                                                            obstacle=obstacle, t_action=t_action,
                                                            v0x=v0x, v0y=v0y, x0=x0, y0=y0)

    return fear_by_actor


def gridsearch_fear_by_actor_for_scenario(actor=None, scenarios_json_file=None, scenario_name=None,
                                          t_action=None, n_timesteps=None,
                                          collision_free_trajs=False, compute_mdrs=False,
                                          a_min=None, a_max=None, a_num=None,
                                          theta_min=None, theta_max=None, theta_num=None, check_collisions=False,
                                          ):
    """
        Calculate the Feasible Action-Space Reduction values for an actor agent within a given scenario
        over a time-window for different actions of the actor agent.

        Parameters
        ----------
        actor : int
            actor agent ID
        scenarios_json_file : str
            Path to the JSON file containing the scenario data.
        scenario_name : str
            The name of the specific scenario within the JSON file to be used.
        t_action : float
            Time window to be considered.
        n_timesteps : int
            Number of timesteps to consider within the time window.
        collision_free_trajs : bool, optional
            If True, only consider collision-free trajectories.
        compute_mdrs : bool, optional
            If True, compute the Move de Rigueurs (MDRs) for the agents.
            Optionally, if a dict is passed in with the mdr_a and mdr_theta arrays,
             these will be used to override the mdrs
        a_min : float
            Lower limit of the magnitude of acceleration for the affected agent.
        a_max : float
            Upper limit of the magnitude of acceleration for the affected agent.
        a_num : int
            Number of intervals/subdivisions for magnitude of acceleration for the affected agent.
        theta_min : float
            Lower limit of the angle for direction of acceleration for the affected agent.
        theta_max : float
            Upper limit of the angle for direction of acceleration for the affected agent.
        theta_num : int
            Number of intervals/subdivisions for direction of acceleration for the affected agent.
        check_collisions : bool
            If true, it checks whether actions by the actor agent leads to collisions of the actor.
            If there is a collision of the actor is identified, the corresponding FeAR values are set as np.nan

        Returns
        -------
        fears_by_actor : ndarray
            Array containing Feasible Action-Space Reduction values
            for different actions of the actor in the specified scenario.
        actor_as : ndarray
            Array containing the magnitudes of the acceleration of the actor
        actor_thetas : ndarray
            Array containing the directions of the acceleration of the actor
        collisions_4_actor: ndarray
            Array containing the boolean values for collisions of the actor

        Notes
        -----
        This function loads the scenario data from a JSON file and prepares the necessary parameters
        to call the `get_fear` function. It extracts the trajectory data, initial positions, velocities,
        and other relevant parameters from the specified scenario within the JSON file.
        """

    if actor is None:
        print('Actor agent ID not passed!')
        return False
    if scenarios_json_file is None:
        print('Missing json file with scenarios !')
        return False
    if scenario_name is None:
        print('No scenario_name passed in!')
        return False

    if t_action is None:
        print('No t_action passed!')
        return False
    if n_timesteps is None:
        print('No n_timesteps passed!')
        return False
    if a_min is None:
        print('No a_min passed!')
        return False
    if a_max is None:
        print('No a_max passed!')
        return False
    if a_num is None:
        print('No a_num passed!')
        return False
    if theta_min is None:
        print('No theta_min passed!')
        return False
    if theta_max is None:
        print('No theta_max passed!')
        return False
    if theta_num is None:
        print('No theta_num passed!')
        return False

    # -------------------------------------------------------------------------------------------------------------

    # Load Scenario

    scenarios = Scenario.load_scenarios(scenarios_json_file)
    scenario = scenarios[scenario_name]

    # Populating variables with scenario data
    nn = scenario.n_agents()
    l = scenario.l
    x0 = scenario.x0
    y0 = scenario.y0
    v0x = scenario.v0x
    v0y = scenario.v0y
    a = scenario.a
    theta = scenario.theta
    mdr_a = scenario.mdr_a
    mdr_theta = scenario.mdr_theta

    if scenario.obstacle != "None":
        obstacle = scenario.obstacle
    else:
        obstacle = None

    # -------------------------------------------------------------------------------------------------------------

    # Compute MdRs

    # Use the values in the dict if compute_mdrs is a dict
    if isinstance(compute_mdrs, dict):
        mdr_a = compute_mdrs['mdr_a']
        mdr_theta = compute_mdrs['mdr_theta']

    # Else, compute the mdrs if compute_mdrs is true
    elif compute_mdrs:
        mdr_a, mdr_theta = get_mdr(x=x0, y=y0, v0x=v0x, v0y=v0y, time_per_step=t_action,
                                   buffer=2 * l)
        print(f'{mdr_a=}')
        print(f'{mdr_theta=}')

    actor_as = np.linspace(a_min, a_max, a_num)
    actor_thetas = np.linspace(theta_min, theta_max, theta_num)

    fears_by_actor = np.zeros((len(actor_as), len(actor_thetas), nn))
    collisions_4_actor = np.zeros((len(actor_as), len(actor_thetas)), dtype='bool')

    for ii_a, actor_a in enumerate(tqdm(actor_as, colour="red", ncols=100)):
        for ii_theta, actor_theta in enumerate(actor_thetas):
            # Add collision check
            if check_collisions:
                actor_collision = check_collision_of_actor(actor=actor, actor_a=actor_a, actor_theta=actor_theta,
                                                           a=a, theta=theta,
                                                           v0x=v0x, v0y=v0y, x0=x0, y0=y0, l=l,
                                                           n_timesteps=n_timesteps, t_action=t_action,
                                                           obstacle=obstacle)
            else:
                actor_collision = False
            collisions_4_actor[ii_a, ii_theta] = actor_collision

            # if actor_collision:
            #     # # Assign FeAR = np.nan in case of collisions
            #     # array_of_nans = np.ones((1, nn))
            #     # array_of_nans[:] = np.nan
            #     # fears_by_actor[ii_a, ii_theta, :] = array_of_nans
            #
            #     # Assign FeAR = 1 in case of collisions
            #     fears_by_actor[ii_a, ii_theta, :] = np.ones((1, nn))
            # else:

            fears_by_actor[ii_a, ii_theta, :] = swap_actor_action_and_get_fear_by_actor(
                actor=actor,
                actor_a=actor_a, actor_theta=actor_theta,
                a=a, theta=theta,
                a_max=a_max, a_min=a_min, a_num=a_num,
                theta_max=theta_max, theta_min=theta_min, theta_num=theta_num,
                nn=nn, collision_free_trajs=collision_free_trajs, l=l,
                mdr_a=mdr_a, mdr_theta=mdr_theta,
                n_timesteps=n_timesteps,
                obstacle=obstacle, t_action=t_action,
                v0x=v0x, v0y=v0y, x0=x0, y0=y0)

    if check_collisions:
        return fears_by_actor, actor_as, actor_thetas, collisions_4_actor

    return fears_by_actor, actor_as, actor_thetas


def check_collision_of_actor(actor, actor_a, actor_theta,
                             a, theta,
                             v0x, v0y, x0, y0, l,
                             n_timesteps, t_action,
                             obstacle
                             ):
    a_ = copy.copy(a)
    theta_ = copy.copy(theta)

    # Swapping the action of the actor
    a_[actor] = actor_a
    theta_[actor] = actor_theta
    # -------------------------------------------------------------------------------------------------------------

    #  Creating the trajectories for the initial conditions
    x_, y_, t_ = make_trajectories(x0=x0, y0=y0, v0x=v0x, v0y=v0y, a=a_, theta=theta_,
                                   t_action=t_action, n_timesteps=n_timesteps)
    trajs_boxes = get_boxes_for_trajs(x_, y_, lx=l, ly=l)
    trajs_hulls = get_trajs_hulls(trajs_boxes)

    # -------------------------------------------------------------------------------------------------------------

    # The following code simultaneously checks whether trajs_hulls is an ndarray
    # and also provides the dimensions.
    try:
        n_traj, n_hulls_per_traj = trajs_hulls.shape
    except AttributeError:
        trajs_hulls = np.array(trajs_hulls, dtype=object)
        n_traj, n_hulls_per_traj = trajs_hulls.shape

    # -----------------------------------------------------------

    trajs_hulls_i = trajs_hulls[[actor]]

    # Check collision with static obstacles
    if obstacle is not None:
        hull_over_time_i = get_hull_of_polygons(trajs_hulls_i)
        collision = obstacle.intersects(hull_over_time_i)
        if collision:
            return collision

    # Get list of agents other than the affected
    not_actor = get_other_agents(n=n_traj, ii=actor)

    trajs_hulls_not_i = trajs_hulls[not_actor]
    trajs_boxes_not_i = trajs_boxes[not_actor]

    # Resolve collisions for other agents
    trajs_hulls_not_i, trajs_boxes_not_i = get_collision_free_trajs(trajs_hulls_not_i,
                                                                    trajs_boxes_not_i,
                                                                    obstacle=obstacle,
                                                                    plot_polygons_one_by_one=False)

    # Add agent j as the ego agent with index 0
    traj_hulls_i_not_i = np.concatenate((trajs_hulls_i, trajs_hulls_not_i))

    # The affected agent j would be under index 0 in this array.
    collision = check_ego_collisions_as_time_rolls(n_hulls_per_traj=n_hulls_per_traj,
                                                   traj_hulls_j_not_j=traj_hulls_i_not_i)

    return collision


def swap_actor_action_and_get_fear_by_actor(nn,
                                            a,
                                            a_max, a_min, a_num, actor, actor_a, actor_theta,
                                            collision_free_trajs, l, mdr_a, mdr_theta, n_timesteps, obstacle, t_action,
                                            theta,
                                            theta_max, theta_min, theta_num, v0x, v0y, x0, y0):
    a_ = copy.copy(a)
    theta_ = copy.copy(theta)

    # Swapping the action of the actor
    a_[actor] = actor_a
    theta_[actor] = actor_theta
    # -------------------------------------------------------------------------------------------------------------
    #  Creating the trajectories for the initial conditions
    x_, y_, t_ = make_trajectories(x0=x0, y0=y0, v0x=v0x, v0y=v0y, a=a_, theta=theta_,
                                   t_action=t_action, n_timesteps=n_timesteps)
    trajs_boxes = get_boxes_for_trajs(x_, y_, lx=l, ly=l)
    trajs_hulls = get_trajs_hulls(trajs_boxes)
    # -------------------------------------------------------------------------------------------------------------
    # Check for collisions and rectify collisions
    if collision_free_trajs:
        trajs_hulls, trajs_boxes = get_collision_free_trajs(trajs_hulls, trajs_boxes, obstacle=obstacle,
                                                            plot_polygons_one_by_one=False)
    # -------------------------------------------------------------------------------------------------------------
    # Compute FeAR
    fear_by_actor = get_fear_for_actor(actor=actor, a_max=a_max, a_min=a_min, a_num=a_num,
                                       theta_max=theta_max, theta_min=theta_min, theta_num=theta_num,
                                       mdr_a=mdr_a, mdr_theta=mdr_theta,
                                       n_timesteps=n_timesteps, t_action=t_action, nn=nn, l=l,
                                       trajs_boxes=trajs_boxes, trajs_hulls=trajs_hulls,
                                       obstacle=obstacle,
                                       v0x=v0x, v0y=v0y, x0=x0, y0=y0)
    return fear_by_actor


def hyper_fear(hyperparameter_name, values, base_params, scenarios_json_file, scenario_name,
               return_mdr=True,
               return_feasibility=False):
    """
    Vary a specified hyperparameter over a range of values and compute fear values.

    Parameters
    ----------
    hyperparameter_name : str
        The name of the hyperparameter to vary.
    values : list
        A list of values to assign to the hyperparameter.
    base_params : dict
        A dictionary of the other parameters to pass to get_fear_for_scenario.
    scenarios_json_file : str
        Path to the JSON file containing the scenario data.
    scenario_name : str
        The name of the specific scenario within the JSON file to be used.
    return_feasibility : bool, optional
        Whether to return the dict with feasibility information of the subspaces.
    return_mdr: bool, optional
        Whether to return the dict with MdRs used



    Returns
    -------
    results : dict
        Dictionary containing the range of the hyperparameter, the corresponding fear values,
        the base parameters, and the scenario information.
    """
    fears = []
    feasibilities = []

    if return_mdr:
        mdrs = []
    else:
        mdrs = None

    for value in values:
        params = base_params.copy()
        params[hyperparameter_name] = value
        fear = get_fear_for_scenario(
            scenarios_json_file=scenarios_json_file,
            scenario_name=scenario_name,
            return_feasibility=return_feasibility,
            return_mdr=return_mdr,
            **params
        )

        if return_mdr and return_feasibility:
            fear, feasibility, mdr = fear
        elif return_mdr:
            fear, mdr = fear
        elif return_feasibility:
            fear, feasibility = fear

        if return_mdr:
            mdrs.append(mdr)
        if return_feasibility:
            feasibilities.append(feasibility)
        fears.append(fear)

    fears = np.array(fears)
    results = {
        'scenario_name': scenario_name,
        'scenarios_json_file': scenarios_json_file,
        'hyperparameter': hyperparameter_name,
        'values': values,
        'fear_values': fears,
        'feasibilities': feasibilities,
        'mdrs': mdrs,
        'base_params': base_params
    }

    return results


def save_results(filename, results, folder='Results'):
    """
    Save the results dictionary to a file using pickle.

    Parameters
    ----------
    filename : str
        The name of the file where the results will be saved.
    results : dict
        Dictionary containing the hyperparameter, its values, the computed fear values,
        base parameters, and scenario information.
    folder : str, optional
        The folder where the file will be saved. If None, the file is saved in the current directory.
    """
    if folder:
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, filename)
    else:
        file_path = filename

    with open(file_path, 'wb') as file:
        pickle.dump(results, file)


def name_and_save_results(results, folder='FeAR_Results'):
    t = datetime.datetime.now()
    time_string = t.strftime('%Y-%m-%d_%H-%M-%S')

    filename = results['scenario_name'] + '-' + results['hyperparameter'] \
               + '-' + time_string + '.pkl'

    save_results(filename, results, folder=folder)
    print(f'Saved results to {filename} in folder {folder}.')

    return filename


def distance_cost(params, x0, y0, v0x, v0y, t, a0, theta0, lane_polynomial_str, w_a=0.1):
    """
    The cost function with the distance to the lane.

    Parameters
    ----------
    params : tuple (a, theta)
    x0 : initial x location
    y0 : initial y location
    v0x : initial velocity in x direction
    v0y : initial velocity in y direction
    t : time_window
    lane_polynomial_str : string expression of the lane equation
    w_a : weight for the acceleration term.

    Returns
    -------
    cost: float
    """

    a, theta = params
    ax = a * np.sin(theta)
    ay = a * np.cos(theta)

    # Final positions after time t
    px = x0 + v0x * t + 0.5 * ax * t ** 2
    py = y0 + v0y * t + 0.5 * ay * t ** 2

    # Minimize the distance to the curve
    def distance_to_curve(point):
        x, y = point
        return np.sqrt((x - px) ** 2 + (y - py) ** 2)

    # Constraint to ensure point is on the curve
    def constraint(point):
        x, y = point
        return eval(lane_polynomial_str, {"x": x, "y": y, "np": np})

    # Find the closest point on the curve
    result = minimize(distance_to_curve, x0=[px, py], constraints={'type': 'eq', 'fun': constraint})
    x_min, y_min = result.x
    min_distance = distance_to_curve((x_min, y_min))

    return min_distance + w_a * ((a - a0) ** 2 + (theta - theta0) ** 2)


def accelerate_2_lane_eventually(x0, y0, v0x, v0y, t, w_a=0.1,
                                 lane_polynomial_str=None,
                                 bounds=[(0, 5), (-np.pi, np.pi)],
                                 initial_guess=[0, 0], plots=True):
    """
    Function to find the optimal acceleration magnitude and direction to go towards a lane.

    Parameters
    ----------
    x0 : initial x location
    y0 : initial y location
    v0x : initial velocity in x direction
    v0y : initial velocity in y direction
    t : time window
    lane_polynomial_str : string expression for the equation of the lane.
    bounds : [(a_min, a_max), (theta_min, theta_max)]
    initial_guess : [a_0, theta_0]
    w_a : weight of the acceleration term in the cost function
    plots : bool
        whether to make plots

    Returns
    -------
    optimal_a, optimal_theta
    """
    if lane_polynomial_str is None:
        print('No lane function passed! Returning the initial guess!')
        return initial_guess

    # Optimize a and theta to minimize the distance
    # result = minimize(distance_cost, initial_guess, args=(x0, y0, v0x, v0y, t, lane_polynomial_str, w_a),
    #                   bounds=bounds)
    a0 = initial_guess[0]
    theta0 = initial_guess[1]

    result = differential_evolution(distance_cost, args=(x0, y0, v0x, v0y, t, a0, theta0, lane_polynomial_str, w_a),
                                    bounds=bounds)

    # Get the optimal a and theta
    optimal_a, optimal_theta = result.x

    if plots:
        # Calculate the final position with optimal a and theta
        ax_opt = optimal_a * np.cos(optimal_theta)
        ay_opt = optimal_a * np.sin(optimal_theta)
        px_opt = x0 + v0x * t + 0.5 * ax_opt * t ** 2
        py_opt = y0 + v0y * t + 0.5 * ay_opt * t ** 2

        def implicit_polynomial(x, y):
            return eval(lane_polynomial_str, {"x": x, "y": y, "np": np})

        # Plot the polynomial curve
        x_vals = np.linspace(-15, 15, 400)
        y_vals = np.linspace(-15, 15, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = implicit_polynomial(X, Y)

        plotcf.plt.figure(figsize=(10, 8))
        plotcf.plt.contour(X, Y, Z, levels=[0], colors='blue')

        # Plot the initial location
        plotcf.plt.plot(x0, y0, 'go', label='Initial Location (x0, y0)')

        # Plot the final optimal location
        plotcf.plt.plot(px_opt, py_opt, 'ro', label='Final Optimal Location')

        # Plot the path followed by the point mass
        t_vals = np.linspace(0, t, 100)
        x_path = x0 + v0x * t_vals + 0.5 * ax_opt * t_vals ** 2
        y_path = y0 + v0y * t_vals + 0.5 * ay_opt * t_vals ** 2
        plotcf.plt.plot(x_path, y_path, 'k--', label='Path Followed')

        plotcf.plt.xlabel('x')
        plotcf.plt.ylabel('y')
        plotcf.plt.title('Polynomial Curve and Point Mass Path')
        plotcf.plt.legend()
        plotcf.plt.grid()
        plotcf.plt.axis('equal')
        plotcf.plt.show()

        print(f"Optimal a: {optimal_a}, Optimal theta: {optimal_theta}")

    return optimal_a, optimal_theta


def accelerate_2_lane(x0, y0, v0x, v0y, t,
                      lane_polynomial_str=None,
                      plots=False):
    """
    Function to find the acceleration magnitude and direction to go towards the closest point on a lane.

    Parameters
    ----------
    x0 : initial x location
    y0 : initial y location
    v0x : initial velocity in x direction
    v0y : initial velocity in y direction
    t : time window
    lane_polynomial_str : string expression for the equation of the lane.
    plots : bool
        whether to make plots

    Returns
    -------
    a_2_lane, theta_2_lane
    """

    if lane_polynomial_str is None:
        print('No lane function passed! Returning the initial guess!')
        return False

    # Find the closest point on lane to the initial location
    x_min, y_min = closest_point_on_lane(px=x0, py=y0, lane_polynomial_str=lane_polynomial_str)

    # Acceleration direction assuming zero initial velocity
    ax = 2 * (x_min - x0) / t ** 2
    ay = 2 * (y_min - y0) / t ** 2
    a_2_lane, theta_2_lane = get_a_and_theta(ax=ax, ay=ay)

    if plots:
        # Calculate the final position with optimal a and theta
        ax_opt = a_2_lane * np.cos(theta_2_lane)
        ay_opt = a_2_lane * np.sin(theta_2_lane)
        px_opt = x0 + 0.5 * ax_opt * t ** 2
        py_opt = y0 + 0.5 * ay_opt * t ** 2

        def implicit_polynomial(x, y):
            return eval(lane_polynomial_str, {"x": x, "y": y, "np": np})

        # Plot the polynomial curve
        x_vals = np.linspace(-15, 15, 400)
        y_vals = np.linspace(-15, 15, 400)
        X, Y = np.meshgrid(x_vals, y_vals)
        Z = implicit_polynomial(X, Y)

        plotcf.plt.figure(figsize=(10, 8))
        plotcf.plt.contour(X, Y, Z, levels=[0], colors='blue')

        # Plot the initial location
        plotcf.plt.plot(x0, y0, 'go', label='Initial Location (x0, y0)')

        # Plot the final optimal location
        plotcf.plt.plot(px_opt, py_opt, 'ro', label='Final Optimal Location')
        plotcf.plt.plot(x_min, y_min, 'ro', label='Closest point on Curve')

        # Plot the path followed by the point mass
        t_vals = np.linspace(0, t, 100)
        x_path = x0 + 0.5 * ax_opt * t_vals ** 2
        y_path = y0 + 0.5 * ay_opt * t_vals ** 2
        plotcf.plt.plot(x_path, y_path, 'k--', label='Path Followed')

        plotcf.plt.xlabel('x')
        plotcf.plt.ylabel('y')
        plotcf.plt.title('Polynomial Curve and Point Mass Path')
        plotcf.plt.legend()
        plotcf.plt.grid()
        plotcf.plt.axis('equal')
        plotcf.plt.show()

        print(f"a_2_lane: {a_2_lane}, theta_2_lane: {theta_2_lane}")

    return a_2_lane, theta_2_lane


def closest_point_on_lane(px, py, lane_polynomial_str=None):
    # Minimize the distance to the curve
    def distance_to_curve(point):
        x, y = point
        return np.sqrt((x - px) ** 2 + (y - py) ** 2)

    # Constraint to ensure point is on the curve
    def constraint(point):
        x, y = point
        return eval(lane_polynomial_str, {"x": x, "y": y, "np": np})

    # Find the closest point on the curve
    result = minimize(distance_to_curve, x0=[px, py], constraints={'type': 'eq', 'fun': constraint})
    x_min, y_min = result.x

    return x_min, y_min


def minimise_fear(actor_as, actor_thetas, fear_by_actor, collisions_4_actor):
    if collisions_4_actor is None:
        collisions_4_actor = np.zeros_like(fear_by_actor, dtype=bool)

    # Find the minimum value in fear_by_actor and its indices
    min_value = np.min(fear_by_actor, where=~collisions_4_actor, initial=1)
    # Find the indices of all occurrences of the minimum value
    min_indices = np.where((fear_by_actor == min_value) & ~collisions_4_actor)
    # Corresponding actor_a and actor_theta
    min_actors_a = [actor_as[i] for i in min_indices[0]]
    min_actors_theta = [actor_thetas[j] for j in min_indices[1]]
    return min_actors_a, min_actors_theta, min_value


def find_actions_with_collisions(actor_as, actor_thetas, collisions_4_actor):
    # Find the indices of all occurrences of True
    collision_indices = np.where(collisions_4_actor)
    # Corresponding actor_a and actor_theta
    collision_a = [actor_as[i] for i in collision_indices[0]]
    collision_theta = [actor_thetas[j] for j in collision_indices[1]]
    return collision_a, collision_theta


def check_array_consecutive_ints(arr=None):
    if arr is None:
        print('No array passed. Exiting')
        return False

    # Convert input to a NumPy array
    a = np.array(arr)

    # Check if all elements are integers
    if not np.issubdtype(a.dtype, np.integer):
        return False

    # Check for uniqueness and consecutive property
    return len(np.unique(a)) == len(a) and np.max(a) - np.min(a) + 1 == len(a)


class Scenario:
    """
       A class to store information about a scenario for continuous FeAR simulations.

       Attributes:
       -----------
       x0 : numpy.ndarray
           Initial x-coordinates of the agents. Shape (nn, 1) where nn is the number of agents.
       y0 : numpy.ndarray
           Initial y-coordinates of the agents. Shape (nn, 1) where nn is the number of agents.
       v0x : numpy.ndarray
           Initial x-velocities of the agents. Shape (nn, 1) where nn is the number of agents.
       v0y : numpy.ndarray
           Initial y-velocities of the agents. Shape (nn, 1) where nn is the number of agents.
       a : numpy.ndarray
           Magnitude of acceleration of the agents. Shape (nn, 1) where nn is the number of agents.
       theta : numpy.ndarray
           Direction of the acceleration of the agents. Shape (nn, 1) where nn is the number of agents.
       mdr_a : numpy.ndarray
           Magnitude of the acceleration of the Move de Rigueur of the agents.
           Shape (nn, 1) where nn is the number of agents.
       mdr_theta : numpy.ndarray
           Direction of the acceleration of the Move de Rigueur of the agents.
           Shape (nn, 1) where nn is the number of agents.
        obstacle : shapely polygon
           Obstacle contains all the information about static obstacles in the scene.

       Methods:
       --------
       add_agent(x0, y0, v0x, v0y, a, theta, mdr_a, mdr_theta)
           Add information about a new agent to the system.
       print_agents()
           Print out the current information stored for each agent.
       print_data()
           Print out the current information stored in each variable.
           This is useful for reviewing the structure of the data
       n_agents()
           Get the number of agents
       """

    def __init__(self):
        """
        Initializes the AgentSystem with empty arrays for agent information.
        """
        self.x0 = np.array([])
        self.y0 = np.array([])
        self.v0x = np.array([])
        self.v0y = np.array([])

        self.a = np.array([])
        self.theta = np.array([])
        self.mdr_a = np.array([])
        self.mdr_theta = np.array([])

        self.obstacle = []
        self.l = 10

    def n_agents(self):
        """
        Function that returns the number of agents
        """
        return len(self.x0)

    def add_agent(self, x0, y0, v0x, v0y,
                  a=0, theta=0, mdr_a=0, mdr_theta=0):
        """
        Add information about a new agent to the system.

        Parameters:
        -----------
        x0 : float
            Initial x-coordinate of the new agent.
        y0 : float
            Initial y-coordinate of the new agent.
        v0x : float
            Initial x-velocity of the new agent.
        v0y : float
            Initial y-velocity of the new agent.
        a : float
            Magnitude of acceleration of the new agent.
        theta : float
            Direction of the acceleration of the new agent in radians.
        mdr_a : float
            Magnitude of the acceleration of the Move de Rigueur of the new agent.
        mdr_theta : float
            Direction of the acceleration of the Move de Rigueur of the new agent.
        """

        self.x0 = np.append(self.x0, x0).reshape(-1, 1)
        self.y0 = np.append(self.y0, y0).reshape(-1, 1)
        self.v0x = np.append(self.v0x, v0x).reshape(-1, 1)
        self.v0y = np.append(self.v0y, v0y).reshape(-1, 1)
        self.a = np.append(self.a, a).reshape(-1, 1)
        self.theta = np.append(self.theta, theta).reshape(-1, 1)
        self.mdr_a = np.append(self.mdr_a, mdr_a).reshape(-1, 1)
        self.mdr_theta = np.append(self.mdr_theta, mdr_theta).reshape(-1, 1)

    def remove_agents(self, indices):
        """
        Remove agents from the scenario based on their indices.

        Parameters:
        -----------
        indices : list
            A list of indices of agents to be removed.
        """
        self.x0 = np.delete(self.x0, indices, axis=0)
        self.y0 = np.delete(self.y0, indices, axis=0)
        self.v0x = np.delete(self.v0x, indices, axis=0)
        self.v0y = np.delete(self.v0y, indices, axis=0)
        self.a = np.delete(self.a, indices, axis=0)
        self.theta = np.delete(self.theta, indices, axis=0)
        self.mdr_a = np.delete(self.mdr_a, indices, axis=0)
        self.mdr_theta = np.delete(self.mdr_theta, indices, axis=0)

    def print_data(self):
        """
        Prints the data as it is stored in variables.
        """
        x0 = self.x0
        y0 = self.y0
        v0x = self.v0x
        v0y = self.v0y
        a = self.a
        theta = self.theta
        mdr_a = self.mdr_a
        mdr_theta = self.mdr_theta

        print(f'{x0=}')
        print(f'{y0=}')
        print(f'{v0x=}')
        print(f'{v0y=}')
        print(f'{a=}')
        print(f'{theta=}')
        print(f'{mdr_a=}')
        print(f'{mdr_theta=}')

    def print_agents(self):
        """
        Prints the data associated with each agent.
        """
        print("\n ----------------------------------------------------------- \n")

        print("Agent Locations (x0, y0):")
        print("-------------------------")
        for ii in range(len(self.x0)):
            print(f"Agent {ii + 1}: ({self.x0[ii]}, {self.y0[ii]})")
        print("\n ----------------------------------------------------------- \n")

        print("Agent Velocities (v0x, v0y):")
        print("----------------------------")
        for ii in range(len(self.v0x)):
            print(f"Agent {ii + 1}: ({self.v0x[ii]}, {self.v0y[ii]})")
        print("\n ----------------------------------------------------------- \n")

        print("Agent actions (a, theta):")
        print("-------------------------")
        for ii in range(len(self.a)):
            print(f"Agent {ii + 1}: ({self.a[ii]}, {self.theta[ii]})")
        print("\n ----------------------------------------------------------- \n")

        print("Agent MdRs (mdr_a, mdr_theta):")
        print("------------------------------")
        for ii in range(len(self.a)):
            print(f"Agent {ii + 1}: ({self.mdr_a[ii]}, {self.mdr_theta[ii]})")
        print("\n ----------------------------------------------------------- \n")

    def to_dict(self):
        """
        Convert the scenario to a dictionary.

        Returns:
        --------
        dict
            A dictionary containing the scenario information.
        """
        return {
            'x0': self.x0.tolist(),
            'y0': self.y0.tolist(),
            'v0x': self.v0x.tolist(),
            'v0y': self.v0y.tolist(),
            'a': self.a.tolist(),
            'theta': self.theta.tolist(),
            'mdr_a': self.mdr_a.tolist(),
            'mdr_theta': self.mdr_theta.tolist(),
            'l': self.l,
            'obstacle': f'{self.obstacle}'
        }

    @classmethod
    def from_dict(cls, data):
        """
        Create a Scenario instance from a dictionary.

        Parameters:
        -----------
        data : dict
            A dictionary containing the scenario information.

        Returns:
        --------
        Scenario
            A Scenario instance initialized with the data from the dictionary.
        """
        scenario = cls()
        scenario.x0 = np.array(data['x0'])
        scenario.y0 = np.array(data['y0'])
        scenario.v0x = np.array(data['v0x'])
        scenario.v0y = np.array(data['v0y'])
        scenario.a = np.array(data['a'])
        scenario.theta = np.array(data['theta'])
        scenario.mdr_a = np.array(data['mdr_a'])
        scenario.mdr_theta = np.array(data['mdr_theta'])
        scenario.l = data['l']

        # print(f"{data['obstacle']=}")
        # print(f"{data['obstacle'] == [] =}")

        if data['obstacle'] == "None":
            scenario.obstacle = None
        else:
            scenario.obstacle = shapely.from_wkt(data['obstacle'])

        # print(f'{scenario.obstacle=}')
        # scenario.print_agents()
        return scenario

    @staticmethod
    def save_scenarios(scenarios, filename):
        """
        Save scenarios to a JSON file.

        Parameters:
        -----------
        scenarios : dict
            A dictionary where keys are scenario names and values are scenario objects.
        filename : str
            The filename to save the JSON data.
        """
        data = {name: scenario.to_dict() for name, scenario in scenarios.items()}

        # pretty_print_json = pprint.pformat(data, width=200).replace("'", '"')
        pretty_print_json = pprint.pformat(data, width=40, newline='\\').replace("'", '"')

        with open(filename, 'w') as f:
            f.write(pretty_print_json)

    @staticmethod
    def load_scenarios(filename):
        """
        Load scenarios from a JSON file.

        Parameters:
        -----------
        filename : str
            The filename of the JSON data.

        Returns:
        --------
        dict
            A dictionary where keys are scenario names and values are scenario objects.
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        scenarios = {name: Scenario.from_dict(scenario_data) for name, scenario_data in data.items()}
        return scenarios

    @staticmethod
    def append_scenario(filename, scenario_name, scenario):
        """
        Append a new scenario to an existing JSON file.

        Parameters:
        -----------
        filename : str
            The filename of the existing JSON file.
        scenario_name : str
            The name of the new scenario.
        scenario : Scenario
            The scenario object to append.
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {}

        data[scenario_name] = scenario.to_dict()

        pretty_print_json = NoStringWrappingPrettyPrinter().pformat(data).replace("'", '"')

        with open(filename, 'w') as f:
            f.write(pretty_print_json)
            # json.dumps(data, f, indent=1, separators=(',', ':'))


class NoStringWrappingPrettyPrinter(pprint.PrettyPrinter):
    """
    Changing the subclass behaviour of pprint so that strings are not split
    From Martijn Pieters
    https://stackoverflow.com/questions/31485402/can-i-make-pprint-in-python3-not-split-strings-like-in-python2
    """

    def _format(self, object, *args):
        if isinstance(object, str):
            width = self._width
            self._width = sys.maxsize
            try:
                super()._format(object, *args)
            finally:
                self._width = width
        else:
            super()._format(object, *args)
