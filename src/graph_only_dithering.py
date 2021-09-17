import os
import random

import time
import cv2
import scipy.spatial
import scipy.ndimage
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import svgwrite.extensions
from numba import njit, jit
from PIL import Image


def get_unique_list_from_points(pairs):
    pairs = np.asarray(pairs).flatten().reshape(-1, 2)
    return np.unique(pairs, axis=0)


def get_nodes_and_positions(points):
    features = []
    pos = dict()
    node_count = points.shape[0]
    for i in range(node_count):
        features.append({'pos': points[i, :]})
        pos[i] = points[i, :2]
    nodes = zip(range(node_count), features)
    return nodes, pos


def get_graph_and_node_pos(points, triangles):
    pairs = [[[a, b], [b, c], [c, a]] for a, b, c in triangles]

    unique_list = get_unique_list_from_points(pairs)
    nodes, pos = get_nodes_and_positions(points)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(unique_list, fill_chance=0)
    return G, pos, unique_list


def is_in_hull(point_list, hull_vertices):
    """
    Determine if the list of points P lies inside the hull
    :return: list
    List of boolean where true means that the point is inside the convex hull
    """
    hull = scipy.spatial.ConvexHull(hull_vertices)
    A = hull.equations[:, 0:-1]
    b = hull.equations[:, -1:]
    # vertices_rolled = np.roll(hull_vertices, 1)
    # cur_det = 1/(hull_vertices[:, 0] * vertices_rolled[:, 1] - hull_vertices[:, 1] * vertices_rolled[:, 0])
    # x_coefs = vertices_rolled[:, 1] - hull_vertices[:, 1]
    # y_coefs = hull_vertices[:, 0] - vertices_rolled[:, 0]
    # A = np.stack((x_coefs, y_coefs), axis=1)
    # b = np.expand_dims(cur_det * np.ones_like(x_coefs), axis=1)
    # # assert np.all(A == A2)
    # # assert np.all(b == b2)
    point_values = A @ np.transpose(point_list)
    out = np.all(point_values <= -b, axis=0)
    return out


def get_grid_points_in_hull(hull, max_dimensions):
    # low_bound = np.min(hull, axis=0)
    # low_bound = np.maximum(low_bound, np.zeros_like(low_bound))
    # high_bound = np.max(hull, axis=0)
    # high_bound = np.minimum(high_bound, np.asarray(max_dimensions) - 1)
    low_bound = (np.min(hull[:, 0]), np.min(hull[:, 1]))
    high_bound = (np.max(hull[:, 0]), np.max(hull[:, 1]))

    x, y = np.meshgrid(np.arange(np.floor(low_bound[0]), np.ceil(high_bound[0])),
                       np.arange(np.floor(low_bound[1]), np.ceil(high_bound[1])))
    x, y = x.flatten(), y.flatten()
    grid_points = np.stack([x, y], axis=1)
    is_grid_points_in_hull = is_in_hull(grid_points, hull)
    grid_points_in_hull = np.asarray(grid_points)[is_grid_points_in_hull]
    return grid_points_in_hull


def get_perimeter(points):
    side_lengths = np.empty(len(points))
    for i in range(len(points)):
        side_lengths[i] = np.linalg.norm(points[i] - points[i + 1 % len(points)])
    return sum(side_lengths), side_lengths


def shoelace(x_y):
    x = x_y[:, 0]
    y = x_y[:, 1]
    S1 = np.sum(x * np.roll(y, -1))
    S2 = np.sum(y * np.roll(x, -1))
    return .5 * np.absolute(S1 - S2)


def get_order(n, vec=np.asarray([0, 1])):
    if len(vec) > n:
        return vec / len(vec)
    else:
        if np.random.rand() < .5:
            return get_order(n, np.concatenate([vec * 2, vec * 2 + 1]))
        else:
            return get_order(n, np.concatenate([vec * 2 + 1, vec * 2]))


def start():
    """
        TODO: Pathwise ordered dithering
        TODO: Floyd Steinberg Error passing?
        TODO: Parallel iterating over the triangles
    """
    # -----------------------------------------------------------------------------------------------------------------
    #                                           Parameters
    # -----------------------------------------------------------------------------------------------------------------
    downsampling_factor = 3
    edge_weighted_point_count = 5000
    random_node_count = 50000
    highlight_point_count = 15000
    darkness = .45
    # source_file = 'starry_night.jpg'
    source_filepath = 'Russell_Head.jpg'
    source_highlight_filepath = 'Russell_Head_Highlights.jpg'

    # -----------------------------------------------------------------------------------------------------------------
    #                                           Load Data
    # -------------------------------  ----------------------------------------------------------------------------------
    working_directory = 'C:\\Users\\Russell\\Dropbox\\Hobbies\\DitheringAlgorithms'
    source_directory = 'data\\input_images'
    image_filepath = os.path.join(working_directory, source_directory, source_filepath)
    if highlight_point_count > 0:
        image_highlight_filepath = os.path.join(working_directory, source_directory, source_highlight_filepath)
    output_filepath = os.path.join(working_directory, 'outputs', 'raw_svg', 'network_dithering.svg'.format())
    # -----------------------------------------------------------------------------------------------------------------
    #                                           Map the values to a smaller image
    # -----------------------------------------------------------------------------------------------------------------
    start = time.time()
    image = np.asarray(Image.open(image_filepath)) / 255
    image = image[:, :, 0]
    image = cv2.resize(image,
                       dsize=(image.shape[1] // downsampling_factor,
                              image.shape[0] // downsampling_factor),
                       interpolation=cv2.INTER_CUBIC)

    # Flatten the image colors
    background_ones_count = np.sum(image == 1)
    new_values = np.linspace(0, 1, image.size - background_ones_count)
    new_values = np.concatenate([new_values, np.ones(background_ones_count)])
    indices = np.argsort(image.flatten())
    flattened_image_values = np.ones_like(image.flatten())
    flattened_image_values[indices] = new_values
    image = flattened_image_values.reshape(image.shape)

    # Ensure the image is in the proper range
    image = 1 - np.clip(image, 0, 1)

    print("Loaded and processed image in {}", time.time() - start)
    # -----------------------------------------------------------------------------------------------------------------
    #                                           Pick points and construct graph
    # -----------------------------------------------------------------------------------------------------------------
    start = time.time()
    np.random.seed(1234)
    if edge_weighted_point_count > 0:
        smoothed_image = scipy.ndimage.gaussian_filter(abs(image), sigma=25, order=0)
        edges_of_images = scipy.ndimage.sobel(smoothed_image)
        smoothed_edges = scipy.ndimage.gaussian_filter(abs(edges_of_images), sigma=5, order=0)
        new_points_inds = np.random.choice(smoothed_edges.size, edge_weighted_point_count, replace=False,
                                           p=smoothed_edges.flatten() / np.sum(smoothed_edges))
        edge_points = np.stack(np.unravel_index(new_points_inds, edges_of_images.shape), axis=1)

    if highlight_point_count > 0:
        image_highlights = np.asarray(Image.open(image_highlight_filepath).convert('L')) / 255
        image_highlights = cv2.resize(image_highlights,
                                      dsize=(image_highlights.shape[1] // downsampling_factor,
                                             image_highlights.shape[0] // downsampling_factor),
                                      interpolation=cv2.INTER_CUBIC)
        image_highlights = image_highlights < 0.5
        image_highlights = scipy.ndimage.gaussian_filter(abs(image_highlights.astype(float)), sigma=15, order=0)
        new_points_inds = np.random.choice(image_highlights.size, highlight_point_count, replace=False,
                                           p=image_highlights.flatten() / np.sum(image_highlights))
        highlight_points = np.stack(np.unravel_index(new_points_inds, image_highlights.shape), axis=1)

    if random_node_count > 0:
        random_points = np.random.rand(random_node_count, 2) * image.shape


    list_of_active_points = []
    if random_node_count != 0:
        list_of_active_points.append(random_points.astype(np.float64))

    if edge_weighted_point_count != 0:
        list_of_active_points.append(edge_points.astype(np.float64))

    if highlight_point_count != 0:
        list_of_active_points.append(highlight_points.astype(np.float64))
    points = np.concatenate(list_of_active_points)

    triangles = scipy.spatial.Delaunay(points).simplices
    G, pos, _ = get_graph_and_node_pos(points, triangles)

    print("Picked random points and constructed graph in {}".format(time.time() - start))
    # -----------------------------------------------------------------------------------------------------------------
    #                                           Weight Graph Edges According to some properties.
    # -----------------------------------------------------------------------------------------------------------------
    start = time.time()
    for a_triangle in triangles:
        a_triangle_coords = np.stack([pos[ind] for ind in a_triangle])
        a_triangle_area = shoelace(a_triangle_coords)
        in_triangle_points = get_grid_points_in_hull(a_triangle_coords, image.shape).astype(int)
        in_triangle_values = image[in_triangle_points[:, 0], in_triangle_points[:, 1]]
        in_triangle_sum = sum(in_triangle_values)
        for u, v in zip(a_triangle, np.roll(a_triangle, 1)):
            G[u][v]['length'] = np.linalg.norm(pos[u] - pos[v])
            G[u][v]['fill_chance'] += in_triangle_sum / max(a_triangle_area, 1)
    print("Compute triangle statistics and edge fill chances {}".format(time.time() - start))
    # -----------------------------------------------------------------------------------------------------------------
    #                                           Normalize Fill Chance values
    # -----------------------------------------------------------------------------------------------------------------
    # max_edge = max(data['fill_chance'] for n1, n2, data in G.edges(data=True))
    # for u, v, data in G.edges(data=True):
    #     G[u][v]['fill_chance'] /= max_edge
    # -----------------------------------------------------------------------------------------------------------------
    #                                           Apply Ordered Dithering
    # -----------------------------------------------------------------------------------------------------------------
    # start = time.time()
    # G_euler = nx.eulerize(G)
    # G_eulerian_circuit = list(nx.eulerian_circuit(G_euler))
    # number_of_euler_edges = len(G_eulerian_circuit)
    # start2 = time.time()
    # ordered_strengths = get_order(number_of_euler_edges)
    # order_time = time.time() - start2
    # print("Computed ordered strengths {}", order_time)
    #
    # for i, an_edge in enumerate(G_eulerian_circuit):
    #     if an_edge in G.edges:
    #         G[an_edge[0]][an_edge[1]]['cut_off'] = ordered_strengths[i]
    #         # print(ordered_strengths[i])
    # print("Compute Eularian Circuit, and apply ordered stengths along it: {}", time.time() - start - order_time)
    # -----------------------------------------------------------------------------------------------------------------
    #                                           Apply Error Diffusing Dithering
    # -----------------------------------------------------------------------------------------------------------------
    start = time.time()
    lengths = []
    for u, v, data in G.edges(data=True):
        G[u][v]['fill_chance'] = G[u][v]['fill_chance'] * darkness
        lengths.append(G[u][v]['length'])

    for u, v, data in G.edges(data=True):
        G[u][v]['cut_off'] = .5
        G[u][v]['was_visited'] = True
        residual = G[u][v]['fill_chance'] - round(G[u][v]['fill_chance'])
        G[u][v]['fill_chance'] = round(G[u][v]['fill_chance'])
        u_edges = list(G.edges(u))
        v_edges = list(G.edges(v))
        nearby_edges = u_edges + v_edges

        unvisited_nearby_edges = []
        if nearby_edges is not None:
            for p, q in nearby_edges:
                if G[p][q].get('was_visited') is None:
                    unvisited_nearby_edges.append((p, q))

        if len(unvisited_nearby_edges) > 0:
            total_nearby_length = 0
            for p, q in unvisited_nearby_edges:
                total_nearby_length += G[p][q]['length']

            for p, q in unvisited_nearby_edges:
                G[p][q]['fill_chance'] += residual * G[p][q]['length'] / total_nearby_length \
                                          * G[u][v]['length'] / total_nearby_length

    # G[u][v]['fill_chance'] = np.power(G[u][v]['fill_chance'], 3 / 4)
    print("Error diffusion {}".format(time.time() - start))
    # -----------------------------------------------------------------------------------------------------------------
    #                                           Write to SVG file
    # -----------------------------------------------------------------------------------------------------------------
    start = time.time()
    dwg = svgwrite.Drawing(output_filepath, size=(11 * 96, 8.5 * 96))
    inkscape = svgwrite.extensions.Inkscape(dwg)
    layer = inkscape.layer(label="network_lines", locked=False)
    dwg.add(layer)

    for u, v, data in G.edges(data=True):
        p1, p2 = np.round(pos[u]), np.round(pos[v])
        data['cut_off'] = .5
        if data['fill_chance'] > data['cut_off']:
            layer.add(svgwrite.shapes.Line(
                np.around(p1[::-1], decimals=2), np.around(p2[::-1], decimals=2),
                stroke='black'))
            # stroke_opacity="{:.2}".format(a_fill_chance)))
    dwg.save()
    print("Wrote to SVG in {}", time.time() - start)


if __name__ == "__main__":
    start()
