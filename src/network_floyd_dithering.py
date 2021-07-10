import os
import cv2
import scipy.spatial
import numpy as np

import svgwrite.extensions
import matplotlib.pyplot as plt

import networkx as nx
from PIL import Image


def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices_coords in each revised Voronoi regions.
    vertices_coords : list of tuples
        Coordinates for revised Voronoi vertices_coords. Same as coordinates
        of input vertices_coords, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def is_in_hull(point_list,
               hull_vertices):
    """
    Determine if the list of points P lies inside the hull
    :return: list
    List of boolean where true means that the point is inside the convex hull
    """

    hull = scipy.spatial.ConvexHull(hull_vertices)
    A = hull.equations[:, 0:-1]
    b = np.transpose(np.array([hull.equations[:, -1]]))
    return np.all((A @ np.transpose(point_list)) <= np.tile(-b, (1, len(point_list))), axis=0)


def get_grid_points_in_hull(hull, max_dimensions):
    low_bound = np.min(hull, axis=0)
    low_bound = np.maximum(low_bound, np.zeros_like(low_bound))
    high_bound = np.max(hull, axis=0)
    high_bound = np.minimum(high_bound, np.asarray(max_dimensions) - 1)

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


if __name__ == "__main__":
    max_points = 100000
    cut_off_ind = 10000
    # -----------------------------------------------------------------------------------------------------------------
    #                                           Load Data
    # -----------------------------------------------------------------------------------------------------------------
    working_directory = 'C:\\Users\\Russell\\Dropbox\\Hobbies\\DitheringAlgorithms'
    source_directory = 'data\\input_images'
    source_file = 'Russell_Head.jpg'
    open_file = os.path.join(working_directory, source_directory, source_file)
    write_file = 'outputs\\raw_svg\\floyd_stippling_{}.svg'.format(source_file)

    image = np.asarray(Image.open(open_file)) / 255
    image = image[:, :, 0]
    n = 3

    # -----------------------------------------------------------------------------------------------------------------
    #                                           Map the values to a smaller image
    # -----------------------------------------------------------------------------------------------------------------
    image = cv2.resize(image, dsize=(image.shape[1] // n, image.shape[0] // n), interpolation=cv2.INTER_CUBIC)
    background_ones_count = np.sum(image == 1)
    new_values = np.linspace(0, 1, image.size - background_ones_count)
    new_values = np.concatenate([new_values, np.ones(background_ones_count)])

    indices = np.argsort(image.flatten())
    flattened_image_values = np.ones_like(image.flatten())
    flattened_image_values[indices] = new_values
    image = flattened_image_values.reshape(image.shape)
    image = 1 - np.clip(image, 0, 1)

    # -----------------------------------------------------------------------------------------------------------------
    #                                        Construct a Voronoi Diagram
    # -----------------------------------------------------------------------------------------------------------------
    # make up data points
    # print("trying to run")
    np.random.seed(1234)

    xs = np.random.uniform(low=0.0, high=image.shape[0], size=max_points)
    ys = np.random.uniform(low=0.0, high=image.shape[1], size=max_points)
    points = np.stack([xs, ys], axis=1)

    # compute Voronoi tesselation
    vor = scipy.spatial.Voronoi(points)

    # plot
    regions, vertices_coords = voronoi_finite_polygons_2d(vor)
    # print(regions)

    # -----------------------------------------------------------------------------------------------------------------
    #                                        Build a Graph
    # -----------------------------------------------------------------------------------------------------------------

    G = nx.Graph()
    G.add_nodes_from([
        (i, {'coords': coords}) for i, coords in enumerate(vertices_coords)
    ])

    for region_vert_indices in regions:
        for i in range(len(region_vert_indices)):
            side_length = np.sqrt(np.linalg.norm(region_vert_indices[i] - region_vert_indices[(i + 1) % len(region_vert_indices)]))
            G.add_edge(region_vert_indices[i],
                       region_vert_indices[(i + 1) % len(region_vert_indices)],
                       side_len=side_length,
                       side_color=0.0,
                       fill_chance=0.0,
                       visited=False)

    region_area = np.empty(len(regions))
    region_sum = np.empty(len(regions))
    region_sums = []
    region_areas = []

    for i, a_region in enumerate(regions):
        # So we have
        # region_area: the number of pixels in that region
        # region_sum: the sum of the pixels in that same area,
        # side length: side lengths,

        polygon = vertices_coords[a_region]
        active_points = get_grid_points_in_hull(polygon, image.shape)
        active_points = active_points.astype(int)
        region_sum = np.sum(np.take(image, active_points))
        region_area = len(active_points)
        perimeter = 0

        region_sums.append(region_sum)
        region_areas.append(region_area)

        for j in range(len(a_region)):
            edge_data = G.get_edge_data(a_region[j], a_region[(j + 1) % len(a_region)])
            perimeter += edge_data['side_len']

        for j in range(len(a_region)):
            node_i, node_j = a_region[j], a_region[(j + 1) % len(a_region)]
            edge_data = G.get_edge_data(node_i, node_j)

            G[node_i][node_j]['fill_chance'] += region_sum * \
                                                edge_data['side_len'] / perimeter

    svg_file_name = "outputs/raw_svg/network_dithering.svg"
    page_size = (8.5 * 96,
                 11 * 96)
    dwg = svgwrite.Drawing(svg_file_name,
                           size=page_size)
    inkscape = svgwrite.extensions.Inkscape(dwg)
    layer = inkscape.layer(label="network_lines",
                           locked=False)
    dwg.add(layer)

    fill_chances = []
    for u, v, edge_info in G.edges(data=True):
        fill_chances.append(G[node_i][node_j]['fill_chance'])

    fill_chances = np.sort(fill_chances)
    fill_cut_off = fill_chances[cut_off_ind]
    for u, v, edge_info in G.edges(data=True):
        p1, p2 = np.round(G.nodes[u]['coords']), np.round(G.nodes[v]['coords'])
        # if not (any(p1 < 0) or any(p1 > image.shape) or any(p2 < 0) or any(p2 > image.shape)):
        if edge_info['fill_chance'] > fill_cut_off:
            layer.add(svgwrite.shapes.Line(
                p1, p2,
                stroke='black'))
    print("plotting")


    plt.show()
    dwg.save()