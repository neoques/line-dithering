from functools import lru_cache
import re
from PIL import Image
import numpy as np
import cv2
import os
import svgwrite
import svgwrite.extensions


@lru_cache(maxsize=255)
def rgb2hex(r, g, b):
    clamp = lambda x: max(0, min(x, 255))
    return "#{0:02x}{1:02x}{2:02x}".format(clamp(r), clamp(g), clamp(b))


def convert_dense_to_endpoints(a_array_of_image: list) -> list(list()):
    a_array_of_image = np.asarray(a_array_of_image)
    unique_values, return_indexes = np.unique(a_array_of_image, return_index=True)
    all_lines = dict()

    for a_val in unique_values:
        first_indexes = []
        final_indexes = []
        active_indexes = np.where(a_array_of_image == a_val)[0]
        first_indexes.append(active_indexes[0])  # for a particular value
        for i, a_index in enumerate(active_indexes[1:]):
            if a_index != active_indexes[i] + 1:
                final_indexes.append(active_indexes[i] + 1)
                first_indexes.append(a_index)
        final_indexes.append(active_indexes[-1])
        assert len(final_indexes) == len(first_indexes)
        all_lines[a_val] = list(zip(first_indexes, final_indexes))
    return all_lines


def simplify_point_str(paths: list):
    new_pathes = []
    start_points = {}
    end_points = {}
    match_front = "M \d+, ?\d+"
    i = 0
    while len(paths) > 0:
        if len(paths) == 1:
            print()
        a_path = paths.pop()
        curr_line_vals = a_path.replace(",", " ").split(" ")
        curr_line_vals = list(filter(None, curr_line_vals))
        start_pt = (curr_line_vals[1], curr_line_vals[2])
        if curr_line_vals[3] == "H":
            end_pt = (curr_line_vals[4], curr_line_vals[2])
        elif curr_line_vals[3] == "V":
            end_pt = (curr_line_vals[1], curr_line_vals[4])

        if start_pt in start_points:
            line_to_replace_ind = start_points[start_pt]
            old_line = new_pathes[line_to_replace_ind]
            kept_old_line = old_line[re.match(match_front, old_line).end():]

            if curr_line_vals[3] == "V":
                new_pathes[line_to_replace_ind] = f"M {end_pt[0]}, {end_pt[1]} V {start_pt[0]}{kept_old_line}"
            else:
                new_pathes[line_to_replace_ind] = f"M {end_pt[0]}, {end_pt[1]} H {start_pt[0]}{kept_old_line}"

        elif start_pt in end_points:
            line_to_replace_ind = end_points[start_pt]
            old_line = new_pathes[line_to_replace_ind]
            kept_new_line = a_path[re.match(match_front, a_path).end():]
            new_pathes[line_to_replace_ind] = old_line + kept_new_line

        elif end_pt in start_points:
            line_to_replace_ind = start_points[end_pt]
            old_line = new_pathes[line_to_replace_ind]
            kept_old_line = old_line[re.match(match_front, old_line).end():]
            new_pathes[line_to_replace_ind] = a_path + kept_old_line

        else:
            start_points[start_pt] = i
            end_points[end_pt] = i
            new_pathes.append(a_path)
            i += 1
    return new_pathes


class Line:
    @staticmethod
    def from_string(a_path: str):
        vals = a_path.replace(',', " ").split(" ")
        vals[1], vals[2], vals[4] = int(vals[1]), int(vals[2]), int(vals[4])
        is_horizontal = vals[3] == 'H'
        is_vertical = vals[3] == 'V'
        if not is_horizontal and not is_vertical:
            raise NotImplementedError

        x1, y1, = vals[1], vals[2]
        if is_horizontal:
            x2 = vals[4]
            y2 = y1
        if is_vertical:
            y2 = vals[4]
            x2 = x1
        return Line(is_vertical, is_horizontal, x1 + int((1.5+1/12)*96), x2 + int((1.5+1/12)*96), y1, y2)

    def __init__(self, is_vert: bool, is_horz: bool, x1: int, x2: int, y1: int, y2: int):
        # force x1 < x2
        if x1 > x2:
            x1, x2 = x2, x1
        self.is_vertical = is_vert
        self.is_horizontal = is_horz
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def shift(self, shift_interval):
        x_min = min(self.x1, self.x2) // shift_interval
        self.x1, self.x2 = self.x1 - x_min * shift_interval, self.x2 - x_min * shift_interval
        return str(self)

    def split_at_x(self, xsplit):
        if self.x1 < xsplit < self.x2:
            return Line(self.is_vertical, self.is_horizontal, self.x1, xsplit, self.y1, self.y2), \
                   Line(self.is_vertical, self.is_horizontal, xsplit, self.x2, self.y1, self.y2)
        else:
            return None

    def __str__(self):
        if self.is_horizontal:
            return f"M {self.x1},{self.y1} H {self.x2}"
        if self.is_vertical:
            return f"M {self.x1},{self.y1} V {self.y2}"


def group_paths(paths, boundary_interval: int, max_pages=5):
    bi = boundary_interval
    path_groups = [[] for _ in range(max_pages)]
    for a_path in paths:
        a_line = Line.from_string(a_path)
        if a_line.is_horizontal:
            i = 1
            x1_page = a_line.x1 // bi
            x2_page = a_line.x2 // bi
            while not a_line.split_at_x(bi * i):
                i += 1
                if i > max_pages:
                    path_groups[x1_page].append(a_line.shift(bi))
                    break
            else:
                l1, l2 = a_line.split_at_x(bi * i)
                path_groups[x1_page].append(l1.shift(bi))
                path_groups[x2_page].append(l2.shift(bi))

        elif a_line.is_vertical:
            x1_page = a_line.x1 // bi
            path_groups[x1_page].append(a_line.shift(bi))
    return path_groups


def main():
    # ---------------------------------------------------------------------------------------------------------------------
    #                                               Load Data
    # ---------------------------------------------------------------------------------------------------------------------
    working_directory = 'C:\\Users\\Russell\\Dropbox\\Hobbies\\DitheringAlgorithms'
    source_directory = 'data\\input_images'
    source_file = 'founders.jpg'
    source_file_name = source_file.split('.')[0]
    open_file = os.path.join(working_directory, source_directory, source_file)

    write_file = 'outputs\\raw_svg\\floyd_stippling_{}'.format(source_file_name)

    image = np.asarray(Image.open(open_file)) / 255
    image = image[:, :, 0]
    n = 3

    image = cv2.resize(image, dsize=(image.shape[1] // n, image.shape[0] // n), interpolation=cv2.INTER_CUBIC)
    background_ones_count = np.sum(image == 1)
    new_values = np.linspace(0, 1, image.size - background_ones_count)
    new_values = np.concatenate([new_values, np.ones(background_ones_count)])

    inds = np.argsort(image.flatten())
    flattened_image_values = np.ones_like(image.flatten())
    flattened_image_values[inds] = new_values
    image = flattened_image_values.reshape(image.shape)
    image = np.clip(image, 0, 1)

    vertical_lines = (image[0:-1] + image[1:]) / 2
    horizontal_lines = (image[:, 0:-1] + image[:, 1:]) / 2

    # ---------------------------------------------------------------------------------------------------------------------
    #                                     Construct Dithering Parameters
    # ---------------------------------------------------------------------------------------------------------------------
    allowed_values = np.linspace(0.0, 1.0, 2)
    in_vertical = lambda i, j: i < vertical_lines.shape[0] and j < vertical_lines.shape[1]
    in_horizontal = lambda i, j: i < horizontal_lines.shape[0] and j < horizontal_lines.shape[1]

    from_horizontal = {
        'horizontal': {
            (0, 1): 2 / 3
        },
        'vertical': {
            (0, 0): 1 / 6,
            (0, 1): 1 / 6,
        }
    }

    from_vertical = {
        'horizontal': {
            (1, -1): 1 / 6,
            (1, 0): 1 / 6
        },
        'vertical': {
            (1, 0): 2 / 3,
        }
    }

    # ---------------------------------------------------------------------------------------------------------------------
    #                                     Dither the horizontal and vertical grids
    # ---------------------------------------------------------------------------------------------------------------------
    for i in range(image.shape[0]):
        # Horizontal row update
        for j in range(image.shape[1] - 1):
            curr_pnt = (i, j)
            old_line = horizontal_lines[i, j]
            new_line = allowed_values[np.abs(allowed_values - old_line).argmin()]
            horizontal_lines[i, j] = new_line
            qe = old_line - new_line

            for index, weight in from_horizontal['horizontal'].items():
                new_index = [a_a + a_b for a_a, a_b in zip(curr_pnt, index)]
                if in_horizontal(*new_index):
                    horizontal_lines[new_index[0], new_index[1]] += weight * qe

            for index, weight in from_horizontal['vertical'].items():
                new_index = [a_a + a_b for a_a, a_b in zip(curr_pnt, index)]
                if in_vertical(*new_index):
                    vertical_lines[new_index[0], new_index[1]] += weight * qe

        if i == image.shape[0] - 1:
            break

        # Vertical row update
        for j in range(image.shape[1]):
            curr_pnt = (i, j)
            old_line = vertical_lines[i, j]
            new_line = allowed_values[np.abs(allowed_values - old_line).argmin()]
            vertical_lines[i, j] = new_line
            qe = old_line - new_line  # quant error

            for index, weight in from_vertical['horizontal'].items():
                new_index = [a_a + a_b for a_a, a_b in zip(curr_pnt, index)]
                if in_horizontal(*new_index):
                    horizontal_lines[new_index[0], new_index[1]] += weight * qe

            for index, weight in from_vertical['vertical'].items():
                new_index = [a_a + a_b for a_a, a_b in zip(curr_pnt, index)]
                if in_vertical(*new_index):
                    vertical_lines[new_index[0], new_index[1]] += weight * qe

    # ---------------------------------------------------------------------------------------------------------------------
    #                                     Write the grid to a svg
    # ---------------------------------------------------------------------------------------------------------------------
    # page_size = (8.5 * 96, 11 * 96)
    page_size = (11 * 96, 17 * 96)
    boundaries_intervals = 11 * 96
    dims = image.shape
    scale = np.max(np.asarray(page_size[::-1]) / np.asarray(dims))
    # images = 3
    paths = []
    for i, a_line in enumerate(horizontal_lines):
        grouped_line_endpoints = convert_dense_to_endpoints(a_line)
        for a_color, a_line_endpoints in grouped_line_endpoints.items():
            if a_color == 1.0:
                continue
            int_color = int(a_color * 255)
            curr_color = rgb2hex(int_color, int_color, int_color)
            curr_color = "black"
            for x_0, x_1 in a_line_endpoints:
                paths.append("M {},{} H {}".format(int(x_0 * scale), int(i * scale), int(x_1 * scale)))

    # layer = inkscape.layer(label="vertical_lines", locked=False)
    # dwg.add(layer)
    for i, a_line in enumerate(vertical_lines.T):
        grouped_line_endpoints = convert_dense_to_endpoints(a_line)
        for a_color, a_line_endpoints in grouped_line_endpoints.items():
            if a_color == 1.0:
                continue
            int_color = int(a_color ** 2 * 255)
            curr_color = rgb2hex(int_color, int_color, int_color)
            curr_color = "black"

            for y_0, y_1 in a_line_endpoints:
                paths.append("M {},{} V {}".format(int(i * scale), int(y_0 * scale), int(y_1 * scale)))

    paths_grouped = group_paths(paths, boundaries_intervals)
    # simplify
    for i, paths in enumerate(paths_grouped):
        if len(paths) == 0:
            continue
        paths = simplify_point_str(paths)
        dwg = svgwrite.Drawing(write_file + f"_{i}.svg", size=page_size)
        inkscape = svgwrite.extensions.Inkscape(dwg)
        layer = inkscape.layer(label="horizontal_lines", locked=False)
        dwg.add(layer)
        line = svgwrite.path.Path(
            d=" ".join(paths),
            stroke=curr_color,
            fill='none'
        )
        layer.add(line)
        dwg.save()


if __name__ == "__main__":
    main()
