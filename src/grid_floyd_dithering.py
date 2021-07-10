from functools import lru_cache

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


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------------------------------------------
    #                                               Load Data
    # ---------------------------------------------------------------------------------------------------------------------
    working_directory = 'C:\\Users\\Russell\\Dropbox\\Hobbies\\DitheringAlgorithms'
    source_directory = 'data\\input_images'
    source_file = 'Russell_Head.jpg'
    open_file = os.path.join(working_directory, source_directory, source_file)

    write_file = 'outputs\\raw_svg\\floyd_stippling_{}.svg'.format(source_file)

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

    from_horizontal: dict[str, dict[tuple[int, int], float]] = {
        'horizontal': {
            (0, 1): 2 / 3
        },
        'vertical': {
            (0, 0): 1 / 6,
            (0, 1): 1 / 6,
        }
    }

    from_vertical: dict[str, dict[tuple[int, int], float]] = {
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
    page_size = (8.5 * 96, 11 * 96)

    dims = image.shape
    scale = np.max(np.asarray(page_size[::-1]) / np.asarray(dims))

    dwg = svgwrite.Drawing(write_file, size=page_size)
    inkscape = svgwrite.extensions.Inkscape(dwg)

    layer = inkscape.layer(label="horizontal_lines", locked=False)
    dwg.add(layer)
    for i, a_line in enumerate(horizontal_lines):
        grouped_line_endpoints = convert_dense_to_endpoints(a_line)
        for a_color, a_line_endpoints in grouped_line_endpoints.items():
            if a_color == 1.0:
                continue
            int_color = int(a_color * 255)
            curr_color = rgb2hex(int_color, int_color, int_color)

            for x_0, x_1 in a_line_endpoints:
                line = svgwrite.path.Path(
                    d="M {}, {} H {}".format(x_0 * scale, i * scale, x_1 * scale),
                    stroke=curr_color
                )
                layer.add(line)

    layer = inkscape.layer(label="vertical_lines", locked=False)
    dwg.add(layer)
    for i, a_line in enumerate(vertical_lines.T):
        grouped_line_endpoints = convert_dense_to_endpoints(a_line)
        for a_color, a_line_endpoints in grouped_line_endpoints.items():
            if a_color == 1.0:
                continue
            int_color = int(a_color ** 2 * 255)
            curr_color = rgb2hex(int_color, int_color, int_color)

            for y_0, y_1 in a_line_endpoints:
                line = svgwrite.path.Path(
                    d="M {},{} V {}".format(i * scale, y_0 * scale, y_1 * scale),
                    stroke=curr_color
                )
                layer.add(line)
    dwg.save()
    dwg
