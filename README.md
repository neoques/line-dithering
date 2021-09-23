# line-dithering
Modifies existing image dithering techniques to generate a line dithered image.

The Steinberg-Floyd dithering algorithm pushes the error from one node to futures ones.
In the same way we push the error through the lines.

In the grid formulation (recommended) I choose edges values based on the average of the grayscale value of two neighboring pixels.

In the network formulation I choose edges between arbitry points (possibly modified with a density mask). Then I initiate the weights according to the size and darkness of the contained pixels.

