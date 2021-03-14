import matplotlib.pylab as plt
import os

import click
import numpy as np

from mlops_examples.domain.data_generation.mandelbrot_generator import MandelbrotGenerator


@click.command()
@click.option("--output-directory", type=click.Path(dir_okay=True, file_okay=False), default="output")
def main(output_directory: str):
    os.makedirs(output_directory, exist_ok=True)
    result = MandelbrotGenerator().get(255**2)

    image  = np.empty((255, 255), dtype=np.short)
    x_indices = np.sort(np.unique(result.values[:, 0]))
    y_indices = np.sort(np.unique(result.values[:, 1]))
    for row in result.values:
        image[np.where(y_indices == row[1])[0][0], np.where(x_indices == row[0])[0][0]] = row[2]

    plt.imshow(image, cmap="plasma", vmax=25)
    plt.axis("off")
    plt.show()
    input('enter to quit')


if __name__ == "__main__":
    main()