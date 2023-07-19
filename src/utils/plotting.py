import imageio
import matplotlib.pyplot as plt
import numpy as np


def setup_ax(ax, title, xlim=(-1000, 1000), ylim=(-1000, 1000)):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def save_figures_to_gif(figures, filename):
    # Create a list to store images
    images = []

    for fig in figures:
        # Render the figure to a PNG image in memory
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()

        # Get figure dimensions
        ncols, nrows = 2400, 1200

        # Convert image to numpy array and reshape
        try:
            image = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        except:
            raise ValueError("All figures should have the same size.")

        # Append image to list of images
        images.append(image)

        # Close the figure
        plt.close(fig)
    # Save images as an animated GIF
    imageio.mimsave(filename, images)
