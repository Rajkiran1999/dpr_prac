import numpy as np
import matplotlib.pyplot as plt


def plot_error_rate_diagram(constellations, ber=False, snr_min=0, snr_max=30, num_snr_values=20,
                            num_samples=2e4):
    """
    Create a SNR-error-rate plot for a list of constellations
    Args:
        constellations: list of constellation objects
        ber: True if we want bit error rate instead of symbol error rate to be plotted
        snr_min: minimum snr in the plot
        snr_max: maximum snr in the plot
        num_snr_values: resolution of snr in simulation
        num_samples: number of samples used to estimate each error rate
    """
    snrs = np.linspace(snr_min, snr_max, num_snr_values)

    for constellation in constellations:
        error_rates =  # TODO (TASK 10)
        plt.semilogy(snrs, error_rates, marker='o', label=constellation.name)

    plt.xlabel('SNR in dB')
    if ber:
        plt.ylabel('bit error rate')
    else:
        plt.ylabel('symbol error rate')
    plt.legend()


def plot_decision_regions(constellation, bounding_box=1.5, num_grid_points=500):
    """
    Highlight decision regions of maximum likelihood detector for a given constellation
    Args:
        constellation: constellation object
        bounding_box: bounding box of plot
        num_grid_points: resolution of plot per axis
    """
    # create a (quadratic) grid in the IQ-plane, these are the points where the discriminant values
    # are evaluated to estimate the sent label
    grid_points = np.linspace(-bounding_box, bounding_box, num_grid_points)
    real, im = np.meshgrid(grid_points, grid_points)
    points = np.stack([real, im], axis=-1)
    estimated_labels =  # TODO (TASK 11)
    plt.pcolormesh(real, im, estimated_labels, shading='auto', cmap='coolwarm')
    plt.title('decision regions of the symbols')
    plt.colorbar(label='symbol label')
    constellation.plot_constellation_points(label=True)


def save_scatter_plot(filename):
    """
    Helper function to save scatter plots as raster image for usage with pgplots
    Args:
        filename: path to store the raster image
    """
    plt.axis([-2.0, 2.0, -2.0, 2.0])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=True, dpi=200)
