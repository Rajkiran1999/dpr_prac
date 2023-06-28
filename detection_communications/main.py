from absl import logging
import matplotlib.pyplot as plt

from constellations import QAMConstellation, PSKConstellation
from experiments import plot_error_rate_diagram, plot_decision_regions


def main():
    """
    GENERAL REMARKS / HINTS *****************************************************************************
    *) The framework of the code to this assignment is already prepared. You only have to complete the
       code at positions introduced by 'TODO'. This is to simplify the assignment for you.
       Feel free to also change other parts of the code if you want to.
    *) Some tasks are more challenging than others. The assignment is designed progressively. If you
       struggle to cope with the backmost tasks, just comment the respective code in this main function.
    *) Have fun!
    """

    # TASK 9: *******************************************************************************************
    # Plot the constellation schemes of, e.g., 16-QAM and 8-PSK
    qam = QAMConstellation(16, 0.0, 10)
    qam.plot_constellation_points()
    plt.show()
    psk = PSKConstellation(8, 0.0, 10)
    psk.plot_constellation_points()
    plt.show()

    # TASK 10: ******************************************************************************************
    # Determine the symbol error rates for 16-QAM and 8-PSK for SNR = 10dB
    number_of_samples = int(1e6)
    logging.info("error rates with SNR = 10dB")
    logging.info(f"  16-QAM: {qam.determine_symbol_error_rate(number_of_samples):.2e}")
    logging.info(f"   8-PSK: {psk.determine_symbol_error_rate(number_of_samples):.2e}")

    # Create SNR-symbol-error-rate plots for 8-PSK, 16-PSK, 4-QAM and 16-QAM
    consts = [
        PSKConstellation(8, 0, 10),
        PSKConstellation(16, 0, 10),
        QAMConstellation(4, 0, 10),
        QAMConstellation(16, 0, 10)
    ]

    plot_error_rate_diagram(consts)
    plt.show()

    # TASK 11: ******************************************************************************************
    # Create SNR-symbol-error-rate plots for 16-PSK for alpha = [0.0, 0.1, 0.2, 0.5]
    consts = [
        PSKConstellation(16, 0, 10),
        PSKConstellation(16, 0.1, 10),
        PSKConstellation(16, 0.2, 10),
        PSKConstellation(16, 0.5, 10)
    ]

    plot_error_rate_diagram(consts, snr_max=80, num_snr_values=50)
    plt.show()

    # TASK 12: ******************************************************************************************
    # Plot decision regions for 16-QAM and 16-PSK without intersymbol interference (SNR = 20dB)
    consts = [
        QAMConstellation(16, 0.0, 20),
        PSKConstellation(16, 0.0, 20)
    ]
    for const in consts:
        plot_decision_regions(const)
        plt.show()

    # Plot decision regions for 16-QAM and 16-PSK with intersymbol interference (alpha=0.8, SNR = 20dB)
    consts = [
        QAMConstellation(16, 0.8, 20),
        PSKConstellation(16, 0.8, 20)
    ]
    for const in consts:
        plot_decision_regions(const)
        plt.show()


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    main()
