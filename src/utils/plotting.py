import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import torch


def plot_spectrum(Lambda):
    """
    Plot the spectrum of the discrete dynamics
    :return:
    """
    Lambda = Lambda.clone().detach()

    Lambda = Lambda.view(-1)
    if Lambda.is_complex():
      real_parts = Lambda.real
      imaginary_parts = Lambda.imag
    else:
      real_parts = Lambda
      imaginary_parts = torch.zeros(Lambda.shape)



    # Plotting
    plt.figure(figsize=(10,9))
    plt.scatter(real_parts, imaginary_parts, color='red', marker='o')
    plt.title('Complex Eigs (Discrete Dynamics <-> dt=1)')
    plt.xlabel('Real Part')
    plt.ylabel('Imaginary Part')
    # Adding a unit circle
    circle = Circle((0, 0), 1, fill=False, color='blue', linestyle='--')
    plt.gca().add_patch(circle)

    plt.grid(True)
    plt.axhline(y=0, color='k')  # Adds x-axis
    plt.axvline(x=0, color='k')  # Adds y-axis
    plt.show()
