import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_spectrum(Lambda):
    """
    Plot the spectrum of the discrete dynamics
    :return:
    """
    if Lambda.is_complex():
        # Ensure the tensor is detached from the computation graph and cloned
        Lambda = Lambda.clone().detach()
        # Extract real and imaginary parts
        real_parts = Lambda.real
        imaginary_parts = Lambda.imag
    else:
        # Fallback if Lambda is somehow not a complex tensor
        # Assume Lambda is structured with real parts in the first column and imaginary parts in the second
        Lambda = Lambda.clone().detach()
        real_parts = Lambda[:, 0]
        imaginary_parts = Lambda[:, 1]

    # Plotting
    plt.figure(figsize=(10, 10))
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
    