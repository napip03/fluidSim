
import matplotlib.pyplot as plt
import numpy as np


def initialize_simulation(Nx, Ny, NL, rho0):
    """
    Initializes the simulation grid, including the distribution function and
    the initial random perturbations. Returns initial conditions for F, rho, and a cylinder mask.
    """
    # initial distribution function (F) and random perturbations
    F = np.ones((Ny, Nx, NL))
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)

    # set initial density 
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))

    # calculate density and normalize 
    rho = np.sum(F, axis=2)
    for i in range(NL):
        F[:, :, i] *= rho0 / rho
    cylinder = (X - Nx / 4) ** 2 + (Y - Ny / 2) ** 2 < (Ny / 4) ** 2
    return F, cylinder


def compute_macroscopic_variables(F, cxs, cys):
    """
    Computes macroscopic fluid variables: density, velocity in x and y directions.
    """
    rho = np.sum(F, axis=2)

    ux = np.sum(F * cxs, axis=2) / rho
    uy = np.sum(F * cys, axis=2) / rho

    return rho, ux, uy


def collision_step(F, Feq, tau):
    """
    Performs the collision step in the lattice Boltzmann method.
    F is the distribution function, Feq is the equilibrium distribution, and tau is the relaxation time.
    """
    return F - (1.0 / tau) * (F - Feq)


def apply_boundary_conditions(F, cylinder):
    """
    Applies reflective boundary conditions at the cylinder.
    """
    bndryF = F[cylinder, :]
    bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]
    F[cylinder, :] = bndryF
    return F


def equilibrium_distribution(F, rho, ux, uy, cxs, cys, weights):
    """
    Computes the equilibrium distribution Feq based on the macroscopic variables.
    """
    Feq = np.zeros(F.shape)
    for i, cx, cy, w in zip(range(9), cxs, cys, weights):
        Feq[:, :, i] = rho * w * (1 + 3 * (cx * ux + cy * uy) +
                                  9 * (cx * ux + cy * uy) ** 2 / 2 - 3 * (ux ** 2 + uy ** 2) / 2)
    return Feq


def update_fluid(F, Feq, tau, cylinder):
    """
    Updates the distribution function and applies boundary conditions.
    """
    # Perform collision step
    F = collision_step(F, Feq, tau)

    # Apply boundary conditions
    F = apply_boundary_conditions(F, cylinder)

    return F


def plot_vorticity(ux, uy, cylinder, Nx, Ny, it, Nt):
    """
    Plots the vorticity in real-time during the simulation.
    """
    if it % 10 == 0 or it == Nt - 1:
        plt.cla()
        ux[cylinder] = 0
        uy[cylinder] = 0
        vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (
                np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        vorticity[cylinder] = np.nan
        vorticity = np.ma.array(vorticity, mask=cylinder)
        plt.imshow(vorticity, cmap='PuOr')
        plt.imshow(~cylinder, cmap='gray', alpha=0.3)
        plt.clim(-0.1, 0.1)
        ax = plt.gca()
        ax.invert_yaxis()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_aspect('equal')
        plt.pause(0.001)


def main():
    # simulation parameters
    Nx, Ny = 400, 100 # dimensions
    rho0, tau = 100, 0.6 # 
    Nt = 4000 # number of iterations
    plotRealTime = True

    # lattice speeds and weights
    NL = 9
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])

    # init SIM
    F, cylinder = initialize_simulation(Nx, Ny, NL, rho0)

    # plt
    fig = plt.figure(figsize=(4, 2), dpi=80)

    # main loop 
    for it in range(Nt):
        print(f"Iteration: {it}")

        # drift step:
        for i, cx, cy in zip(range(NL), cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        rho, ux, uy = compute_macroscopic_variables(F, cxs, cys)
        Feq = equilibrium_distribution(F, rho, ux, uy, cxs, cys, weights)

        # update and apply boundary conditions
        F = update_fluid(F, Feq, tau, cylinder)

        # plot vorticity in real time
        if plotRealTime:
            plot_vorticity(ux, uy, cylinder, Nx, Ny, it, Nt)

    # Save the final result
    plt.savefig('latticeboltzmann.png', dpi=240)
    plt.show()


if __name__ == "__main__":
    main()
