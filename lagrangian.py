import matplotlib.pyplot as plt
import numpy as np


def initialize_particles(Nx, Ny, rho0, num_particles=10000):
    """
    Initialize particles with random positions and velocities within the fluid domain.
    Returns particle positions and velocities.
    """
    # random initial positions
    x_positions = np.random.randint(0, Nx, num_particles).astype(float)
    y_positions = np.random.randint(0, Ny, num_particles).astype(float)
    # random initial velocities
    velocities = np.random.randn(num_particles, 2) * 0.5  # Random velocities with some spread
    # normalize velocities
    velocities /= np.linalg.norm(velocities, axis=1)[:, np.newaxis]
    return x_positions, y_positions, velocities


def update_positions(x_positions, y_positions, velocities, Nx, Ny):
    """
    Update particle positions based on their velocities, and apply periodic boundary conditions.
    """
    # update particle positions
    x_positions += velocities[:, 0]
    y_positions += velocities[:, 1]
    # apply conditions
    x_positions = np.mod(x_positions, Nx)
    y_positions = np.mod(y_positions, Ny)

    return x_positions, y_positions


def apply_boundary_conditions(x_positions, y_positions, velocities, cylinder, Nx, Ny):
    """
    Reflect particles off the boundaries (ex. cylinder)
    """
    # check which particles are inside the cylinder
    inside_cylinder = (x_positions - Nx / 4) ** 2 + (y_positions - Ny / 2) ** 2 < (Ny / 4) ** 2

    # reflect velocity
    velocities[inside_cylinder] = -velocities[inside_cylinder]

    return velocities


def compute_macroscopic_variables(x_positions, y_positions, velocities, Nx, Ny):
    """
    Compute macroscopic variables: density and velocity fields from particle positions and velocities.
    """
    # compute density as the number of particles in each grid cell
    grid_density = np.zeros((Ny, Nx))
    for i in range(len(x_positions)):
        x = int(np.mod(x_positions[i], Nx))
        y = int(np.mod(y_positions[i], Ny))
        grid_density[y, x] += 1

    # velocity field 
    ux = np.zeros((Ny, Nx))
    uy = np.zeros((Ny, Nx))
    for i in range(len(x_positions)):
        x = int(np.mod(x_positions[i], Nx))
        y = int(np.mod(y_positions[i], Ny))
        ux[y, x] += velocities[i, 0]
        uy[y, x] += velocities[i, 1]

    # normalization
    nonzero = grid_density > 0
    ux[nonzero] /= grid_density[nonzero]
    uy[nonzero] /= grid_density[nonzero]

    return grid_density, ux, uy


def plot_vorticity(ux, uy, cylinder, Nx, Ny, it, Nt):
    """
    Plot the vorticity in real-time during the simulation.
    """
    if it % 10 == 0 or it == Nt - 1:
        plt.cla()
        # plt.xlim(100, 300) # zoomed in the figure
        # plt.ylim(25, 75)
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
    # init sim
    Nx, Ny = 400, 100
    rho0 = 100  # density
    Nt = 4000  # time steps / iterations
    num_particles = 10000  # num particles
    plotRealTime = True

    # cylinder boundary 
    X, Y = np.meshgrid(range(Nx), range(Ny))
    cylinder = (X - Nx / 4) ** 2 + (Y - Ny / 2) ** 2 < (Ny / 4) ** 2

    x_positions, y_positions, velocities = initialize_particles(Nx, Ny, rho0, num_particles)

    # main loop
    for it in range(Nt):
        print(f"Iteration: {it}")

        # update particle positions
        x_positions, y_positions = update_positions(x_positions, y_positions, velocities, Nx, Ny)

        # apply boundary conditions 
        velocities = apply_boundary_conditions(x_positions, y_positions, velocities, cylinder, Nx, Ny)
        grid_density, ux, uy = compute_macroscopic_variables(x_positions, y_positions, velocities, Nx, Ny)

        # plot vorticity
        if plotRealTime:
            plot_vorticity(ux, uy, cylinder, Nx, Ny, it, Nt)
            
    plt.savefig('lagrangian_simulation.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
