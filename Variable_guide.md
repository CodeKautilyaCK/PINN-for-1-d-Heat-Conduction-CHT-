1. Domain and Geometry Variables
- L: Length of the rod / spatial domain (e.g., 1.0 meter)
- nx: Number of spatial points along the rod (e.g., 100)
- x: Spatial coordinates, typically generated using numpy.linspace(0, L, nx)

2. Material Properties
- alpha: Thermal diffusivity of the material (e.g., 0.01 m²/s)
- k (optional): Thermal conductivity (e.g., 1.0 W/m·K)
- rho (optional): Density (e.g., 1.0 kg/m³)
- Cp (optional): Specific heat (e.g., 1.0 J/kg·K)

3. Time Variables
- t0: Start time of the simulation (e.g., 0.0 s)
- t_end: End time of the simulation (e.g., 0.5 s)
- nt: Number of time points (e.g., 50)
- t: Time coordinates, typically generated using numpy.linspace(t0, t_end, nt)
- dt (optional): Time step size, dt = t_end / nt

4. Boundary and Initial Conditions
- T0: Temperature at the left boundary (x=0), e.g., 0.0 °C
- T1: Temperature at the right boundary (x=L), e.g., 1.0 °C
- T_init: Initial temperature along the rod, e.g., 0.0 °C

5. PINN / Neural Network Variables
- layers: Neural network architecture, e.g., [2, 50, 50, 1]
- activation: Activation function for hidden layers, e.g., "tanh"
- x_tensor: PyTorch tensor for spatial coordinates
- t_tensor: PyTorch tensor for time coordinates
- XT: Combined input (x, t) fed to the network
- T_pred: Predicted temperature from the network
- loss_pde: PDE residual loss (∂T/∂t - α ∂²T/∂x²)
- loss_bc: Boundary condition loss
- loss_ic: Initial condition loss
- loss_total: Total training loss = loss_pde + loss_bc + loss_ic
- optimizer: Optimizer used for training (e.g., Adam)
- epochs: Number of training iterations (e.g., 5000–10000)
- T_exact: Analytical solution for validation

6. Output / Visualization Variables
- T_plot: Temperature data used for plotting
- fig: Figure object for plotting
- ax: Axes object for multiple plots
- error: Difference between predicted and analytical solution
- heatmap: 2D visualization of error over x and t

7. Misc / Utility Variables
- device: Compute device for PyTorch ("cuda" or "cpu")
- seed: Random seed for reproducibility (e.g., 42)
- collocation_points: Randomly sampled points for training, e.g., numpy.random.rand(N,2)
