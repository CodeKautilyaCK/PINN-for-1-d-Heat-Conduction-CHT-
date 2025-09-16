# PINN-for-1-d-Heat-Conduction-CHT-
This project demonstrates a Physics-Informed Neural Network (PINN) solving the 1-D transient heat conduction equation. Unlike traditional CFD solvers that require meshes and time-stepping, this approach learns a continuous temperature field directly by embedding the governing PDE and boundary/initial conditions into the network’s loss function.

## Features
- **Dynamic, mesh-free solution**: Predicts temperature T(x,t) continuously across space and time.  
- **Physics-guided learning**: Loss function enforces PDE residuals and boundary/initial conditions.  
- **Validation**: Analytical solution comparison shows <3% relative L² error.  
- **Visualization**: Line plots of temperature over time and a heatmap of prediction error.  
- **Implementation**: Python with PyTorch, fully reproducible and lightweight.

## Problem Setup
- **Equation**: 1-D transient heat conduction, ∂T/∂t = α ∂²T/∂x²  
- **Domain**: Rod of length L with fixed temperatures at ends  
- **Initial Condition**: Uniform temperature  
- **Network Input**: (x, t)  
- **Loss**: PDE residual + boundary + initial condition errors

## Usage
1. Install dependencies:
```bash

