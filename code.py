import torch
import torch.nn as nn
import numpy as np

# ----------------------------
# Physics-Informed Neural Network (PINN) for 1D heat eq
# ----------------------------
alpha = 0.01  # thermal diffusivity
T_final = 0.5
device = "cuda" if torch.cuda.is_available() else "cpu"

# Simple fully-connected network
class PINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)])
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = torch.tanh(layer(x))
        return self.layers[-1](x)

# ----------------------------
# Derivative helper
# ----------------------------
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)[0]
    else:
        return gradients(gradients(u, x), x, order=order-1)

# ----------------------------
# Generate training data (IC, BC, PDE points)
# ----------------------------
N_f, N_u0, N_b = 2000, 200, 200
x_f = torch.rand(N_f,1, device=device, requires_grad=True)
t_f = torch.rand(N_f,1, device=device, requires_grad=True) * T_final
X_f = torch.cat([x_f, t_f], dim=1)

x_u0 = torch.rand(N_u0,1, device=device)
t_u0 = torch.zeros_like(x_u0)
X_u0 = torch.cat([x_u0, t_u0], dim=1)
u0 = torch.sin(np.pi * x_u0)

t_b = torch.rand(N_b//2,1, device=device) * T_final
X_b = torch.cat([torch.zeros_like(t_b), t_b, torch.ones_like(t_b), t_b], dim=1).reshape(-1,2)
u_b = torch.zeros_like(X_b[:,0:1])

# ----------------------------
# Training loop (Adam)
# ----------------------------
model = PINN([2,64,64,64,1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    optimizer.zero_grad()
    
    # IC loss
    u0_pred = model(X_u0)
    loss_u0 = loss_fn(u0_pred, u0)
    
    # BC loss
    u_b_pred = model(X_b)
    loss_b = loss_fn(u_b_pred, u_b)
    
    # PDE residual loss
    X_f.requires_grad_(True)
    u_f = model(X_f)
    u_t = gradients(u_f, X_f)[:,1:2]
    u_xx = gradients(gradients(u_f, X_f)[:,0:1], X_f)[:,0:1]
    loss_f = loss_fn(u_t - alpha*u_xx, torch.zeros_like(u_t))
    
    # Total loss
    loss = loss_u0 + loss_b + loss_f
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.3e}")
