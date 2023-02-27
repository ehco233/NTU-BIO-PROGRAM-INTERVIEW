import numpy as np
import matplotlib.pyplot as plt

# Define the rate constants
k1 = 100  # 1/min/µM
k2 = 600  # 1/min
k3 = 150  # 1/min

# Define the initial concentrations
E0 = 1    # µM
S0 = 10   # µM
ES0 = 0
P0 = 0

# Define the time step and simulation duration
dt = 0.001   # min
t_max = 0.5   # min

# Define the initial conditions
y0 = np.array([E0, S0, ES0, P0])

# Define the function


def f(t, y):
    E, S, ES, P = y
    dEdt = -k1*E*S + k2*ES + k3*ES
    dSdt = -k1*E*S + k2*ES
    dESdt = k1*E*S - k2*ES - k3*ES
    dPdt = k3*ES
    return np.array([dEdt, dSdt, dESdt, dPdt])

# Define the function for the fourth-order Runge-Kutta method
# traditional classical runge-kutta


def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + k1*dt/2)
    k3 = f(t + dt/2, y + k2*dt/2)
    k4 = f(t + dt, y + k3*dt)
    return y + (k1 + 2*k2 + 2*k3 + k4)*dt/6


# Perform the simulation
t_vals = np.arange(0, t_max+dt, dt)
y_vals = np.zeros((len(t_vals), len(y0)))
y_vals[0, :] = y0
for i in range(len(t_vals)-1):
    y_vals[i+1, :] = rk4_step(f, t_vals[i], y_vals[i, :], dt)

# Plot the results
plt.figure('Runge Kutta numerical results')
plt.plot(t_vals, y_vals[:, 0], label='E')
plt.plot(t_vals, y_vals[:, 1], label='S')
plt.plot(t_vals, y_vals[:, 2], label='ES')
plt.plot(t_vals, y_vals[:, 3], label='P')
plt.xlabel('Time (min)')
plt.ylabel('Concentration (µM)')
plt.legend()
plt.show()
