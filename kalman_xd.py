# coding=utf-8
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from filterpy.common import Q_discrete_white_noise
import simple_kf


# Track a vehicle in 1D, with 2 variables:
# -- position (measured)
# -- velocity (hidden)
# Results: finds track
kf = simple_kf.KalmanFilter(state_dim=2, measurement_dim=1)


def simulate_vehicle(z_var, process_var, count, dt):
    """
    Simulate a vehicle
    :param z_var: measurement (position) variance
    :param process_var: process (velocity) variance
    :param count: number of time steps to simulate
    :param dt: duration of each time step
    :return: a bunch of arrays
    """
    t, x_track, v_ref, v_track = 0., 0., 1., 1.
    ts, xs_track, zs, vs_track = [], [], [], []

    z_std = math.sqrt(z_var)
    p_std = math.sqrt(process_var)

    for _ in range(count):
        t += dt
        ts.append(t)

        # Track shows the actual track of the vehicle. Add process noise to velocity.
        v_track = v_ref + (randn() * p_std)
        x_track += v_track * dt
        vs_track.append(v_track)
        xs_track.append(x_track)

        # Generate a measurement
        zs.append(x_track + randn() * z_std)

    return np.array(ts), np.array(xs_track), np.array(zs), np.array(vs_track)


dt = 1.

# State mean, written formally as a 2x1 matrix, initialized with pretty wild data
kf.x = np.array([[10.0], [4.5]])
print 'x\n', kf.x

# State covariance
kf.P = np.diag([500., 49.])
print 'P\n', kf.P

# Transition matrix
kf.F = np.array([[1, dt], [0, 1]])
print 'F\n', kf.F

# Process noise
Q_var = 0.01
kf.Q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_var)
print 'Q\n', kf.Q

# Measurement matrix, translates state into measurement space
kf.H = np.array([[1., 0.]])
print 'H\n', kf.H

# Measurement noise
R_var = 5.
kf.R = np.array([[R_var]])
print 'R\n', kf.R

ts, xs_track, zs, vs_track = simulate_vehicle(R_var, Q_var, 20, dt)

prior_positions, posterior_positions = [], []
prior_velocities, posterior_velocities = [], []
pc00, pc11 = [], []

for z in zs:
    x, P = kf.predict()
    prior_positions.append(x[0])
    prior_velocities.append(x[1])
    x, P = kf.update(np.array([[z]]))
    posterior_positions.append(x[0])
    posterior_velocities.append(x[1])
    pc00.append(P[0, 0])
    pc11.append(P[1, 1])

plt.figure(1)

plt.subplot(311)
plt.plot(ts, xs_track, linestyle='dotted', label='track')
plt.scatter(ts, zs, marker='o', label='z')
plt.plot(ts, prior_positions, label='prior')
plt.plot(ts, posterior_positions, label='posterior')
plt.grid()
plt.legend()

plt.subplot(312)
plt.plot(ts, vs_track, linestyle='dotted', label='track')
plt.plot(ts, prior_velocities, label='prior')
plt.plot(ts, posterior_velocities, label='posterior')
plt.grid()
plt.legend()

plt.subplot(313)
plt.plot(ts, pc00, label='post cov pos')
plt.plot(ts, pc11, label='post cov vel')
plt.grid()
plt.legend()

plt.show()
