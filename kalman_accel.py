# coding=utf-8
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from filterpy.common import Q_discrete_white_noise
import simple_kf


# Track a vehicle in 1D, with 3 variables:
# -- position (hidden)
# -- velocity (hidden)
# -- acceleration (measured)
# Results: doesn't find track, position variance doesn't converge
kf = simple_kf.KalmanFilter(state_dim=3, measurement_dim=1)


def simulate_vehicle(z_var, process_var, count, dt):
    """
    Simulate a vehicle
    :param z_var: measurement (acceleration) variance
    :param process_var: process (acceleration) variance
    :param count: number of time steps to simulate
    :param dt: duration of each time step
    :return: a bunch of arrays
    """
    t, x_track = 0., 0.
    v_track = 0.
    a_ref, a_track = 0., 0.

    ts, xs_track, zs, vs_track, as_track = [], [], [], [], []

    z_std = math.sqrt(z_var)
    p_std = math.sqrt(process_var)

    for _ in range(count):
        t += dt
        ts.append(t)

        # Track shows the actual track of the vehicle. Add process noise to acceleration.
        a_track = a_ref + (randn() * p_std)
        v_track += a_track * dt
        x_track += v_track * dt

        as_track.append(a_track)
        vs_track.append(v_track)
        xs_track.append(x_track)

        # Generate a measurement
        zs.append(a_track + randn() * z_std)

    return np.array(ts), np.array(xs_track), np.array(vs_track), np.array(as_track), np.array(zs)


dt = 1.

# State mean, written formally as a 2x1 matrix
kf.x = np.array([[10.0], [5.0], [0.1]])
print 'x\n', kf.x

# State covariance
kf.P = np.diag([1., 1., 1.])
print 'P\n', kf.P

# Transition matrix
kf.F = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, 1]])
print 'F\n', kf.F

# Process noise
Q_var = 0.0001
kf.Q = Q_discrete_white_noise(dim=3, dt=dt, var=Q_var)
print 'Q\n', kf.Q

# Measurement matrix, translates state into measurement space
kf.H = np.array([[0., 0., 1.]])
print 'H\n', kf.H

# Measurement noise
R_var = 0.003 * 0.003
kf.R = np.array([[R_var]])
print 'R\n', kf.R

ts, xs_track, vs_track, as_track, zs = simulate_vehicle(R_var, Q_var, 50, dt)

prior_positions, posterior_positions = [], []
prior_velocities, posterior_velocities = [], []
prior_accelerations, posterior_accelerations = [], []
pc00, pc11, pc22 = [], [], []

for z in zs:
    x, P = kf.predict()
    prior_positions.append(x[0])
    prior_velocities.append(x[1])
    prior_accelerations.append(x[2])
    x, P = kf.update(np.array([[z]]))
    posterior_positions.append(x[0])
    posterior_velocities.append(x[1])
    posterior_accelerations.append(x[2])
    pc00.append(P[0, 0])
    pc11.append(P[1, 1])
    pc22.append(P[2, 2])

print 'Final P\n', kf.P

plt.figure(1)

plt.subplot(411)
plt.title('Position')
plt.plot(ts, xs_track, linestyle='dotted', label='track')
plt.plot(ts, prior_positions, label='prior')
plt.plot(ts, posterior_positions, label='posterior')
plt.grid()
plt.legend()

plt.subplot(412)
plt.title('Velocity')
plt.plot(ts, vs_track, linestyle='dotted', label='track')
plt.plot(ts, prior_velocities, label='prior')
plt.plot(ts, posterior_velocities, label='posterior')
plt.grid()
plt.legend()

plt.subplot(413)
plt.title('Acceleration')
plt.plot(ts, as_track, linestyle='dotted', label='track')
plt.plot(ts, prior_accelerations, label='prior')
plt.plot(ts, posterior_accelerations, label='posterior')
plt.scatter(ts, zs, marker='o', label='z')
plt.grid()
plt.legend()

plt.subplot(414)
plt.title('Variance')
plt.plot(ts, pc00, label='position')
plt.plot(ts, pc11, label='velocity')
plt.plot(ts, pc22, label='acceleration')
plt.grid()
plt.legend()

plt.show()
