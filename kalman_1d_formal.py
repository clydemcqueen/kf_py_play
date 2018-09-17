# coding=utf-8
from collections import namedtuple

# Rewrite kalman_1d.py to be a bit more formal

# === Gaussian ===


def gaussian_repr(self):
    return '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(self[0], self[1])


gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = gaussian_repr


# === Kalman filter ===


def predict(posterior, movement):
    x, P = posterior    # mean and variance of posterior
    dx, Q = movement    # mean and variance of movement

    x = x + dx
    P = P + Q

    print "prior x (%f, %f)" % (x, P)

    return gaussian(x, P)


def update(prior, measurement):
    x, P = prior        # mean and variance of prior
    z, R = measurement  # mean and variance of measurement

    y = z - x           # residual
    K = P / (P + R)     # Kalman gain

    x = x + K * y       # posterior
    P = (1 - K) * P     # posterior variance

    print "z %f, K %f, posterior x (%f, %f)" % (z, K, x, P)

    return gaussian(x, P)


dt = 1.
sensor_var = 2.

# Movement, or process
movement = gaussian(1., 1.)

# Initial posterior
posterior = gaussian(0., 20.**2)

zs = [1.354, 1.882, 4.341, 7.156, 6.939, 6.844, 9.847, 12.553, 16.273, 14.800]

for z in zs:
    prior = predict(posterior, movement)
    posterior = update(prior, gaussian(z, sensor_var))
