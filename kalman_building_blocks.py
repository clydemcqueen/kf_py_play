# coding=utf-8
from collections import namedtuple


gaussian = namedtuple('Gaussian', ['mean', 'var'])


def gaussian_repr(self):
    return '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(self[0], self[1])


gaussian.__repr__ = gaussian_repr


def predict(pos, movement):
    return gaussian(pos.mean + movement.mean, pos.var + movement.var)


pos = gaussian(10., .2**2)
move = gaussian(15., .7**2)
prior = predict(pos, move)


def gaussian_multiply(g1, g2):
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return gaussian(mean, variance)


def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior


posterior = update(pos, move)

print prior
print posterior