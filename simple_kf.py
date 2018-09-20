from numpy import dot
from scipy.linalg import inv

# Simple multi-dimensional Kalman filter


def predict(x, P, F, Q):
    x = dot(F, x)
    P = dot(F, P).dot(F.T) + Q
    return x, P


def update(x, P, z, R, H):
    S = dot(H, P).dot(H.T) + R
    K = dot(P, H.T).dot(inv(S))
    y = z - dot(H, x)
    x = x + dot(K, y)
    P = P - dot(K, H).dot(P) # Should be P = (I - KH)P, but might be unstable?
    return x, P
