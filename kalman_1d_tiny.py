# Even simpler fixed-gain Kalman filter

velocity = 1.
dt = 1.

# Initial state x
pos = 0.

zs = [1.354, 1.882, 4.341, 7.156, 6.939, 6.844, 9.847, 12.553, 16.273, 14.800]

print "initial x %f" % (pos)

for z in zs:
    # predict
    pos = pos + velocity * dt
    print "prior x %f" % (pos)

    # update
    K = 0.5 # from experiments
    pos = pos + K * (z - pos)
    print "z %f, posterior x %f" % (z, pos)
