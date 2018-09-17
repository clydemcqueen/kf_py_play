# Super simple 1D continuous (Gaussian) Kalman filter in Python

velocity = 1.
dt = 1.
process_var = 1.
sensor_var = 2.

# Initial state x
pos = 0.
var = 20.**2

zs = [1.354, 1.882, 4.341, 7.156, 6.939, 6.844, 9.847, 12.553, 16.273, 14.800]

print "initial x (%f, %f)" % (pos, var)

for z in zs:
    # predict
    dx = velocity * dt
    pos = pos + dx
    var = var + process_var
    print "prior x (%f, %f)" % (pos, var)

    # update
    pos  = (var * z + sensor_var * pos) / (var + sensor_var)
    var = (var * sensor_var) / (var + sensor_var)
    print "z (%f), posterior x (%f, %f)" % (z, pos, var)
