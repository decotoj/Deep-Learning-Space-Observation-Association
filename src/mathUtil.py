import math


def unitVectorToAzEl(v):

    el = math.asin(v[2] / 1)

    x = v[0]
    y = v[1]
    if x > 0 and y > 0:
        az = math.atan(y / x)
    elif x < 0 and y > 0:
        az = 1.5708 + math.atan(abs(x) / y)
    elif x < 0 and y < 0:
        az = 1.5708 * 2 + math.atan(abs(y) / abs(x))
    else:
        az = 1.5708 * 3 + math.atan(x / abs(y))

    return az, el
