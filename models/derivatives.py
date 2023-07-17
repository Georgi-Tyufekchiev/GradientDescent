import math

import numpy as np
from numpy import sqrt


def model1(u):
    dmdu1 = 2 * (u[0] - 1) * (sqrt((1 - u[0]) ** 2 + (u[1] + 1) ** 2) - sqrt(2)) / sqrt(
        (1 - u[0]) ** 2 + (u[1] + 1) ** 2) + 2 * (u[0] + 1) * \
            (sqrt((u[0] + 1) ** 2 + (u[1] + 1) ** 2) - sqrt(2)) / sqrt((u[0] + 1) ** 2 + (u[1] + 1) ** 2) - 1

    dmdu2 = 2 * (u[1] + 1) * (sqrt((u[0] + 1) ** 2 + (u[1] + 1) ** 2) - sqrt(2)) / sqrt(
        (u[0] + 1) ** 2 + (u[1] + 1) ** 2) + \
            2 * (u[1] + 1) * (sqrt((1 - u[0]) ** 2 + (u[1] + 1) ** 2) - sqrt(2)) / sqrt(
        (1 - u[0]) ** 2 + (u[1] + 1) ** 2) - 1
    return dmdu1, dmdu2


def model2(u, g):
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    u4 = u[3]
    g = np.multiply(-1, g)
    dmdu1 = 2 * u1 * (sqrt(u1 ** 2 + (u2 + 1) ** 2) - 1) / sqrt(u1 ** 2 + (u2 + 1) ** 2) + 2 * (u1 - 1) * (
            sqrt((1 - u1) ** 2 + (u2 + 1) ** 2) - sqrt(2)) / sqrt((1 - u1) ** 2 + (u2 + 1) ** 2) + 2 * (
                    sqrt((-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) - 1) * (u1 - u3 - 1) / sqrt(
        (-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2)

    dmdu2 = 2 * (u2 + 1) * (sqrt((1 - u1) ** 2 + (u2 + 1) ** 2) - sqrt(2)) / sqrt((1 - u1) ** 2 + (u2 + 1) ** 2) + 2 * (
            u2 - u4) * (sqrt((-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) - 1) / sqrt(
        (-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) + 2 * (u2 + 1) * (sqrt(u1 ** 2 + (u2 + 1) ** 2) - 1) / sqrt(
        u1 ** 2 + (u2 + 1) ** 2)
    dmdu3 = 2 * u3 * (sqrt(u3 ** 2 + (u4 + 1) ** 2) - 1) / sqrt(u3 ** 2 + (u4 + 1) ** 2) + 2 * (u3 + 1) * (
            sqrt((u3 + 1) ** 2 + (u4 + 1) ** 2) - sqrt(2)) / sqrt((u3 + 1) ** 2 + (u4 + 1) ** 2) + 2 * (
                    sqrt((-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) - 1) * (-u1 + u3 + 1) / sqrt(
        (-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2)
    dmdu4 = 2 * (-u2 + u4) * (sqrt((-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) - 1) / sqrt(
        (-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) + 2 * (u4 + 1) * (sqrt((u3 + 1) ** 2 + (u4 + 1) ** 2) - sqrt(2)) / sqrt(
        (u3 + 1) ** 2 + (u4 + 1) ** 2) + 2 * (u4 + 1) * (sqrt(u3 ** 2 + (u4 + 1) ** 2) - 1) / sqrt(
        u3 ** 2 + (u4 + 1) ** 2)
    g[0] += dmdu1
    g[1] += dmdu2
    g[2] += dmdu3
    g[3] += dmdu4

    return g


def model2b(u, g):
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    u4 = u[3]
    g = np.multiply(-1, g)
    dmdu1 = 2 * u1 * (sqrt(u1 ** 2 + (u2 + 1) ** 2) - 1) / sqrt(u1 ** 2 + (u2 + 1) ** 2) + 2 * (
                sqrt((-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) - 1) * (u1 - u3 - 1) / sqrt(
        (-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2)

    dmdu2 = 2 * (u2 - u4) * (sqrt((-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) - 1) / sqrt(
        (-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) + 2 * (u2 + 1) * (sqrt(u1 ** 2 + (u2 + 1) ** 2) - 1) / sqrt(
        u1 ** 2 + (u2 + 1) ** 2)

    dmdu3 = 2 * u3 * (sqrt(u3 ** 2 + (u4 + 1) ** 2) - 1) / sqrt(u3 ** 2 + (u4 + 1) ** 2) + 2 * (
                sqrt((-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) - 1) * (-u1 + u3 + 1) / sqrt(
        (-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2)
    dmdu4 = 2 * (-u2 + u4) * (sqrt((-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) - 1) / sqrt(
        (-u2 + u4) ** 2 + (-u1 + u3 + 1) ** 2) + 2 * (u4 + 1) * (sqrt(u3 ** 2 + (u4 + 1) ** 2) - 1) / sqrt(
        u3 ** 2 + (u4 + 1) ** 2)

    g[0] += dmdu1
    g[1] += dmdu2
    g[2] += dmdu3
    g[3] += dmdu4

    return g


def model3a(u, a):
    # term 1
    f = [0.0] * 20
    f[1] = -1
    f[18] = -1

    for i in range(1, 11):
        ai = a[i - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        dmdu1 = 2 * ai * u2im1 * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        dmdu2 = 2 * ai * (u2i + 1) * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        f[2 * i - 1 - 1] += dmdu1
        f[2 * i - 1] += dmdu2

    for i in range(1, 10):
        aip10 = a[i + 10 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        dmdu1 = -2 * aip10 * (u2im1 - u2ip1 - 1) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        dmdu2 = 2 * aip10 * (u2im1 - u2ip1 - 1) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        dmdu3 = -2 * aip10 * (u2i - u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        dmdu4 = 2 * aip10 * (u2i - u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i + 1 - 1] += dmdu1
        f[2 * i - 1 - 1] += dmdu2
        f[2 * i + 2 - 1] += dmdu3
        f[2 * i - 1] += dmdu4

    for i in range(1, 10):
        aip19 = a[i + 19 - 1]
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        dmdu1 = -2 * aip19 * (u2ip1 + 1) * (sqrt(2) - sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)
        dmdu2 = -2 * aip19 * (u2ip2 + 1) * (sqrt(2) - sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)
        f[2 * i + 1 - 1] += dmdu1
        f[2 * i + 2 - 1] += dmdu2

    for i in range(1, 10):
        aip28 = a[i + 28 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        dmdu1 = -2 * aip28 * (u2im1 - 1) * (sqrt(2) - sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2)) / sqrt(
            (u2i + 1) ** 2 + (u2im1 - 1) ** 2)
        dmdu2 = -2 * aip28 * (u2i + 1) * (sqrt(2) - sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2)) / sqrt(
            (u2i + 1) ** 2 + (u2im1 - 1) ** 2)
        f[2 * i - 1 - 1] += dmdu1
        f[2 * i - 1] += dmdu2

    return f


def model3b(u, a):
    # term 1
    f = [0.0] * 20
    f[1] = -1
    f[18] = -1

    for i in range(1, 11):
        ai = a[i - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        dmdu1 = 2 * ai * u2im1 * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        dmdu2 = 2 * ai * (u2i + 1) * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        f[2 * i - 1 - 1] += dmdu1
        f[2 * i - 1] += dmdu2

    for i in range(1, 10):
        aip10 = a[i + 10 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        dmdu1 = -2 * aip10 * (u2im1 - u2ip1 - 1) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        dmdu2 = 2 * aip10 * (u2im1 - u2ip1 - 1) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        dmdu3 = -2 * aip10 * (u2i - u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        dmdu4 = 2 * aip10 * (u2i - u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i + 1 - 1] += dmdu1
        f[2 * i - 1 - 1] += dmdu2
        f[2 * i + 2 - 1] += dmdu3
        f[2 * i - 1] += dmdu4

    return f


def model4a(u, a, g):
    f = g
    f = np.multiply(-1, f)

    for i in range(1, 11):
        ai = a[i - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        d1 = 2 * u2im1 * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        d2 = 2 * (u2i + 1) * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)

        f[2 * i - 1 - 1] += d1 * ai
        f[2 * i - 1] += d2 * ai

    for i in range(1, 10):
        aip10 = a[i + 10 - 1]
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        d3 = 2 * (u2ip1 + 1) * (sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) - sqrt(2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)
        d4 = 2 * (u2ip2 + 1) * (sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) - sqrt(2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)

        f[2 * i + 1 - 1] += d3 * aip10
        f[2 * i + 2 - 1] += d4 * aip10

    for i in range(1, 10):
        aip19 = a[i + 19 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        d5 = 2 * (u2im1 - 1) * (sqrt((1 - u2im1) ** 2 + (u2i + 1) ** 2) - sqrt(2)) / sqrt(
            (1 - u2im1) ** 2 + (u2i + 1) ** 2)
        d6 = 2 * (u2i + 1) * (sqrt((1 - u2im1) ** 2 + (u2i + 1) ** 2) - sqrt(2)) / sqrt(
            (1 - u2im1) ** 2 + (u2i + 1) ** 2)

        f[2 * i - 1 - 1] += d5 * aip19
        f[2 * i - 1] += d6 * aip19

    for i in range(1, 31):
        aip28 = a[i + 28 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        u2ip20 = u[2 * i + 20 - 1]
        u2ip19 = u[2 * i + 19 - 1]
        d7 = 2 * (sqrt((-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2) - 1) * (-u2i + u2ip20 + 1) / sqrt(
            (-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2)
        d8 = 2 * (sqrt((-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2) - 1) * (u2i - u2ip20 - 1) / sqrt(
            (-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2)
        d9 = 2 * (-u2im1 + u2ip19) * (sqrt((-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2) - 1) / sqrt(
            (-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2)
        d10 = 2 * (u2im1 - u2ip19) * (sqrt((-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2) - 1) / sqrt(
            (-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2)
        f[2 * i + 20 - 1] += d7 * aip28
        f[2 * i - 1] += d8 * aip28
        f[2 * i + 19 - 1] += d9 * aip28
        f[2 * i - 1 - 1] += d10 * aip28

    for i in range(1, 37):
        aip58 = a[i + 58 - 1]
        u2ip1p2 = u[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2im1p2 = u[2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2p2 = u[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2f = u[2 * i + 2 * math.floor((i - 1) / 9) - 1]
        d11 = 2 * (sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) * (
                -u2im1p2 + u2ip1p2 + 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        d12 = 2 * (sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) * (u2im1p2 - u2ip1p2 - 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        d13 = 2 * (-u2ip2f + u2ip2p2) * (sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        d14 = 2 * (u2ip2f - u2ip2p2) * (sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)

        f[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1] += d11 * aip58
        f[2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1] += d12 * aip58
        f[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1] += d13 * aip58
        f[2 * i + 2 * math.floor((i - 1) / 9) - 1] += d14 * aip58

    for i in range(1, 28):
        aip94 = a[i + 94 - 1]
        u2ip21p2 = u[2 * i + 21 + 2 * math.floor((i - 1) / 9) - 1]
        u2im1p2 = u[2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip22p2 = u[2 * i + 22 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2f = u[2 * i + 2 * math.floor((i - 1) / 9) - 1]
        d15 = 2 * (sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - sqrt(2)) * (
                -u2im1p2 + u2ip21p2 + 1) / sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)
        d16 = 2 * (sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - sqrt(2)) * (
                u2im1p2 - u2ip21p2 - 1) / sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)
        d17 = 2 * (sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - sqrt(2)) * (
                u2ip22p2 - u2ip2f + 1) / sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)
        d18 = 2 * (sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - sqrt(2)) * (
                -u2ip22p2 + u2ip2f - 1) / sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)

        f[2 * i + 21 + 2 * math.floor((i - 1) / 9) - 1] += d15 * aip94
        f[2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1] += d16 * aip94
        f[2 * i + 22 + 2 * math.floor((i - 1) / 9) - 1] += d17 * aip94
        f[2 * i + 2 * math.floor((i - 1) / 9) - 1] += d18 * aip94

    for i in range(1, 28):
        aip121 = a[i + 121 - 1]
        u2ip19p2 = u[2 * i + 19 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip1p2 = u[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip20p2 = u[2 * i + 20 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2p2 = u[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1]
        d20 = 2 * (sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - sqrt(2)) * (
                u2ip19p2 - u2ip1p2 - 1) / sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)
        d21 = 2 * (sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - sqrt(2)) * (
                -u2ip19p2 + u2ip1p2 + 1) / sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)
        d22 = 2 * (sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - sqrt(2)) * (
                u2ip20p2 - u2ip2p2 + 1) / sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)
        d23 = 2 * (sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - sqrt(2)) * (
                -u2ip20p2 + u2ip2p2 - 1) / sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        f[2 * i + 19 + 2 * math.floor((i - 1) / 9) - 1] += d20 * aip121
        f[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1] += d21 * aip121
        f[2 * i + 20 + 2 * math.floor((i - 1) / 9) - 1] += d22 * aip121
        f[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1] += d23 * aip121
    # print(f)
    return f

def model4b(u, a, g):
    f = g
    f = np.multiply(-1, f)

    for i in range(1, 11):
        ai = a[i - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        d1 = 2 * u2im1 * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        d2 = 2 * (u2i + 1) * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)

        f[2 * i - 1 - 1] += d1 * ai
        f[2 * i - 1] += d2 * ai

    for i in range(1, 31):
        aip28 = a[i + 28 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        u2ip20 = u[2 * i + 20 - 1]
        u2ip19 = u[2 * i + 19 - 1]
        d7 = 2 * (sqrt((-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2) - 1) * (-u2i + u2ip20 + 1) / sqrt(
            (-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2)
        d8 = 2 * (sqrt((-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2) - 1) * (u2i - u2ip20 - 1) / sqrt(
            (-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2)
        d9 = 2 * (-u2im1 + u2ip19) * (sqrt((-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2) - 1) / sqrt(
            (-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2)
        d10 = 2 * (u2im1 - u2ip19) * (sqrt((-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2) - 1) / sqrt(
            (-u2im1 + u2ip19) ** 2 + (-u2i + u2ip20 + 1) ** 2)
        f[2 * i + 20 - 1] += d7 * aip28
        f[2 * i - 1] += d8 * aip28
        f[2 * i + 19 - 1] += d9 * aip28
        f[2 * i - 1 - 1] += d10 * aip28

    for i in range(1, 37):
        aip58 = a[i + 58 - 1]
        u2ip1p2 = u[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2im1p2 = u[2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2p2 = u[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2f = u[2 * i + 2 * math.floor((i - 1) / 9) - 1]
        d11 = 2 * (sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) * (
                -u2im1p2 + u2ip1p2 + 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        d12 = 2 * (sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) * (u2im1p2 - u2ip1p2 - 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        d13 = 2 * (-u2ip2f + u2ip2p2) * (sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        d14 = 2 * (u2ip2f - u2ip2p2) * (sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)

        f[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1] += d11 * aip58
        f[2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1] += d12 * aip58
        f[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1] += d13 * aip58
        f[2 * i + 2 * math.floor((i - 1) / 9) - 1] += d14 * aip58

    return f