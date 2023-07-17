import math

import numpy as np
from numpy import sqrt


def fu(u, a, g):
    acc = 0
    gu = -g @ u
    acc += gu
    for i in range(1, 11):
        ai = a[i - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        m = ai * (sqrt(u2im1 ** 2 + (1 + u2i) ** 2) - 1) ** 2
        acc += m

    for i in range(1, 10):
        aip10 = a[i + 10 - 1]
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        m1 = aip10 * (sqrt((1 + u2ip1) ** 2 + (u2ip2 + 1) ** 2) - sqrt(2)) ** 2
        acc += m1

    for i in range(1, 10):
        aip19 = a[i + 19 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        m2 = aip19 * (sqrt((1 - u2im1) ** 2 + (u2i + 1) ** 2) - sqrt(2)) ** 2
        acc += m2

    for i in range(1, 31):
        aip28 = a[i + 28 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        u2ip20 = u[2 * i + 20 - 1]
        u2ip19 = u[2 * i + 19 - 1]
        m3 = aip28 * (sqrt((1 + u2ip20 - u2i) ** 2 + (u2ip19 - u2im1) ** 2) - 1) ** 2
        acc += m3

    for i in range(1, 37):
        aip58 = a[i + 58 - 1]
        u2ip1p2 = u[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2im1p2 = u[2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2p2 = u[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2f = u[2 * i + 2 * math.floor((i - 1) / 9) - 1]
        m4 = aip58 * (sqrt((1 + u2ip1p2 - u2im1p2) ** 2 + (u2ip2p2 - u2ip2f) ** 2) - 1) ** 2
        acc += m4
    for i in range(1, 28):
        aip94 = a[i + 94 - 1]
        u2ip21p2 = u[2 * i + 21 + 2 * math.floor((i - 1) / 9) - 1]
        u2im1p2 = u[2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip22p2 = u[2 * i + 22 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2f = u[2 * i + 2 * math.floor((i - 1) / 9) - 1]
        m5 = aip94 * (sqrt((1 + u2ip21p2 - u2im1p2) ** 2 + (1 + u2ip22p2 - u2ip2f) ** 2) - sqrt(2)) ** 2
        acc += m5

    for i in range(1, 28):
        aip121 = a[i + 121 - 1]
        u2ip19p2 = u[2 * i + 19 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip1p2 = u[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip20p2 = u[2 * i + 20 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2p2 = u[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1]
        m6 = aip121 * (sqrt((1 - u2ip19p2 + u2ip1p2) ** 2 + (1 + u2ip20p2 - u2ip2p2) ** 2) - sqrt(2)) ** 2
        acc += m6
    return acc


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


def hessian(u, f):
    for i in range(1, 11):
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        d1 = 2 * u2im1 ** 2 / (u2im1 ** 2 + (u2i + 1) ** 2) - 2 * u2im1 ** 2 * (
                sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / (u2im1 ** 2 + (u2i + 1) ** 2) ** (3 / 2) + 2 * (
                     sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        d2 = (-u2i - 1) * (2 * u2i + 2) * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / (u2im1 ** 2 + (u2i + 1) ** 2) ** (
                3 / 2) + (u2i + 1) * (2 * u2i + 2) / (u2im1 ** 2 + (u2i + 1) ** 2) + 2 * (
                     sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        f[2 * i - 1 - 1][2 * i - 1 - 1] += d1
        f[2 * i - 1][2 * i - 1] += d2
        d3 = u2im1 * (2 * u2i + 2) / (u2im1 ** 2 + (u2i + 1) ** 2) - u2im1 * (2 * u2i + 2) * (
                sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / (u2im1 ** 2 + (u2i + 1) ** 2) ** (3 / 2)
        f[2 * i - 1 - 1][2 * i - 1] += d3
        f[2 * i - 1][2 * i - 1 - 1] += d3
    for i in range(1, 10):
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        d1 = (-2 * u2ip1 - 2) * (-u2ip1 - 1) * (-sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / (
                (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) ** (3 / 2) - (-2 * u2ip1 - 2) * (u2ip1 + 1) / (
                     (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) - 2 * (
                     -sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)
        d2 = (-2 * u2ip2 - 2) * (-u2ip2 - 1) * (-sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / (
                (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) ** (3 / 2) - (-2 * u2ip2 - 2) * (u2ip2 + 1) / (
                     (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) - 2 * (
                     -sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)
        f[2 * i + 1 - 1][2 * i + 1 - 1] += d1
        f[2 * i + 2 - 1][2 * i + 2 - 1] += d2
        d3 = (-2 * u2ip1 - 2) * (-u2ip1 - 1) * (-sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / (
                (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) ** (3 / 2) - (-2 * u2ip1 - 2) * (u2ip1 + 1) / (
                     (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) - 2 * (
                     -sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)
        f[2 * i + 1 - 1][2 * i + 2 - 1] += d3
        f[2 * i + 2 - 1][2 * i + 1 - 1] += d3
    for i in range(1, 10):
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        d1 = (1 - u2im1) * (2 - 2 * u2im1) * (-sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / (
                (u2i + 1) ** 2 + (u2im1 - 1) ** 2) ** (3 / 2) - (2 - 2 * u2im1) * (u2im1 - 1) / (
                     (u2i + 1) ** 2 + (u2im1 - 1) ** 2) - 2 * (
                     -sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2)
        d2 = (-2 * u2i - 2) * (-u2i - 1) * (-sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / (
                (u2i + 1) ** 2 + (u2im1 - 1) ** 2) ** (3 / 2) - (-2 * u2i - 2) * (u2i + 1) / (
                     (u2i + 1) ** 2 + (u2im1 - 1) ** 2) - 2 * (
                     -sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2)
        f[2 * i - 1 - 1][2 * i - 1 - 1] += d1
        f[2 * i - 1][2 * i - 1] += d2
        d3 = (2 - 2 * u2im1) * (-u2i - 1) * (-sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / (
                (u2i + 1) ** 2 + (u2im1 - 1) ** 2) ** (3 / 2) - (2 - 2 * u2im1) * (u2i + 1) / (
                     (u2i + 1) ** 2 + (u2im1 - 1) ** 2)
        f[2 * i - 1 - 1][2 * i - 1] += d3
        f[2 * i - 1][2 * i - 1 - 1] += d3
    for i in range(1, 31):
        u2ip1 = u[2 * i + 20 - 1]
        u2im1 = u[2 * i - 1]
        u2ip2 = u[2 * i + 19 - 1]
        u2i = u[2 * i + -1 - 1]
        d1 = (-2 * u2im1 + 2 * u2ip1 + 2) * (-u2im1 + u2ip1 + 1) / (
                (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + 2 * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (-2 * u2im1 + 2 * u2ip1 + 2) * (
                     u2im1 - u2ip1 - 1) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        d2 = (u2im1 - u2ip1 - 1) * (2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + 2 * (
                sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (-u2im1 + u2ip1 + 1) * (
                     2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        d3 = (-2 * u2i + 2 * u2ip2) * (-u2i + u2ip2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                -2 * u2i + 2 * u2ip2) * (u2i - u2ip2) * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / (
                     (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + 2 * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        d4 = (-u2i + u2ip2) * (2 * u2i - 2 * u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / (
                (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + (u2i - u2ip2) * (
                     2 * u2i - 2 * u2ip2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + 2 * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i + 20 - 1][2 * i + 20 - 1] += d1
        f[2 * i + -1 - 1][2 * i + -1 - 1] += d2
        f[2 * i + 19 - 1][2 * i + 19 - 1] += d3
        f[2 * i + -1 - 1][2 * i + -1 - 1] += d4

        d5 = (-2 * u2im1 + 2 * u2ip1 + 2) * (u2im1 - u2ip1 - 1) / (
                (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 2 * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (-2 * u2im1 + 2 * u2ip1 + 2) * (
                     -u2im1 + u2ip1 + 1) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        f[2 * i + 20 - 1][2 * i + -1 - 1] += d5
        f[2 * i + -1 - 1][2 * i + 20 - 1] += d5
        d6 = (-u2i + u2ip2) * (-2 * u2im1 + 2 * u2ip1 + 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                u2i - u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (
                     -2 * u2im1 + 2 * u2ip1 + 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        f[2 * i + 20 - 1][2 * i + 19 - 1] += d6
        f[2 * i + 19 - 1][2 * i + 20 - 1] += d6
        d7 = (-u2i + u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (
                -2 * u2im1 + 2 * u2ip1 + 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + (
                     u2i - u2ip2) * (-2 * u2im1 + 2 * u2ip1 + 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i + 20 - 1][2 * i + -1 - 1] += d7
        f[2 * i + -1 - 1][2 * i + -1 - 1] += d7

        d8 = (-u2i + u2ip2) * (2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                u2i - u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (
                     2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        f[2 * i + -1 - 1][2 * i + 19 - 1] += d8
        f[2 * i + 19 - 1][2 * i + -1 - 1] += d8
        d9 = (-u2i + u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (
                2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + (
                     u2i - u2ip2) * (2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i + -1 - 1][2 * i + -1 - 1] += d9
        f[2 * i + -1 - 1][2 * i + -1 - 1] += d9

        d10 = (-2 * u2i + 2 * u2ip2) * (-u2i + u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / (
                (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + (-2 * u2i + 2 * u2ip2) * (
                      u2i - u2ip2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 2 * (
                      sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i + 19 - 1][2 * i + -1 - 1] += d10
        f[2 * i + -1 - 1][2 * i + 19 - 1] += d10
    for i in range(1, 37):
        first = 2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1
        second = 2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1
        third = 2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1
        fourth = 2 * i + 2 * math.floor((i - 1) / 9) - 1
        u2ip1p2 = u[first]
        u2im1p2 = u[second]
        u2ip2p2 = u[third]
        u2ip2f = u[fourth]
        d1 = 2 * (-u2im1p2 + u2ip1p2 + 1) ** 2 / ((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) + (
                2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) + (
                     2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) * (
                     -u2im1p2 + u2ip1p2 + 1) * (u2im1p2 - u2ip1p2 - 1) / (
                     (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (3 / 2)
        d2 = 2 * (u2im1p2 - u2ip1p2 - 1) ** 2 / ((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) + (
                2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) + (
                     2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) * (
                     -u2im1p2 + u2ip1p2 + 1) * (u2im1p2 - u2ip1p2 - 1) / (
                     (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (3 / 2)
        d3 = (-2 * u2ip2f + 2 * u2ip2p2) * (-u2ip2f + u2ip2p2) / (
                (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) + (-2 * u2ip2f + 2 * u2ip2p2) * (
                     u2ip2f - u2ip2p2) * (sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / (
                     (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        d4 = (-u2ip2f + u2ip2p2) * (2 * u2ip2f - 2 * u2ip2p2) * (
                sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / (
                     (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (3 / 2) + (u2ip2f - u2ip2p2) * (
                     2 * u2ip2f - 2 * u2ip2p2) / ((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) + 2 * (
                     sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        f[first][first] += d1
        f[second][second] += d2
        f[third][third] += d3
        f[fourth][fourth] += d4

        d5 = 2 * (-u2im1p2 + u2ip1p2 + 1) * (u2im1p2 - u2ip1p2 - 1) / (
                (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - (
                     2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) + (
                     2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) * (
                     -u2im1p2 + u2ip1p2 + 1) ** 2 / ((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (
                     3 / 2)
        f[first][second] += d5
        f[second][first] += d5

        d6 = 2 * (-u2ip2f + u2ip2p2) * (-u2im1p2 + u2ip1p2 + 1) / (
                (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) + (u2ip2f - u2ip2p2) * (
                     2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) * (
                     -u2im1p2 + u2ip1p2 + 1) / ((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (3 / 2)
        f[first][third] += d6
        f[third][first] += d6

        d7 = (-u2ip2f + u2ip2p2) * (2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) * (
                -u2im1p2 + u2ip1p2 + 1) / ((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (
                     3 / 2) + 2 * (u2ip2f - u2ip2p2) * (-u2im1p2 + u2ip1p2 + 1) / (
                     (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        f[first][fourth] += d7
        f[fourth][first] += d7

        d8 = 2 * (-u2ip2f + u2ip2p2) * (u2im1p2 - u2ip1p2 - 1) / (
                (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) + (u2ip2f - u2ip2p2) * (
                     2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) * (
                     u2im1p2 - u2ip1p2 - 1) / ((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (3 / 2)
        f[second][third] += d8
        f[third][second] += d8
        d9 = (-u2ip2f + u2ip2p2) * (2 * sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2) * (
                u2im1p2 - u2ip1p2 - 1) / ((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (
                     3 / 2) + 2 * (u2ip2f - u2ip2p2) * (u2im1p2 - u2ip1p2 - 1) / (
                     (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        f[second][fourth] += d9
        f[fourth][second] += d9

        d10 = (-2 * u2ip2f + 2 * u2ip2p2) * (-u2ip2f + u2ip2p2) * (
                sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / (
                      (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) ** (3 / 2) + (
                      -2 * u2ip2f + 2 * u2ip2p2) * (u2ip2f - u2ip2p2) / (
                      (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 2 * (
                      sqrt((-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2) - 1) / sqrt(
            (-u2ip2f + u2ip2p2) ** 2 + (-u2im1p2 + u2ip1p2 + 1) ** 2)
        f[third][fourth] += d10
        f[fourth][third] += d10

    for i in range(1, 28):
        first = 2 * i + 21 + 2 * math.floor((i - 1) / 9) - 1
        second = 2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1
        third = 2 * i + 22 + 2 * math.floor((i - 1) / 9) - 1
        fourth = 2 * i + 2 * math.floor((i - 1) / 9) - 1
        u2ip21p2 = u[first]
        u2im1p2 = u[second]
        u2ip22p2 = u[third]
        u2ip2f = u[fourth]

        d1 = (2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) + (
                     2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                     -u2im1p2 + u2ip21p2 + 1) * (u2im1p2 - u2ip21p2 - 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2im1p2 + u2ip21p2 + 1) ** 2 / ((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)
        d2 = (2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) + (
                     2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                     -u2im1p2 + u2ip21p2 + 1) * (u2im1p2 - u2ip21p2 - 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                     u2im1p2 - u2ip21p2 - 1) ** 2 / ((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)

        d3 = (2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) + (
                     2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                     -u2ip22p2 + u2ip2f - 1) * (u2ip22p2 - u2ip2f + 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                     u2ip22p2 - u2ip2f + 1) ** 2 / ((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)

        d4 = (2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) + (
                     2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                     -u2ip22p2 + u2ip2f - 1) * (u2ip22p2 - u2ip2f + 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2ip22p2 + u2ip2f - 1) ** 2 / ((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)

        f[first][first] += d1
        f[second][second] += d2
        f[third][third] += d3
        f[fourth][fourth] += d4

        d5 = -(2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) + (
                     2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                     -u2im1p2 + u2ip21p2 + 1) ** 2 / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2im1p2 + u2ip21p2 + 1) * (u2im1p2 - u2ip21p2 - 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)

        f[first][second] += d5
        f[second][first] += d5

        d6 = (2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                -u2im1p2 + u2ip21p2 + 1) * (-u2ip22p2 + u2ip2f - 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2im1p2 + u2ip21p2 + 1) * (u2ip22p2 - u2ip2f + 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)

        f[first][third] += d6
        f[third][first] += d6

        d7 = (2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                -u2im1p2 + u2ip21p2 + 1) * (u2ip22p2 - u2ip2f + 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2im1p2 + u2ip21p2 + 1) * (-u2ip22p2 + u2ip2f - 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)

        f[first][fourth] += d7
        f[fourth][first] += d7

        d8 = (2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                u2im1p2 - u2ip21p2 - 1) * (-u2ip22p2 + u2ip2f - 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                     u2im1p2 - u2ip21p2 - 1) * (u2ip22p2 - u2ip2f + 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)
        f[second][third] += d8
        f[third][second] += d8
        d9 = (2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                u2im1p2 - u2ip21p2 - 1) * (u2ip22p2 - u2ip2f + 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                     u2im1p2 - u2ip21p2 - 1) * (-u2ip22p2 + u2ip2f - 1) / (
                     (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)
        f[second][fourth] += d9
        f[fourth][second] += d9

        d10 = -(2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) + (
                      2 * sqrt((-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) - 2 * sqrt(2)) * (
                      u2ip22p2 - u2ip2f + 1) ** 2 / (
                      (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2) ** (3 / 2) + 2 * (
                      -u2ip22p2 + u2ip2f - 1) * (u2ip22p2 - u2ip2f + 1) / (
                      (-u2im1p2 + u2ip21p2 + 1) ** 2 + (u2ip22p2 - u2ip2f + 1) ** 2)

        f[third][fourth] += d10
        f[fourth][third] += d10

    for i in range(1, 28):
        first = 2 * i + 19 + 2 * math.floor((i - 1) / 9) - 1
        second = 2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1
        third = 2 * i + 20 + 2 * math.floor((i - 1) / 9) - 1
        fourth = 2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1
        u2ip19p2 = u[2 * i + 19 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip1p2 = u[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip20p2 = u[2 * i + 20 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2p2 = u[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1]

        d1 = (2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) + (
                     2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                     -u2ip19p2 + u2ip1p2 + 1) * (u2ip19p2 - u2ip1p2 - 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     u2ip19p2 - u2ip1p2 - 1) ** 2 / ((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        d2 = (2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) + (
                     2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                     -u2ip19p2 + u2ip1p2 + 1) * (u2ip19p2 - u2ip1p2 - 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2ip19p2 + u2ip1p2 + 1) ** 2 / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        d3 = (2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) + (
                     2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                     -u2ip20p2 + u2ip2p2 - 1) * (u2ip20p2 - u2ip2p2 + 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     u2ip20p2 - u2ip2p2 + 1) ** 2 / ((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        d4 = (2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) + (
                     2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                     -u2ip20p2 + u2ip2p2 - 1) * (u2ip20p2 - u2ip2p2 + 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2ip20p2 + u2ip2p2 - 1) ** 2 / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        f[first][first] += d1
        f[second][second] += d2
        f[third][third] += d3
        f[fourth][fourth] += d4

        d5 = -(2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) + (
                     2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                     u2ip19p2 - u2ip1p2 - 1) ** 2 / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2ip19p2 + u2ip1p2 + 1) * (u2ip19p2 - u2ip1p2 - 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        f[first][second] += d5
        f[second][first] += d5

        d6 = (2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                u2ip19p2 - u2ip1p2 - 1) * (-u2ip20p2 + u2ip2p2 - 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     u2ip19p2 - u2ip1p2 - 1) * (u2ip20p2 - u2ip2p2 + 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        f[first][third] += d6
        f[third][first] += d6

        d7 = (2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                u2ip19p2 - u2ip1p2 - 1) * (u2ip20p2 - u2ip2p2 + 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     u2ip19p2 - u2ip1p2 - 1) * (-u2ip20p2 + u2ip2p2 - 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        f[first][fourth] += d7
        f[fourth][first] += d7

        d8 = (2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                -u2ip19p2 + u2ip1p2 + 1) * (-u2ip20p2 + u2ip2p2 - 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2ip19p2 + u2ip1p2 + 1) * (u2ip20p2 - u2ip2p2 + 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        f[second][third] += d8
        f[third][second] += d8
        d9 = (2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                -u2ip19p2 + u2ip1p2 + 1) * (u2ip20p2 - u2ip2p2 + 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                     -u2ip19p2 + u2ip1p2 + 1) * (-u2ip20p2 + u2ip2p2 - 1) / (
                     (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        f[second][fourth] += d9
        f[fourth][second] += d9

        d10 = -(2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) / sqrt(
            (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) + (
                      2 * sqrt((-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) - 2 * sqrt(2)) * (
                      u2ip20p2 - u2ip2p2 + 1) ** 2 / (
                      (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2) ** (3 / 2) + 2 * (
                      -u2ip20p2 + u2ip2p2 - 1) * (u2ip20p2 - u2ip2p2 + 1) / (
                      (-u2ip19p2 + u2ip1p2 + 1) ** 2 + (u2ip20p2 - u2ip2p2 + 1) ** 2)

        f[third][fourth] += d10
        f[fourth][third] += d10

    return f


u = np.zeros(80)
alpha = 0
h = np.zeros(80)
variables = np.zeros(6400).reshape((80, 80))
a = np.ones(148)
tol = 1e-12
g = np.zeros(80)
g[52] = 1
g[54] = 1
f = model4a(u, a, g)

# variables[1] = np.subtract(f[1], 1)
# variables[18] = np.subtract(f[18], 1)
count = 0
l2 = np.sqrt(np.dot(np.transpose(f), f))
print("NOTE: THE CODE IS TAKING FOREVER TO FINISH WITH THIS TOL. AFTER 10 000 ITERATIONS THE PROGRAM STOPS. THE "
      "APPORXIMATIONS SEEM TO BE GOOD ENOUGH.")
while l2 > tol:
    if count == 1e4:
        break
    if count % 1000 == 0:
        print("l2 %.12f" % l2)

    K = hessian(u, variables)

    h = np.linalg.solve(K, np.multiply(-1, f))
    for i in range(len(u)):
        u[i] += h[i]

    f = model4a(u, a, g)
    l2 = np.sqrt(np.dot(np.transpose(f), f))
    count += 1

# print("u %.4f" % u[0])
# print("u %.4f" % u[1])
print(u)
print("mu %.4f" % fu(u, a, g))
