import math

import numpy as np
from numpy import sqrt


def model1(u):
    u1 = u[0]
    u2 = u[1]
    return (sqrt((1 + u1) ** 2 + (1 + u2) ** 2) - sqrt(2)) ** 2 + (
            sqrt((1 - u1) ** 2 + (1 + u2) ** 2) - sqrt(2)) ** 2 - u1 - u2


def model2a(u, g):
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    u4 = u[3]
    m1 = (sqrt(u1 ** 2 + (1 + u2) ** 2) - 1) ** 2
    m2 = (sqrt(u3 ** 2 + (1 + u4) ** 2) - 1) ** 2
    m3 = (sqrt((1 + u3 - u1) ** 2 + (u4 - u2) ** 2) - 1) ** 2
    m4 = (sqrt((1 + u3) ** 2 + (u4 + 1) ** 2) - sqrt(2)) ** 2
    m5 = (sqrt((1 - u1) ** 2 + (1 + u2) ** 2) - sqrt(2)) ** 2
    g = np.transpose(np.multiply(-1, g))
    m6 = np.dot(g, u)

    return m1 + m2 + m3 + m4 + m5 + m6


def model2b(u, g):
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    u4 = u[3]
    m1 = (sqrt(u1 ** 2 + (1 + u2) ** 2) - 1) ** 2
    m2 = (sqrt(u3 ** 2 + (1 + u4) ** 2) - 1) ** 2
    m3 = (sqrt((1 + u3 - u1) ** 2 + (u4 - u2) ** 2) - 1) ** 2
    g = np.transpose(np.multiply(-1, g))
    m6 = np.dot(g,u)
    return m1+m2+m3 + m6


def model3a(u, a=None, g=None):
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
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        m1 = aip10 * (sqrt((1 + u2ip1 - u2im1) ** 2 + (u2ip2 - u2i) ** 2) - 1) ** 2
        acc += m1

    for i in range(1, 10):
        aip19 = a[i + 19 - 1]
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        m2 = aip19 * (sqrt((1 + u2ip1) ** 2 + (u2ip2 + 1) ** 2) - sqrt(2)) ** 2
        acc += m2

    for i in range(1, 10):
        aip28 = a[i + 28 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        m3 = aip28 * (sqrt((1 - u2im1) ** 2 + (1 + u2i) ** 2) - sqrt(2)) ** 2
        acc += m3
    # print("acc",acc)
    return acc


def model3b(u, g, a):
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
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        m1 = aip10 * (sqrt((1 + u2ip1 - u2im1) ** 2 + (u2ip2 - u2i) ** 2) - 1) ** 2
        acc += m1

    return acc


def model4a(u, a=None, g=None):
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


def model4b(a, g, u):
    acc = 0
    gu = -g @ u
    acc += gu

    for i in range(1, 11):
        ai = a[i - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        m = ai * (sqrt(u2im1 ** 2 + (1 + u2i) ** 2) - 1) ** 2
        acc += m

    for i in range(1, 31):
        aip28 = a[i + 28 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        u2ip20 = u[2 * i + 20 - 1]
        u2ip19 = u[2 * i + 19 - 1]
        m3 = aip28 * (sqrt((1 + u2ip20 - u2i) ** 2 + (u2ip19 - u2im1) ** 2) - 1) ** 2
        acc += m3

    for i in range(1, 37):
        aip40 = a[i + 40 - 1]
        u2ip1p2 = u[2 * i + 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2im1p2 = u[2 * i - 1 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2p2 = u[2 * i + 2 + 2 * math.floor((i - 1) / 9) - 1]
        u2ip2f = u[2 * i + 2 * math.floor((i - 1) / 9) - 1]
        m4 = aip40 * (sqrt((1 + u2ip1p2 - u2im1p2) ** 2 + (u2ip2p2 - u2ip2f) ** 2) - 1) ** 2
        acc += m4

    return acc