import math

import numpy as np
from numpy import sqrt


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


def computeFunc(u, a=None, g=None):
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


def beta(fNew: list, fOld: list, variation: int, hOld=None) -> float:
    if variation == 1:
        return np.dot(fNew, fNew) / np.dot(fOld, fOld)
    if variation == 2:
        a = np.dot(fNew, (np.subtract(fNew, fOld)))
        b = np.dot(fOld, fOld)
        return np.divide(a, b)
    if variation == 3:
        return np.dot(fNew, (np.subtract(fNew, fOld))) / np.dot(hOld, (np.subtract(fNew, fOld)))
    if variation == 4:
        return np.dot(fNew, fNew) / np.dot(hOld, (np.subtract(fNew, fOld)))


n_zeroes = 80
a = np.ones(148)
g = np.zeros(n_zeroes)
g[52] = 1
g[54] = 1
alpha_init = 0.5
r = 0.5
c = 0.5
u = np.zeros(n_zeroes)
hNew = np.zeros(n_zeroes)
nReset = 25

mNew = computeFunc(u, a, g)
mOld = 10 ** 100

fNew = np.zeros(n_zeroes)
cnt = 0
alpha = alpha_init
ux = np.zeros(n_zeroes)

while mOld > mNew:

    mOld = mNew
    fOld = fNew
    hOld = hNew
    fNew = model4a(u, a, g)

    if cnt % nReset == 0:
        hNew = np.multiply(-1, fNew)
    else:
        v = np.multiply(max(0.0, beta(fNew, fOld, 3, hOld)), hOld)
        for i in range(len(fNew)):
            hNew[i] = -fNew[i] + v[i]

    # Backtracking line search
    alpha = alpha_init / r
    mx = 10 ** 100  # dummy value

    while mx > mNew + c * alpha * np.dot(hNew, fNew):
        alpha = r * alpha  # decrease step size by r
        if alpha == 0:
            #  With alpha = 0.5 it goes all the way to 0 and enters infinite, not sure where things go wrong
            break
        for i in range(len(hNew)):
            ux[i] = u[i] + alpha * hNew[i]
        mx = computeFunc(ux, a, g)
    mNew = mx
    u = ux
    cnt += 1

m = mOld
for i in range(len(hNew)):
    u[i] = u[i] - alpha * hNew[i]
print(m, u)
