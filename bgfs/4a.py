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


n = 80
a = np.ones(148)
g = np.zeros(n)
g[61] = 1
g[78] = 1
nvar = 80
a_init = 0.5
r = 0.5
tol = 1e-12
c1 = 1e-4
c2 = 0.9
u = np.zeros(n)
Lnew = np.identity(nvar)

mnew = fu(u, a, g)
f2 = model4a(u, a, g)
mold = 10 ** 100
fnew = np.zeros(n)
cnt = 0
h = np.zeros(n)
delta_u = np.zeros(n)
while mnew < mold:
    mold = mnew
    fold = fnew
    Lold = Lnew
    fnew = f2
    deltaF = np.subtract(fnew, fold)

    if cnt == 0:
        h = np.multiply(-1, fnew)
    else:
        b = np.dot(delta_u, deltaF)
        c = np.dot(np.dot(deltaF.T, Lold), deltaF)
        d = np.dot(delta_u, deltaF)
        fraction_1 = ((b + c) / d ** 2) * (np.outer(delta_u, delta_u))

        k = np.outer(Lold.dot(deltaF), delta_u)
        t = np.outer(delta_u, deltaF).dot(Lold)
        fraction_2 = (1 / d) * (k + t)
        L_new = Lold + fraction_1 - fraction_2

        h = -np.dot(L_new, fnew)
    signal1 = 0
    alpha3 = a_init
    ux = np.add(u, np.dot(alpha3, h))
    m3 = fu(ux, a, g)
    f3 = model4a(ux, a, g)
    if m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and np.dot(h, f3) >= c2 * np.dot(h, fnew):
        signal1 = 1

    while m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and signal1 == 0:
        alpha3 = alpha3 / r
        ux = np.add(u, np.dot(alpha3, h))
        m3 = fu(ux, a, g)
        f3 = model4a(ux, a, g)
        if m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and np.dot(h, f3) >= c2 * np.dot(h, fnew):
            signal1 = 1
            break

    if signal1 == 0:
        signal2 = 0
        alpha1 = 0
        m1 = mnew
        f1 = fnew
        alpha2 = alpha3 / 2
        ux = np.add(u, np.dot(alpha2, h))
        m2 = fu(ux, a, g)
        f2 = model4a(ux, a, g)
        while signal2 == 0:
            if alpha3 - alpha1 < tol:
                signal2 = 1
                m2 = mnew
                f2 = fnew
            elif m2 > mnew + c1 * alpha2 * np.dot(h, fnew):
                alpha3 = alpha2
                m3 = m2
                f3 = f2
                alpha2 = (alpha1 + alpha2) / 2
                ux = np.add(u, np.dot(alpha2, h))
                m2 = fu(ux, a, g)
                f2 = model4a(ux, a, g)
            elif np.dot(h, f2) < c2 * np.dot(h, fnew):
                alpha1 = alpha2
                m1 = m2
                f1 = f2
                alpha2 = (alpha2 + alpha3) / 2
                ux = np.add(u, np.dot(alpha2, h))
                m2 = fu(ux, a, g)
                f2 = model4a(ux, a, g)
            else:
                signal2 = 1

    delta_u = np.subtract(ux, u)
    u = ux
    cnt += 1
    if signal1 == 1:
        mnew = m3
        f2 = f3
    else:
        mnew = m2

m = mold
u = np.subtract(u, delta_u)
print()
print(m)
print(u)
