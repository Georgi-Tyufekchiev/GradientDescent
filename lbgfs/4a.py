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
g[52] = 1
g[54] = 1
nvar = 80
a_init = 0.5
r = 0.5
c1 = 1e-4
c2 = 0.9
u = np.zeros(n)
L = np.identity(nvar)
nli = 3
deltaF = np.ndarray(shape=(nvar, nli))
deltaU = np.ndarray(shape=(nvar, nli))
tol = 1e-12
mnew = fu(u, a, g)
f2 = model4a(u, a, g)
mold = 10 ** 100
cnt = 0
h = np.zeros(n)
fnew = np.zeros(n)
delta_u = np.zeros(n)
delta_f = np.zeros(n)
p = np.zeros(n)

while mnew < mold:
    mold = mnew
    fnew = f2
    cnt += 1

    if cnt == 1:
        h = np.dot(fnew, -1)

    else:
        gamma = np.zeros(nli)
        h = fnew
        for j in range(nli, max(0, nli - cnt + 1), -1):
            pj = p[j - 1]
            delta_u = deltaU[:, [j - 1]].reshape((len(h),))
            delta_f = deltaF[:, [j - 1]].reshape((len(h),))
            gammaj = pj * np.dot(delta_u, h)
            h = np.subtract(h, np.dot(gammaj, delta_f))
            gamma[j - 1] = gammaj

        h = np.dot(L, h)

        for j in range(max(0, nli - cnt + 2), nli + 1):
            pj = p[j - 1]
            delta_u = deltaU[:, [j - 1]].reshape((len(h),))
            delta_f = deltaF[:, [j - 1]].reshape((len(h),))
            gammaj = gamma[j - 1]
            ni = pj * np.dot(delta_f, h)
            h = np.add(h, np.dot(gammaj - ni, delta_u))

        h = np.multiply(-1, h)

    signal1 = 0
    alpha3 = a_init
    # print(alpha3)
    # print(h)
    ux = np.add(u, np.dot(h, alpha3))
    m3 = fu(ux, a, g)
    f3 = model4a(ux, a, g)
    if m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and np.dot(h, f3) >= c2 * np.dot(h, fnew):
        signal1 = 1

    while m3 < mnew + c1 * alpha3 * np.dot(h, fnew) and signal1 == 0:
        alpha3 = alpha3 / r
        ux = np.add(u, np.dot(h, alpha3))
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
        ux = np.add(u, np.dot(h, alpha2))

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
                ux = np.add(u, np.dot(h, alpha2))

                m2 = fu(ux, a, g)
                f2 = model4a(ux, a, g)

            elif np.dot(h, f2) < c2 * np.dot(h, fnew):
                alpha1 = alpha2
                m1 = m2
                f1 = f2
                alpha2 = (alpha2 + alpha3) / 2
                ux = np.add(u, np.dot(h, alpha2))

                m2 = fu(ux, a, g)
                f2 = model4a(ux, a, g)

            else:
                signal2 = 1

    delta_u = np.subtract(ux, u)
    u = ux
    if signal1 == 1:
        mnew = m3
        f2 = f3
        delta_f = np.subtract(f2, fnew)


    else:
        if signal2 == 1:
            mnew = m2
            delta_f = np.subtract(f2, fnew)

        else:
            mnew = mnew

    deltaU[:, [0, nli - 2]] = deltaU[:, [1, nli - 1]]
    deltaU[:, [nli - 1]] = delta_u.reshape(deltaU[:, [nli - 1]].shape)
    deltaF[:, [0, nli - 2]] = deltaF[:, [1, nli - 1]]

    deltaF[:, [nli - 1]] = delta_f.reshape(deltaF[:, [nli - 1]].shape)
    p[:nli - 2] = p[1:nli - 1]
    p[nli - 1] = 1 / np.dot(delta_f, delta_u)

m = mold
u = np.subtract(u, delta_u)
print(m)
print(u)
