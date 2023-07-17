import numpy as np
from numpy import sqrt


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


n_zeroes = 20
n_ones = 20
a = np.ones(37)
g = np.zeros(n_zeroes)
g[1] = 1
g[18] = 1
alpha_init = 0.05
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
    fNew = model3a(u, a)

    if cnt % nReset == 0:
        hNew = np.multiply(-1, fNew)
    else:
        v = np.multiply(max(0.0, beta(fNew, fOld, 2)), hOld)
        for i in range(len(fNew)):
            hNew[i] = -fNew[i] + v[i]

    # Backtracking line search
    alpha = alpha_init / r
    mx = 10 ** 100  # dummy value

    while mx > mNew + c * alpha * np.dot(hNew, fNew):
        alpha = r * alpha  # decrease step size by r
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
