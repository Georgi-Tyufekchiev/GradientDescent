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
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        m1 = aip10 * (sqrt((1 + u2ip1 - u2im1) ** 2 + (u2ip2 - u2i) ** 2) - 1) ** 2
        acc += m1

    return acc

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


n = 20
a = np.ones(19)
g = np.zeros(n)
g[1] = 1
g[18] = 1
nvar = 20
a_init = 0.5
r = 0.5
c1 = 1e-4
c2 = 0.9
u = np.zeros(n)
L = np.identity(nvar)
nli = 5
deltaF = np.ndarray(shape=(nvar, nli))
deltaU = np.ndarray(shape=(nvar, nli))
tol = 1e-12
mnew = fu(u, a, g)
f2 = model3b(u, a)
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
        h = np.dot(fnew,-1)

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

        h = -1 * h

    signal1 = 0
    alpha3 = a_init
    # print(alpha3)
    # print(h)
    ux = np.add(u, h * alpha3)
    m3 = fu(ux, a, g)
    f3 = model3b(ux, a)
    if m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and np.dot(h, f3) >= c2 * np.dot(h, fnew):
        signal1 = 1

    while m3 < mnew + c1 * alpha3 * np.dot(h, fnew) and signal1 == 0:
        alpha3 = alpha3 / r
        ux = np.add(u, h * alpha3)
        m3 = fu(ux, a, g)
        f3 = model3b(ux, a)
        if m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and np.dot(h, f3) >= c2 * np.dot(h, fnew):
            signal1 = 1
            break

    if signal1 == 0:
        signal2 = 0
        alpha1 = 0
        m1 = mnew
        f1 = fnew
        alpha2 = alpha3 / 2
        ux = np.add(u, h * alpha2)
        m2 = fu(ux, a, g)
        f2 = model3b(ux, a)
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
                ux = np.add(u, h * alpha2)
                m2 = fu(ux, a, g)
                f2 = model3b(ux, a)
            elif np.dot(h, f2) < c2 * np.dot(h, fnew):
                alpha1 = alpha2
                m1 = m2
                f1 = f2
                alpha2 = (alpha2 + alpha3) / 2
                ux = np.add(u, h * alpha2)
                m2 = fu(ux, a, g)
                f2 = model3b(ux, a)
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

