import numpy as np
from numpy import sqrt


def fu(u, g):
    u1 = u[0]
    u2 = u[1]
    u3 = u[2]
    u4 = u[3]
    m1 = (sqrt(u1 ** 2 + (1 + u2) ** 2) - 1) ** 2
    m2 = (sqrt(u3 ** 2 + (1 + u4) ** 2) - 1) ** 2
    m3 = (sqrt((1 + u3 - u1) ** 2 + (u4 - u2) ** 2) - 1) ** 2
    g = np.transpose(np.multiply(-1, g))
    m6 = np.dot(g, u)
    return m1 + m2 + m3 + m6


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


n = 4
a = np.ones(n)
g = np.zeros(n)
g[1] = 1
g[2] = 1
nvar = 4
a_init = 0.5
r = 0.5
c1 = 1e-4
c2 = 0.9
u = np.zeros(n)
L = np.identity(nvar)
nli = 4
deltaF = np.array([np.zeros(nli), np.zeros(nvar), np.zeros(nvar), np.zeros(nvar)])
deltaU = np.array([np.zeros(nli), np.zeros(nvar), np.zeros(nvar), np.zeros(nvar)])
tol = 1e-12
mnew = fu(u, g)
f2 = model2b(u, g)
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
        h = fnew * -1
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
    ux = np.add(u, h * alpha3)
    m3 = fu(ux, g)
    f3 = model2b(ux, g)
    if m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and np.dot(h, f3) >= c2 * np.dot(h, fnew):
        signal1 = 1

    while m3 < mnew + c1 * alpha3 * np.dot(h, fnew) and signal1 == 0:
        alpha3 = alpha3 / r
        ux = np.add(u, h * alpha3)
        m3 = fu(ux, g)
        f3 = model2b(ux, g)
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
        m2 = fu(ux, g)
        f2 = model2b(ux, g)
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
                m2 = fu(ux, g)
                f2 = model2b(ux, g)
            elif np.dot(h, f2) < c2 * np.dot(h, fnew):
                alpha1 = alpha2
                m1 = m2
                f1 = f2
                alpha2 = (alpha2 + alpha3) / 2
                ux = np.add(u, h * alpha2)
                m2 = fu(ux, g)
                f2 = model2b(ux, g)
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
