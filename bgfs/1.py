import sys

import numpy as np
from numpy import sqrt


def fu(u):
    u1 = u[0]
    u2 = u[1]
    return (sqrt((1 + u1) ** 2 + (1 + u2) ** 2) - sqrt(2)) ** 2 + (
            sqrt((1 - u1) ** 2 + (1 + u2) ** 2) - sqrt(2)) ** 2 - u1 - u2


def model1(u):
    u1 = u[0]
    u2 = u[1]
    dmdu1 = 2 * (u1 - 1) * (sqrt((1 - u1) ** 2 + (u2 + 1) ** 2) - sqrt(2)) / sqrt(
        (1 - u1) ** 2 + (u2 + 1) ** 2) + 2 * (u1 + 1) * \
            (sqrt((u1 + 1) ** 2 + (u2 + 1) ** 2) - sqrt(2)) / sqrt((u1 + 1) ** 2 + (u2 + 1) ** 2) - 1

    dmdu2 = 2 * (u2 + 1) * (sqrt((u1 + 1) ** 2 + (u2 + 1) ** 2) - sqrt(2)) / sqrt(
        (u1 + 1) ** 2 + (u2 + 1) ** 2) + \
            2 * (u2 + 1) * (sqrt((1 - u1) ** 2 + (u2 + 1) ** 2) - sqrt(2)) / sqrt(
        (1 - u1) ** 2 + (u2 + 1) ** 2) - 1
    return [dmdu1, dmdu2]


n = 2
a = np.ones(n)
g = np.ones(n)
nvar = 2
a_init = 0.5
r = 0.5
tol = 1e-12
c1 = 1e-4
c2 = 0.9
u = np.zeros(n)
Lnew = np.identity(nvar)

mnew = fu(u)
f2 = model1(u)
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
    m3 = fu(ux)
    f3 = model1(ux)
    if m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and np.dot(h, f3) >= c2 * np.dot(h, fnew):
        signal1 = 1

    while m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and signal1 == 0:
        alpha3 = alpha3 / r
        ux = np.add(u, np.dot(alpha3, h))
        m3 = fu(ux)
        f3 = model1(ux)
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
        m2 = fu(ux)
        f2 = model1(ux)
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
                m2 = fu(ux)
                f2 = model1(ux)
            elif np.dot(h, f2) < c2 * np.dot(h, fnew):
                alpha1 = alpha2
                m1 = m2
                f1 = f2
                alpha2 = (alpha2 + alpha3) / 2
                ux = np.add(u, np.dot(alpha2, h))
                m2 = fu(ux)
                f2 = model1(ux)
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
print(m)
print(u)
