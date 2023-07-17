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


n = 20
a = np.ones(47)
g = np.zeros(n)
g[1] = 1
g[18] = 1
nvar = 20
a_init = 0.25
r = 0.5
tol = 1e-12
c1 = 1e-4
c2 = 0.9
u = np.zeros(n)
Lnew = np.identity(nvar)

mnew = fu(u, a, g)
f2 = model3a(u, a)
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
        # k = np.dot(delta_u, deltaF)
        # b = np.dot(np.dot(deltaF, Lold), deltaF)
        # c = np.dot(delta_u, delta_u)
        # d = np.dot(delta_u, deltaF)
        # first_fraction = np.divide(np.multiply(np.add(k, b), c), d ** 2)
        #
        # p = np.dot(np.dot(Lold, deltaF), delta_u)
        # t = np.dot(np.dot(delta_u, deltaF), Lold)
        # second_fraction = np.divide(np.add(p, t), d)
        # Lnew = np.subtract(np.add(Lold, first_fraction), second_fraction)
        # h = np.dot(np.dot(Lnew, -1), fnew)

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
    f3 = model3a(ux, a)
    if m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and np.dot(h, f3) >= c2 * np.dot(h, fnew):
        signal1 = 1

    while m3 <= mnew + c1 * alpha3 * np.dot(h, fnew) and signal1 == 0:
        alpha3 = alpha3 / r
        ux = np.add(u, np.dot(alpha3, h))
        m3 = fu(ux, a, g)
        f3 = model3a(ux, a)
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
        f2 = model3a(ux, a)
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
                f2 = model3a(ux, a)
            elif np.dot(h, f2) < c2 * np.dot(h, fnew):
                alpha1 = alpha2
                m1 = m2
                f1 = f2
                alpha2 = (alpha2 + alpha3) / 2
                ux = np.add(u, np.dot(alpha2, h))
                m2 = fu(ux, a, g)
                f2 = model3a(ux, a)
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
