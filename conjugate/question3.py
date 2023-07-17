from numpy import dot
import numpy as np


def fu(u, x, y):
    u1 = u[0]
    u2 = u[1]
    acc = 0
    for i in range(1, 5):
        m = (y[i - 1] - (u1 * x[i - 1] + u2)) ** 2
        acc += m
    return acc


def beta(fNew, fOld, variation: int, hOld=None) -> float:
    if variation == 1:
        return np.dot(fNew, fNew) / np.dot(fOld, fOld)
    if variation == 2:
        a = np.dot(fNew, (np.subtract(fNew, fOld)))
        b = np.dot(fOld, fOld)
        return a / b
    if variation == 3:
        return np.dot(fNew, (np.subtract(fNew, fOld))) / np.dot(hOld, (np.subtract(fNew, fOld)))
    if variation == 4:
        return np.dot(fNew, fNew) / np.dot(hOld, (np.subtract(fNew, fOld)))


a_init = 0.5
c = 0.5
r = 0.5
n = 25
u = [0.25, 0.25]
x = [1, 2, 3, 4]
y = [1, 1.5, 1.8, 2.2]
mnew = fu(u, x, y)
mold = 10 * 100
alpha = a_init
fnew = np.zeros(2)
cnt = 0
hnew = np.zeros(2)
while mold > mnew:
    mold = mnew
    fold = fnew
    hold = hnew
    u1 = u[0]
    u2 = u[1]
    dmdu1 = 0
    dmdu2 = 0
    for i in range(1, 5):
        dmdu1 += -2 * x[i - 1] * (-u1 * x[i - 1] - u2 + y[i - 1])
        dmdu2 += 2 * u1 * x[i - 1] + 2 * u2 - 2 * y[i - 1]

    fnew = [dmdu1, dmdu2]

    if cnt % n == 0:
        hnew = np.multiply(fnew, -1)
    else:
        v = np.multiply(max(0.0, beta(fnew, fold, 1)), hold)
        for i in range(len(fnew)):
            hnew[i] = -fnew[i] + v[i]

    alpha = a_init / r
    mx = 10 ** 100
    ux = np.zeros(2)
    while mx > mnew + c * alpha * dot(hnew, fnew):
        alpha = r * alpha
        for i in range(len(ux)):
            ux[i] = u[i] + alpha * hnew[i]
        mx = fu(ux, x, y)
    mnew = mx
    u = ux
    cnt += 1

m = mold
for i in range(len(u)):
    u[i] = u[i] - alpha * hnew[i]

print(m, u)
