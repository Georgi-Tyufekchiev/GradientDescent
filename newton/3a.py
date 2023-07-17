import numpy as np
from numpy import sqrt


def mu(u, a, g):
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
        u2ip1 = u[2 * i + 1 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        u2i = u[2 * i - 1]
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


def hessian(u, f):
    for i in range(1, 11):
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        d1 = 2 * u2im1 ** 2 / (u2im1 ** 2 + (u2i + 1) ** 2) - 2 * u2im1 ** 2 * (
                sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / (u2im1 ** 2 + (u2i + 1) ** 2) ** (3 / 2) + 2 * (
                     sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        d2 = (-u2i - 1) * (2 * u2i + 2) * (sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / (u2im1 ** 2 + (u2i + 1) ** 2) ** (
                3 / 2) + (u2i + 1) * (2 * u2i + 2) / (u2im1 ** 2 + (u2i + 1) ** 2) + 2 * (
                     sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / sqrt(u2im1 ** 2 + (u2i + 1) ** 2)
        f[2 * i - 1 - 1][2 * i - 1 - 1] += d1
        f[2 * i - 1][2 * i - 1] += d2
        d3 = u2im1 * (2 * u2i + 2) / (u2im1 ** 2 + (u2i + 1) ** 2) - u2im1 * (2 * u2i + 2) * (
                sqrt(u2im1 ** 2 + (u2i + 1) ** 2) - 1) / (u2im1 ** 2 + (u2i + 1) ** 2) ** (3 / 2)
        f[2 * i - 1 - 1][2 * i - 1] += d3
        f[2 * i - 1][2 * i - 1 - 1] += d3

    for i in range(1, 10):
        u2ip1 = u[2 * i + 1 - 1]
        u2im1 = u[2 * i - 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        u2i = u[2 * i - 1]
        d1 = (-2 * u2im1 + 2 * u2ip1 + 2) * (-u2im1 + u2ip1 + 1) / (
                (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + 2 * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (-2 * u2im1 + 2 * u2ip1 + 2) * (
                     u2im1 - u2ip1 - 1) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        d2 = (u2im1 - u2ip1 - 1) * (2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + 2 * (
                sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (-u2im1 + u2ip1 + 1) * (
                     2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        d3 = (-2 * u2i + 2 * u2ip2) * (-u2i + u2ip2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                -2 * u2i + 2 * u2ip2) * (u2i - u2ip2) * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / (
                     (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + 2 * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        d4 = (-u2i + u2ip2) * (2 * u2i - 2 * u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / (
                (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + (u2i - u2ip2) * (
                     2 * u2i - 2 * u2ip2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + 2 * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i + 1 - 1][2 * i + 1 - 1] += d1
        f[2 * i - 1 - 1][2 * i - 1 - 1] += d2
        f[2 * i + 2 - 1][2 * i + 2 - 1] += d3
        f[2 * i - 1][2 * i - 1] += d4

        d5 = (-2 * u2im1 + 2 * u2ip1 + 2) * (u2im1 - u2ip1 - 1) / (
                (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 2 * (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                     sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (-2 * u2im1 + 2 * u2ip1 + 2) * (
                     -u2im1 + u2ip1 + 1) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        f[2 * i + 1 - 1][2 * i - 1 - 1] += d5
        f[2 * i - 1 - 1][2 * i + 1 - 1] += d5
        d6 = (-u2i + u2ip2) * (-2 * u2im1 + 2 * u2ip1 + 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                u2i - u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (
                     -2 * u2im1 + 2 * u2ip1 + 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        f[2 * i + 1 - 1][2 * i + 2 - 1] += d6
        f[2 * i + 2 - 1][2 * i + 1 - 1] += d6
        d7 = (-u2i + u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (
                -2 * u2im1 + 2 * u2ip1 + 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + (
                     u2i - u2ip2) * (-2 * u2im1 + 2 * u2ip1 + 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i + 1 - 1][2 * i - 1] += d7
        f[2 * i - 1 - 1][2 * i - 1] += d7

        d8 = (-u2i + u2ip2) * (2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) + (
                u2i - u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (
                     2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2)
        f[2 * i - 1 - 1][2 * i + 2 - 1] += d8
        f[2 * i + 2 - 1][2 * i - 1 - 1] += d8
        d9 = (-u2i + u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) * (
                2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + (
                     u2i - u2ip2) * (2 * u2im1 - 2 * u2ip1 - 2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i - 1 - 1][2 * i - 1] += d9
        f[2 * i - 1][2 * i - 1 - 1] += d9

        d10 = (-2 * u2i + 2 * u2ip2) * (-u2i + u2ip2) * (sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / (
                (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) ** (3 / 2) + (-2 * u2i + 2 * u2ip2) * (
                      u2i - u2ip2) / ((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 2 * (
                      sqrt((u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2) - 1) / sqrt(
            (u2i - u2ip2) ** 2 + (u2im1 - u2ip1 - 1) ** 2)
        f[2 * i + 2 - 1][2 * i - 1] += d10
        f[2 * i - 1][2 * i + 2 - 1] += d10

    for i in range(1, 10):
        u2ip1 = u[2 * i + 1 - 1]
        u2ip2 = u[2 * i + 2 - 1]
        d1 = (-2 * u2ip1 - 2) * (-u2ip1 - 1) * (-sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / (
                (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) ** (3 / 2) - (-2 * u2ip1 - 2) * (u2ip1 + 1) / (
                     (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) - 2 * (
                     -sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)
        d2 = (-2 * u2ip2 - 2) * (-u2ip2 - 1) * (-sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / (
                (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) ** (3 / 2) - (-2 * u2ip2 - 2) * (u2ip2 + 1) / (
                     (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) - 2 * (
                     -sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)
        f[2 * i + 1 - 1][2 * i + 1 - 1] += d1
        f[2 * i + 2 - 1][2 * i + 2 - 1] += d2
        d3 = (-2 * u2ip1 - 2) * (-u2ip1 - 1) * (-sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / (
                (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) ** (3 / 2) - (-2 * u2ip1 - 2) * (u2ip1 + 1) / (
                     (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) - 2 * (
                     -sqrt((u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2) + sqrt(2)) / sqrt(
            (u2ip1 + 1) ** 2 + (u2ip2 + 1) ** 2)
        f[2 * i + 1 - 1][2 * i + 2 - 1] += d3
        f[2 * i + 2 - 1][2 * i + 1 - 1] += d3

    for i in range(1, 10):
        u2im1 = u[2 * i - 1 - 1]
        u2i = u[2 * i - 1]
        d1 = (1 - u2im1) * (2 - 2 * u2im1) * (-sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / (
                (u2i + 1) ** 2 + (u2im1 - 1) ** 2) ** (3 / 2) - (2 - 2 * u2im1) * (u2im1 - 1) / (
                     (u2i + 1) ** 2 + (u2im1 - 1) ** 2) - 2 * (
                     -sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2)
        d2 = (-2 * u2i - 2) * (-u2i - 1) * (-sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / (
                (u2i + 1) ** 2 + (u2im1 - 1) ** 2) ** (3 / 2) - (-2 * u2i - 2) * (u2i + 1) / (
                     (u2i + 1) ** 2 + (u2im1 - 1) ** 2) - 2 * (
                     -sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2)
        f[2 * i - 1 - 1][2 * i - 1 - 1] += d1
        f[2 * i - 1][2 * i - 1] += d2
        d3 = (2 - 2 * u2im1) * (-u2i - 1) * (-sqrt((u2i + 1) ** 2 + (u2im1 - 1) ** 2) + sqrt(2)) / (
                (u2i + 1) ** 2 + (u2im1 - 1) ** 2) ** (3 / 2) - (2 - 2 * u2im1) * (u2i + 1) / (
                     (u2i + 1) ** 2 + (u2im1 - 1) ** 2)
        f[2 * i - 1 - 1][2 * i - 1] += d3
        f[2 * i - 1][2 * i - 1 - 1] += d3

    return f


u = np.zeros(20)
alpha = 0
h = np.zeros(20)
variables = np.zeros(400).reshape((20, 20))
a = np.ones(37)
f = model3a(u, a)
tol = 1e-12
g = np.zeros(20)
g[1] = 1
g[18] = 1
# variables[1] = np.subtract(f[1], 1)
# variables[18] = np.subtract(f[18], 1)
count = 0
l2 = np.sqrt(np.dot(np.transpose(f), f))
print("NOTE: THE CODE IS TAKING FOREVER TO FINISH WITH THIS TOL. AFTER 1000 ITERATIONS THE PROGRAM STOPS. THE "
      "APPORXIMATIONS SEEM TO BE GOOD ENOUGH")
while l2 > tol:
    if count == 1e3:
        break
    if count % 100 == 0:
        print("l2 %.12f" % l2)

    K = hessian(u, variables)

    h = np.linalg.solve(K, np.multiply(-1, f))
    for i in range(len(u)):
        u[i] += h[i]

    f = model3a(u, a)
    l2 = np.sqrt(np.dot(np.transpose(f), f))
    count += 1

print(u)
print("mu %.4f" % mu(u, a, g))
