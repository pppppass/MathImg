import numpy


def calc_init(image, eps):
    i = image
    n_x, n_y = i.shape
    x, y = (numpy.arange(n_x) / (n_x - 1))[:, None], (numpy.arange(n_y) / (n_y - 1))[None, :]
    u = numpy.maximum(numpy.maximum(-x, x - 1.0), numpy.maximum(-y, y - 1.0)) + eps
    return u

def evolve_reinit(func, steps, gamma=None):
    
    u = func
    
    n_x, n_y = u.shape
    h = 1.0 / min(n_x, n_y)
    
    if gamma is None:
        gamma = 0.5 * h
    
    for j in range(steps):

        u_x = (u[1:, :] - u[:-1, :]) / h
        u_y = (u[:, 1:] - u[:, :-1]) / h

        s = numpy.zeros((n_x, n_y))
        s[0, :] += numpy.minimum(u_x[0, :], 0.0)**2
        s[-1, :] += numpy.maximum(u_x[-1, :], 0.0)**2
        s[1:-1, :] += numpy.maximum(numpy.maximum(u_x[:-1, :], 0.0)**2, numpy.minimum(u_x[1:, :], 0.0)**2)
        s[:, 0] += numpy.minimum(u_y[:, 0], 0.0)**2
        s[:, -1] += numpy.maximum(u_y[:, -1], 0.0)**2
        s[:, 1:-1] += numpy.maximum(numpy.maximum(u_y[:, :-1], 0.0)**2, numpy.minimum(u_y[:, 1:], 0.0)**2)

        t = numpy.zeros((n_x, n_y))
        t[0, :] += numpy.maximum(u_x[0, :], 0.0)**2
        t[-1, :] += numpy.minimum(u_x[-1, :], 0.0)**2
        t[1:-1, :] += numpy.maximum(numpy.minimum(u_x[:-1, :], 0.0)**2, numpy.maximum(u_x[1:, :], 0.0)**2)
        t[:, 0] += numpy.maximum(u_y[:, 0], 0.0)**2
        t[:, -1] += numpy.minimum(u_y[:, -1], 0.0)**2
        t[:, 1:-1] += numpy.maximum(numpy.minimum(u_y[:, :-1], 0.0)**2, numpy.maximum(u_y[:, 1:], 0.0)**2)

        f = (u >= 0)

        l = (
              f * (numpy.sqrt(s) - 1.0) * u / numpy.sqrt(u**2 + s * h**2 + 1.0e-15)
            + (~f) * (numpy.sqrt(t) - 1.0) * u / numpy.sqrt(u**2 + t * h**2 + 1.0e-15)
        )

        u -= gamma * l
    
    return u


def evolve_gac(image, alpha, tau, edge, iters, reinit, func):

    i = image
    u = func
    
    def edge_detect(grad2):
        g2 = grad2
        e = 1.0 / (1.0 + edge * g2)
        return e
    
    n_x, n_y = image.shape
    h = 1.0 / min(n_x, n_y)
    
    i_x = numpy.zeros((n_x, n_y))
    i_x[1:-1, :] = (i[2:, :] - i[:-2, :]) / (2.0 * h)
    i_x[[0, -1], :] = (i[[1, -1], :] - i[[0, -2], :]) / (2.0 * h)
    i_y = numpy.zeros((n_x, n_y))
    i_y[:, 1:-1] = (i[:, 2:] - i[:, :-2]) / (2.0 * h)
    i_y[:, [0, -1]] = (i[:, [1, -1]] - i[:, [0, -2]]) / (2.0 * h)
    
    e = edge_detect(i_x**2 + i_y**2)
    
    e_x = numpy.zeros((n_x, n_y))
    e_x[1:-1, :] = (e[2:, :] - e[:-2, :]) / (2.0 * h)
    e_x[[0, -1], :] = (e[[1, -1], :] - e[[0, -2], :]) / (2.0 * h)
    e_y = numpy.zeros((n_x, n_y))
    e_y[:, 1:-1] = (e[:, 2:] - e[:, :-2]) / (2.0 * h)
    e_y[:, [0, -1]] = (e[:, [1, -1]] - e[:, [0, -2]]) / (2.0 * h)
    
    for j in range(iters):
    
        u_xc = numpy.zeros((n_x, n_y))
        u_xc[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * h)
        u_xc[[0, -1], :] = (u[[1, -1], :] - u[[0, -2], :]) / (2.0 * h)
        u_xd = (u[1:, :] - u[:-1, :]) / h
        u_yc = numpy.zeros((n_x, n_y))
        u_yc[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * h)
        u_yc[:, [0, -1]] = (u[:, [1, -1]] - u[:, [0, -2]]) / (2.0 * h)
        u_yd = (u[:, 1:] - u[:, :-1]) / h

        k = numpy.zeros((n_x, n_y))
        t = numpy.zeros((n_x, n_y))
        t[:-1, :] += u_xd / h
        t[1:, :] -= u_xd / h
        k += u_yc**2 * t
        t = numpy.zeros((n_x, n_y))
        t[1:-1, 1:-1] = (u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] + u[:-2, :-2]) / (2.0 * h**2)
        t[1:-1, [0, -1]] = (u[2:, [1, -1]] - u[2:, [0, -2]] - u[:-2, [1, -1]] + u[:-2, [0, -2]]) / (2.0 * h**2)
        t[[0, -1], 1:-1] = (u[[1, -1], 2:] - u[[0, -2], 2:] - u[[1, -1], :-2] + u[[0, -2], :-2]) / (2.0 * h**2)
        t[[[0], [-1]], [[0, -1]]] = (u[[[1], [-1]], [[1, -1]]] - u[[[1], [-1]], [[0, -2]]] - u[[[0], [-2]], [[1, -1]]] + u[[[0], [-2]], [[0, -2]]]) / (2.0 * h**2)
        k -= 2.0 * u_xc * u_yc * t
        t = numpy.zeros((n_x, n_y))
        t[:, :-1] += u_yd / h
        t[:, 1:] -= u_yd / h
        k += u_xc**2 * t

        l = e * k / (u_xc**2 + u_yc**2 + 1.0e-15)
    
        s = numpy.zeros((n_x, n_y))
        s[1:, :] += numpy.maximum(u_xd, 0.0)**2
        s[:-1, :] += numpy.minimum(u_xd, 0.0)**2
        s[:, 1:] += numpy.maximum(u_yd, 0.0)**2
        s[:, :-1] += numpy.minimum(u_yd, 0.0)**2
        s = alpha * e * numpy.sqrt(s)
        
        t = numpy.zeros((n_x, n_y))
        t[1:, :] += numpy.minimum(e_x[1:, :], 0.0) * u_xd
        t[:-1, :] += numpy.maximum(e_x[:-1, :], 0.0) * u_xd
        t[:, 1:] += numpy.minimum(e_y[:, 1:], 0.0) * u_yd
        t[:, :-1] += numpy.maximum(e_y[:, :-1], 0.0) * u_yd
    
        u += tau * (l + s + t)
        
        if (j + 1) % reinit[0] == 0:
            u = evolve_reinit(u, reinit[1])
        
    return u
