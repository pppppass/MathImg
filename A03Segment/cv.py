import numpy


def grad(image):
    u = image
    n_x, n_y = u.shape
    h = 1.0 / min(n_x, n_y)
    g = numpy.zeros((n_x, n_y, 2))
    g[:-1, :, 0] = (u[1:, :] - u[:-1, :]) / h
    g[:, :-1, 1] = (u[:, 1:] - u[:, :-1]) / h
    return g


def grad_t(grad):
    g = grad
    n_x, n_y, _ = g.shape
    h = 1.0 / min(n_x, n_y)
    u = numpy.zeros((n_x, n_y))
    u[:-1, :] -= g[:-1, :, 0] / h
    u[1:, :] += g[:-1, :, 0] / h
    u[:, :-1] -= g[:, :-1, 1] / h
    u[:, 1:] += g[:, :-1, 1] / h
    return u


def evolve_cv(image, mu, c1, c2, alpha, beta, out_iters, in_iters, debug=False):
    
    i = image
    
    n_x, n_y = i.shape
    
    u = numpy.ones((n_x, n_y)) * 0.5
    p = numpy.zeros((n_x, n_y, 2))
    
    for k in range(out_iters):
        r = (i - c1)**2 - (i - c2)**2
        for j in range(in_iters):
            p -= beta * grad(u)
            l = numpy.linalg.norm(p, 2.0, axis=2)[:, :, None]
            p = p / (l + 1.0e-10) * numpy.minimum(l, 1.0)
            u -= alpha * (mu * r - grad_t(p))
            u = numpy.maximum(numpy.minimum(u, 1.0), 0.0)
            if debug:
                print("objective:", numpy.sum(numpy.abs(grad(u))) + mu * numpy.sum(u * r))
        f = u > 0.5
        c1, c2 = i[f].mean(), i[~f].mean()
    
    return u, c1, c2
