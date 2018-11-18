import numpy
import skimage.measure


def minmod(u, v):
    w = ((u > 0.0) & (v > 0.0)) * numpy.minimum(u, v) + ((u < 0.0) & (v < 0.0)) * numpy.maximum(u, v)
    return w


def evolve_shock(image, nu, iters, ind="curve", func="sign", eps=None, debug=False, truth=None):
    
    f = image
    i_truth = truth
    
    n_x, n_y = f.shape
    h = 1.0 / min(n_x, n_y)
    tau = nu * h
    
    if ind == "lap":
        def func_l(u):
            l = numpy.zeros((n_x, n_y))
            l[:-1, :] += (u[1:, :] - u[:-1, :]) / h**2
            l[1:, :] += (u[:-1, :] - u[1:, :]) / h**2
            l[:, :-1] += (u[:, 1:] - u[:, :-1]) / h**2
            l[:, 1:] += (u[:, :-1] - u[:, 1:]) / h**2
            l = l / h**2
            return l
    elif ind == "curve":
        def func_l(u):
            l = numpy.zeros((n_x, n_y))
            u_x = numpy.zeros((n_x, n_y))
            u_x[1:-1, :] = (u[2:, :] - u[:-2, :]) / (2.0 * h)
            u_x[[0, -1], :] = (u[[1, -1], :] - u[[0, -2], :]) / (2.0 * h)
            u_y = numpy.zeros((n_x, n_y))
            u_y[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * h)
            u_y[:, [0, -1]] = (u[:, [1, -1]] - u[:, [0, -2]]) / (2.0 * h)
            u_ = numpy.zeros((n_x, n_y))
            u_[:-1, :] += (u[1:, :] - u[:-1, :]) / h**2
            u_[1:, :] += (u[:-1, :] - u[1:, :]) / h**2
            l += u_x**2 * u_
            u_ = numpy.zeros((n_x, n_y))
            u_[1:-1, 1:-1] = (u[2:, 2:] - u[2:, :-2] - u[:-2, 2:] - u[:-2, :-2]) / (2.0 * h*2)
            u_[1:-1, [0, -1]] = (u[2:, [1, -1]] - u[2:, [0, -2]] - u[:-2, [1, -1]] + u[:-2, [0, -2]]) / (2.0 * h**2)
            u_[[0, -1], 1:-1] = (u[[1, -1], 2:] - u[[0, -2], 2:] - u[[1, -1], :-2] + u[[0, -2], :-2]) / (2.0 * h**2)
            u_[[[0], [-1]], [[0, -1]]] = (u[[[1], [-1]], [[1, -1]]] - u[[[1], [-1]], [[0, -2]]] - u[[[0], [-2]], [[1, -1]]] + u[[[0], [-2]], [[0, -2]]]) / (2.0 * h**2)
            l += 2.0 * u_x * u_y * u_
            u_ = numpy.zeros((n_x, n_y))
            u_[:, :-1] += (u[:, 1:] - u[:, :-1]) / h**2
            u_[:, 1:] += (u[:, :-1] - u[:, 1:]) / h**2
            l += u_y**2 * u_
            return l
    
    if func == "sign":
        def func_f(l):
            f = numpy.sign(l)
            return f
    elif func == "id":
        def func_f(l):
            f = l
            return f
    
    u = f
    
    for it in range(iters):
        
        g = numpy.zeros((n_x, n_y))
        g[1:-1, :] += minmod(u[1:-1, :] - u[:-2, :], u[2:, :] - u[1:-1, :])**2
        g[:, 1:-1] += minmod(u[:, 1:-1] - u[:, :-2], u[:, 2:] - u[:, 1:-1])**2
        g = numpy.sqrt(g)
        f = func_f(func_l(u))
        u = u - nu * g * f
        
        if debug:
            print("{}-th iteration".format(it))
            if truth is not None:
                psnr = skimage.measure.compare_psnr(i_truth, u)
                ssim = skimage.measure.compare_ssim(i_truth, u)
                print("PSNR = {:.5f}, SSIM = {:.5f}".format(psnr, ssim))
        if eps is not None:
            c = numpy.abs(numpy.corrcoef(u.flatten(), (f - u).flatten())[0, 1])
            if debug:
                print("Corr = {:.5e}".format(c))
            if it > 1 and c_old < c + eps:
                break
            c_old = c
    
    return u, it + 1
