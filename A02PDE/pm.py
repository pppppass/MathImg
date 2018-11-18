import numpy
import skimage.measure


def evolve_pm(image, mu, iters, coef, nonl="sqrt", grad="full", eps=None, debug=False, truth=None):
    
    k = coef
    f = image
    i_truth = truth
    
    n_x, n_y = f.shape
    h = 1.0 / min(n_x, n_y)
    tau = mu * h**2
    
    if nonl == "frac":
        def nonlinear(g):
            b = 1.0 / (1.0 + g / k)
            return b
    elif nonl == "sqrt":
        def nonlinear(g):
            b = 1.0 / numpy.sqrt(1.0 + g)
            return b
    
    if grad == "fast":
        def grad_x_2(u):
            g = ((u[1:, :] - u[:-1, :]) / h)**2
            return g
        def grad_y_2(u):
            g = ((u[:, 1:] - u[:, :-1]) / h)**2
            return g
    elif grad == "full":
        def grad_x_2(u):
            g = numpy.zeros((n_x - 1, n_y))
            g[:, 1:-1] = (u[:-1, 2:] + u[1:, 2:] - u[:-1, :-2] - u[1:, :-2]) / h / 4.0
            g[:, 0] = (u[:-1, 1] + u[1:, 1] - u[:-1, 0] - u[1:, 0]) / h / 4.0
            g[:, -1] = (u[:-1, -1] + u[1:, -1] - u[:-1, -2] - u[1:, -2]) / h / 4.0
            g = g**2 + ((u[1:, :] - u[:-1, :]) / h)**2
            return g
        def grad_y_2(u):
            g = numpy.zeros((n_x, n_y - 1))
            g[1:-1, :] = (u[2:, :-1] + u[2:, 1:] - u[:-2, :-1] - u[:-2, 1:]) / h / 4.0
            g[0, :] = (u[1, :-1] + u[1, 1:] - u[0, :-1] - u[0, 1:]) / h / 4.0
            g[-1, :] = (u[-1, :-1] + u[-1, 1:] - u[-2, :-1] - u[-2, 1:]) / h / 4.0
            g = g**2 + ((u[:, 1:] - u[:, :-1]) / h)**2
            return g
    
    u = f
    
    for it in range(iters):
        
        u_ = numpy.zeros(u.shape)
        b = nonlinear(grad_x_2(u))
        u_[:-1, :] += (u[:-1, :] - u[1:, :]) * b
        u_[1:, :] += (u[1:, :] - u[:-1, :]) * b
        b = nonlinear(grad_y_2(u))
        u_[:, :-1] += (u[:, :-1] - u[:, 1:]) * b
        u_[:, 1:] += (u[:, 1:] - u[:, :-1]) * b
        u = u - mu * u_
        
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
