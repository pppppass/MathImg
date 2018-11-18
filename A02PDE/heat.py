import numpy
import skimage.measure


def evolve_heat(image, mu, iters, eps=None, debug=False, truth=None):
    
    f = image
    i_truth = truth
    
    n_x, n_y = f.shape
    h = 1.0 / min(n_x, n_y)
    tau = mu * h**2
    
    u = f
    
    for it in range(iters):
        
        u_ = numpy.zeros(u.shape)
        u_[:-1, :]  += u[:-1, :] - u[1:, :]
        u_[1:, :] += u[1:, :] - u[:-1, :]
        u_[:, :-1] += u[:, :-1] - u[:, 1:]
        u_[:, 1:] += u[:, 1:] - u[:, :-1]
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
