import numpy
import pywt
import skimage.measure
import skimage.filters


def opt_balance_prox(image, sigma, lamda, kappa, wavelet_name, level, gamma, iters, eps, wavelet_style="swt", bound="periodization", fold=False, nesterov=False, debug=False, truth=None):
    
    f, l, c = image, level, lamda
    i_truth = truth
    n_x, n_y = f.shape
    
    if fold:
        f = numpy.vstack([f, f[::-1, :]])
        f = numpy.hstack([f, f[:, ::-1]])
        n_x_, n_y_ = 2*n_x, 2*n_y
    else:
        n_x_, n_y_ = n_x, n_y
    
    if wavelet_style == "swt":
        def wavelet_decomp(image):
            u = image
            v_a, v_d = [], []
            v = pywt.swt2(u, wavelet_name, l)
            for i in range(l):
                v_a.append(v[i][0][None, :, :])
                lh, hl, hh = v[i][1]
                v_d.append(numpy.stack([lh, hl, hh], axis=0))
            v_ = numpy.array(v_d + v_a, dtype=object)
            return v_
        def wavelet_reconstr(coef):
            v = coef
            v_ = []
            for i in range(l):
                ll, lh, hl, hh = v[i+l][0], v[i][0], v[i][1], v[i][2]
                v_.append((ll, (lh, hl, hh)))
            u = pywt.iswt2(v_, wavelet_name)
            return u
    elif wavelet_style == "dwt":
        def wavelet_decomp(image):
            u = image
            v = []
            for i in range(l):
                u, (lh, hl, hh) = pywt.dwt2(u, wavelet_name, mode=bound)
                v.append(numpy.stack([lh, hl, hh], axis=0))
            v = v[::]
            v.append(u[None, :, :])
            v_ = numpy.array(v, dtype=object)
            return v_
        def wavelet_reconstr(coef):
            v = coef
            u = v[l][0, :, :]
            for i in range(l):
                lh, hl, hh = v[i][0], v[i][1], v[i][2]
                u = pywt.idwt2((u, (lh, hl, hh)), wavelet_name, mode=bound)
            return u
    
    if wavelet_style == "swt":
        lamda = numpy.array([c] * l + [0.0] * l)
    elif wavelet_style == "dwt":
        lamda = numpy.array([c * 2**(l - k) for k in range(l)] + [0.0])
    
    def calc_sol(coef):
        alpha = coef
        u = wavelet_reconstr(alpha)
        u_sol = numpy.clip(u[:n_x, :n_y], 0.0, 1.0)
        return u_sol
    
    a_t_f = skimage.filters.gaussian(f, sigma)
    alpha = wavelet_decomp(f)
    alpha_old = alpha
    
    for it in range(iters):
        
        alpha_old_ = alpha
        if nesterov:
            alpha = alpha + (it - 1) / (it + 2) * (alpha - alpha_old)
        alpha_old = alpha_old_    
        
        w_t_alpha = wavelet_reconstr(alpha)
        alpha = alpha - gamma * (
              wavelet_decomp(
                  skimage.filters.gaussian(w_t_alpha, sigma * numpy.sqrt(2.0)) - a_t_f
                - kappa * w_t_alpha
              )
            + kappa * alpha
        )
        
        alpha_length = numpy.abs(alpha)
        s = numpy.array([
                numpy.maximum(alpha_, 0.0)
            for alpha_ in (
                alpha_length - lamda * gamma
        )], dtype=object) / (alpha_length + 1.0e-15)
        alpha *= s
        
        c = numpy.sqrt(sum(
                numpy.linalg.norm(alpha_.flatten(), 2.0)**2
            for alpha_ in
                (alpha - alpha_old)
        )) / numpy.linalg.norm(f.flatten(), 2.0)
        
        if debug:
            print("{}-th iteration, criterion = {:.5e}".format(it, c))
            if truth is not None:
                u_sol = calc_sol(alpha)
                psnr = skimage.measure.compare_psnr(i_truth, u_sol)
                ssim = skimage.measure.compare_ssim(i_truth, u_sol)
                print("PSNR = {:.5f}, SSIM = {:.5f}".format(psnr, ssim))
        if c < eps:
            break

    u_sol = calc_sol(alpha)
    return u_sol, it
