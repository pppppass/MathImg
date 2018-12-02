import numpy
import scipy.fftpack
import pywt
import skimage.measure


def opt_analysis_admm(image, sigma, lamda, wavelet_name, level, rho, iters, eps, wavelet_style="swt", bound="periodization", fold=False, debug=False, truth=None):
    
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
            v = v[::-1]
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
        lamda = numpy.array([c * 2**(l - k - 1) for k in range(l)] + [0.0])
    
    def calc_sol(image):
        u = image
        u_sol = numpy.clip(u[:n_x, :n_y], 0.0, 1.0)
        return u_sol
    
    if sigma is not None and sigma > 0.0:
        i_x, i_y = numpy.indices((n_x_, n_y_))
        k = numpy.exp(-1.0 / 2.0 / sigma**2 * (i_x**2 + i_y**2))
    else:
        k = numpy.zeros((n_x_, n_y_))
        k[0, 0] = 1.0
    k_ = scipy.fftpack.dctn(k, type=1)
    k_ /= k_[0, 0]
    
    f_ = scipy.fftpack.dctn(f, norm="ortho", type=2)
    a_t_f_ = k_ * f_
    w_ = k_**2 + rho
    
    u = f
    v = wavelet_decomp(u)
    p = numpy.array([numpy.zeros_like(v_) for v_ in v], dtype=object)
    
    for it in range(iters):
        
        e_ = a_t_f_ + scipy.fftpack.dctn(wavelet_reconstr(p + rho * v), norm="ortho", type=2)
        u_ = e_ / w_
        u = scipy.fftpack.idctn(u_, norm="ortho", type=2)
        w_u = wavelet_decomp(u)
        
        v = w_u - p / rho
        v_length = numpy.abs(v)
        s = numpy.array([
                numpy.maximum(v_, 0.0)
            for v_ in (
                v_length - lamda / rho
        )], dtype=object) / (v_length + 1.0e-15)
        v *= s
        
        p += 1.618 * rho * (v - w_u)
        
        c = numpy.sqrt(sum(
                numpy.linalg.norm(v_.flatten(), 2.0)**2
            for v_ in
                (v - w_u)
        )) / numpy.linalg.norm(f.flatten(), 2.0)
        if debug:
            print("{}-th iteration, criterion = {:.5e}".format(it, c))
            if truth is not None:
                u_sol = calc_sol(u)
                psnr = skimage.measure.compare_psnr(i_truth, u_sol)
                ssim = skimage.measure.compare_ssim(i_truth, u_sol)
                print("PSNR = {:.5f}, SSIM = {:.5f}".format(psnr, ssim))
        if c < eps:
            break

    u_sol = calc_sol(u)
    return u_sol, it
