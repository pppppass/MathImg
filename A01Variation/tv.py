import numpy
import scipy.fftpack
import skimage


def opt_tv_admm(image, sigma, lamda, rho, iters, eps, inv="dct", tv="iso", alpha=1.618, debug=False, truth=None):
    
    f = image
    i_truth = truth
    
    if inv == "fft":
        def grad(vec):
            i = vec
            g = numpy.zeros(i.shape + (2,))
            g[:, :, 0] = (numpy.roll(i, -1, axis=0) - i) * i.shape[0]
            g[:, :, 1] = (numpy.roll(i, -1, axis=1) - i) * i.shape[1]
            return g
        def grad_t(vec):
            g = vec
            i = numpy.zeros(g.shape[:-1])
            i[:, :] += (numpy.roll(g[:, :, 0], 1, axis=0) - g[:, :, 0]) * g.shape[0]
            i[:, :] += (numpy.roll(g[:, :, 1], 1, axis=1) - g[:, :, 1]) * g.shape[1]
            return i
    elif inv == "dct":
        def grad(vec):
            i = vec
            g = numpy.zeros((i.shape[0], i.shape[1], 2))
            g[:-1, :, 0] = (i[1:, :] - i[:-1, :]) * i.shape[0]
            g[:, :-1, 1] = (i[:, 1:] - i[:, :-1]) * i.shape[1]
            return g
        def grad_t(vec):
            g = vec
            i = numpy.zeros((g.shape[0], g.shape[1]))
            i[:-1, :] -= g[:-1, :, 0] * g.shape[0]
            i[1:, :] += g[:-1, :, 0] * g.shape[0]
            i[:, :-1] -= g[:, :-1, 1] * g.shape[1]
            i[:, 1:] += g[:, :-1, 1] * g.shape[1]
#             i[1:, :] += (g[:-1, :, 0] - g[1:, :, 0]) * g.shape[0]
#             i[0, :] -= g[0, :, 0] * g.shape[0]
#             i[:, 1:] += (g[:, :-1, 1] - g[:, 1:, 1]) * g.shape[1]
#             i[:, 0] -= g[:, 0, 1] * g.shape[1]
            return i
    elif inv == "dst":
        def grad(vec):
            i = vec
            g = numpy.zeros((i.shape[0] + 1, i.shape[1] + 1, 2))
#             g = numpy.zeros(i.shape + (2,))
#             g[:-1, :, 0] = (i[1:, :] - i[:-1, :]) * i.shape[0]
#             g[-1, :, 0] = -2.0 * i.shape[0] * i[-1, :]
#             g[:, :-1, 1] = (i[:, 1:] - i[:, :-1]) * i.shape[1]
#             g[:, -1, 1] = -2.0 * i.shape[1] * i[:, -1]
#             g[:-1, :-1, 0] += i[:, :] * i.shape[0]
#             g[1:, :-1, 0] -= i[:, :] * i.shape[0]
#             g[:-1, :-1, 1] += i[:, :] * i.shape[1]
#             g[:-1, 1:, 1] -= i[:, :] * i.shape[1]
#             g[1:, :-1, 0] = (i[:, :] - i[:-1, :]) * i.shape[0]
#             g[:, :-1, 1] = (i[:, 1:] - i[:, :-1]) * i.shape[1]
            g[1:-1, :-1, 0] -= i[:-1, :] * i.shape[0]
            g[1:-1, :-1, 0] += i[1:, :] * i.shape[0]
            g[0, :-1, 0] += numpy.sqrt(2.0) * i.shape[0] * i[0, :]
            g[-1, :-1, 0] -= numpy.sqrt(2.0) * i.shape[0] * i[-1, :]
            g[:-1, 1:-1, 1] -= i[:, :-1] * i.shape[1]
            g[:-1, 1:-1, 1] += i[:, 1:] * i.shape[1]
            g[:-1, 0, 1] += numpy.sqrt(2.0) * i.shape[1] * i[:, 0]
            g[:-1, -1, 1] -= numpy.sqrt(2.0) * i.shape[1] * i[:, -1]
            return g
        def grad_t(vec):
            g = vec
#             i = numpy.zeros(g.shape[:-1])
            i = numpy.zeros((g.shape[0] - 1, g.shape[1] - 1))
#             i[1:, :] += (g[:-1, :, 0] - g[1:, :, 0]) * g.shape[0]
#             i[0, :] += (-2.0 * g[:-1, :, 0].sum(axis=0) - g[-1, :, 0] - g[0, :, 0]) * g.shape[0]
#             i[:, 1:] += (g[:, :-1, 1] - g[:, 1:, 1]) * g.shape[1]
#             i[:, 0] += (-2.0 * g[:, :-1, 1].sum(axis=1) - g[:, -1, 1] - g[:, 0, 1]) * g.shape[1]
#             i[1:, :] += (g[:-1, :, 0] - g[1:, :, 0]) * g.shape[0]
#             i[0, :] -= g[0, :, 0] * g.shape[0]
#             i[:, 1:] += (g[:, :-1, 1] - g[:, 1:, 1]) * g.shape[1]
#             i[:, 0] -= g[:, 0, 1] * g.shape[1]
#             i[:, :] += (g[:-1, :-1, 0] - g[1:, :-1, 0]) * i.shape[0]
#             i[0, :] += g[0, :-1, 0] * i.shape[0]
#             i[-1, :] -= g[-1, :-1, 0] * i.shape[0]
#             i[:, :] += (g[:-1, :-1, 1] - g[:-1, 1:, 1]) * i.shape[1]
#             i[:, 0] += g[:-1, 0, 1] * i.shape[1]
#             i[:, -1] -= g[:-1, -1, 1] * i.shape[1]
            i[:-1, :] -= g[1:-1, :-1, 0] * i.shape[0]
            i[1:, :] += g[1:-1, :-1, 0] * i.shape[0]
            i[0, :] += numpy.sqrt(2.0) * i.shape[0] * g[0, :-1, 0]
            i[-1, :] -= numpy.sqrt(2.0) * i.shape[0] * g[-1, :-1, 0]
            i[:, :-1] -= g[:-1, 1:-1, 1] * i.shape[1]
            i[:, 1:] += g[:-1, 1:-1, 1] * i.shape[1]
            i[:, 0] += numpy.sqrt(2.0) * i.shape[1] * g[:-1, 0, 1]
            i[:, -1] -= numpy.sqrt(2.0) * i.shape[1] * g[:-1, -1, 1]
            return i

    n_x, n_y = f.shape
    u = f
    v = grad(u)
#     if inv == "fft":
#         p = numpy.zeros((n_x, n_y, 2))
#     elif inv == "dct":
#         p = numpy.zeros((n_x, n_y, 2))
#     elif inv == "dst":
#         p = numpy.zeros((n_x+1, n_y+1, 2))
    p = numpy.zeros_like(v)
    
    if inv == "fft":
        c_x, c_y = n_x // 2, n_y // 2
        i_x, i_y = numpy.indices((n_x, n_y))
        k = numpy.exp(-1.0 / 2.0 / sigma**2 * ((i_x - c_x)**2 + (i_y - c_y)**2))
        k = numpy.roll(k, (-c_x, -c_y), axis=(0, 1))
        k_ = scipy.fftpack.fft2(k)
        k_ /= k_[0, 0]
    elif inv == "dct" or inv == "dst":
        i_x, i_y = numpy.indices((n_x, n_y))
        k = numpy.exp(-1.0 / 2.0 / sigma**2 * (i_x**2 + i_y**2))
        k_ = scipy.fftpack.dctn(k, type=1)
        k_ /= k_[0, 0]
    
    if inv == "fft":
        d_t_d = numpy.zeros((n_x, n_y))
        d_t_d[0, 0] = 1.0
        d_t_d = grad_t(grad(d_t_d))
        d_t_d_ = scipy.fftpack.fft2(d_t_d)
    elif inv == "dct" or inv == "dst":
        d_t_d = numpy.zeros((n_x, n_y))
        d_t_d[1, 1] = 1.0
        d_t_d = grad_t(grad(d_t_d))
        d_t_d = numpy.roll(d_t_d, shift=(-1, -1), axis=(0, 1))
        d_t_d[2:, :] = 0.0
        d_t_d[:, 2:] = 0.0
        d_t_d_ = scipy.fftpack.dctn(d_t_d, type=1)

    if inv == "fft":
        f_ = scipy.fftpack.fft2(f)
        a_t_f_ = k_.conjugate() * f_
        w_ = k_ * k_.conjugate() + rho * d_t_d_
    elif inv == "dct":
        f_ = scipy.fftpack.dctn(f, norm="ortho", type=2)
        a_t_f_ = k_ * f_
        w_ = k_**2 + rho * d_t_d_
    elif inv == "dst":
        f_ = scipy.fftpack.dstn(f, norm="ortho", type=2)
        a_t_f_ = k_ * f_
        w_ = k_**2 + rho * d_t_d_
    
    if inv == "fft":
        def update_u(u):
            e_ = a_t_f_ + scipy.fftpack.fft2(grad_t(p) + rho * grad_t(v))
            u_ = e_ / w_
            u = scipy.fftpack.ifft2(u_).real
            return u
    elif inv == "dct":
        def update_u(u):
            e_ = a_t_f_ + scipy.fftpack.dctn(grad_t(p) + rho * grad_t(v), norm="ortho", type=2)
            u_ = e_ / w_
            u = scipy.fftpack.idctn(u_, norm="ortho", type=2)
            return u
    elif inv == "dst":
        def update_u(u):
            e_ = a_t_f_ + scipy.fftpack.dstn(grad_t(p) + rho * grad_t(v), norm="ortho", type=2)
            u_ = e_ / w_
            u = scipy.fftpack.idstn(u_, norm="ortho", type=2)
            return u
    
    if tv == "iso":
        def shrink_v(v):
            v_length = numpy.sqrt((v**2).sum(axis=2))
            s = numpy.maximum(v_length - lamda / rho, 0.0) / (v_length + 1.0e-16)
            v *= s[:, :, None]
            return v
    elif tv == "aniso":
        def shrink_v(v):
            v_length = numpy.abs(v)
            s = numpy.maximum(v_length - lamda / rho, 0.0) / (v_length + 1.0e-16)
            v *= s
            return v
    elif tv == "hard":
        def shrink_v(v):
            v_length = numpy.sqrt((v**2).sum(axis=2))
            s = v_length
            s[v_length < lamda / rho] = 0.0
            s = s / (v_length + 1.0e-16)
            v *= s[:, :, None]
            return v
        u_acc = numpy.zeros((n_x, n_y))
        u_ctr = 0
    
    if tv == "hard":
        def update_sol():
            nonlocal u_acc, u_ctr
            u_acc += u
            u_ctr += 1
        def calc_sol():
            return u_acc / u_ctr
    else:
        def update_sol():
            pass
        def calc_sol():
            return u
    
    for it in range(iters):
        
        u = update_u(u)
        d_u = grad(u)
        
        v = d_u - p / rho
        v = shrink_v(v)
        
        p = p + alpha * rho * (v - d_u)
        
        update_sol()
        
        c = numpy.linalg.norm((v - d_u).flatten(), 2.0) / numpy.linalg.norm(f.flatten(), 2.0)
        if debug:
            print("{}-th iteration, criterion = {:.5e}".format(it, c))
            if truth is not None:
                u_sol = calc_sol()
                psnr = skimage.measure.compare_psnr(i_truth, u_sol)
                ssim = skimage.measure.compare_ssim(i_truth, u_sol)
                print("PSNR = {:.5f}, SSIM = {:.5f}".format(psnr, ssim))
        if c < eps:
            break

    u_sol = calc_sol()
    return u_sol, it