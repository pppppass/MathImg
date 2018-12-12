import numpy
import pywt


def evolve_wcv(image, mu, wavelet_name, level, c1, c2, alpha, beta, out_iters, in_iters, debug=False):
    
    i = image
    l = level
    
    n_x, n_y = i.shape
    h = 1.0 / min(n_x, n_y)
    
    lamda = numpy.array([1.0 / 2.0**(l - k - 1) for k in range(l)] + [0.0] * l)

    def wavelet_decomp(image):
        u = image
        v_a, v_d = [], []
        v = pywt.swt2(u, wavelet_name, l)
        for i in range(l):
            v_a.append(v[i][0][None, :, :] / 2.0**(l - i))
            lh, hl, hh = v[i][1][0] / 2.0**(l - i), v[i][1][1] / 2.0**(l - i), v[i][1][2] / 2.0**(l - i)
            v_d.append(numpy.stack([lh, hl, hh], axis=0))
        v_ = numpy.array(v_d + v_a, dtype=object)
        return v_

    def wavelet_reconstr(coef):
        v = coef
        v_ = []
        for i in range(l):
            ll, lh, hl, hh = v[i+l][0] * 2.0**(l - i), v[i][0] * 2.0**(l - i), v[i][1] * 2.0**(l - i), v[i][2] * 2.0**(l - i)
            v_.append((ll, (lh, hl, hh)))
        u = pywt.iswt2(v_, wavelet_name)
        return u
    
    u = numpy.ones((n_x, n_y)) * 0.5
    p = numpy.array([numpy.zeros((3, n_x, n_y)) for t in range(l)] + [numpy.zeros((1, n_x, n_y)) for t in range(l)], dtype=object)
    
    for k in range(out_iters):
        r = (i - c1)**2 - (i - c2)**2
        for j in range(in_iters):
            p -= (2.0 * beta / h) * wavelet_decomp(u)
            p = numpy.array([
                    numpy.maximum(numpy.minimum(p_, lambda_), -lambda_)
                for p_, lambda_ in
                    zip(p, lamda)
            ], dtype=object)
            u -= alpha * (mu * r - (2.0 / h) * wavelet_reconstr(p))
            u = numpy.maximum(numpy.minimum(u, 1.0), 0.0)
            if debug:
                print("Lagrange: ", mu * numpy.sum(u * r) - sum(
                        lambda_ * numpy.sum(p_ * w_u_)
                    for p_, w_u_, lambda_ in
                        zip(p, 1024.0 * wavelet_decomp(u), lamda)
                ))
        f = u > 0.5
        c1, c2 = i[f].mean(), i[~f].mean()
    
    return u, c1, c2
