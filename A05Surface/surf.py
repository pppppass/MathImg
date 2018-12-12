import numpy
import pywt


def evolve_surf(func, size, phi, mu, alpha, wavelet_name, level, iters, debug=False):
    
    n, l, f = size, level, func
    
    f = f[:n[0], :n[1], :n[2]]
    phi = phi[:n[0], :n[1], :n[2]]
    
    r = 2.0 * f - 1.0
    
    keys = ["aaa", "aad", "ada", "add", "daa", "dad", "dda", "ddd"]
    
    eta = 1.618
    
    c = [{k: 2.0 ** (3.0 / 2.0 * (l - j)) for k in keys} for j in range(l)]
    mask = [{**{k: 1.0 for k in keys if k != "aaa"}, **{"aaa": 0.0}} for j in range(l)]
    
    u = 1.0 - f
    w_u = pywt.swtn(u, wavelet_name, level=l)
    d = [{k: w_u[j][k] / c[j][k] for k in keys} for j in range(l)]
    v = [{k: numpy.zeros(n) for k in keys} for j in range(l)]
    
    for it in range(iters):
        
        u_old = u
        
        u = pywt.iswtn([{k:
            c[j][k] * (d[j][k] - v[j][k])
        for k in keys} for j in range(l)], wavelet_name) - r / mu
        u = numpy.maximum(numpy.minimum(u, 1.0), 0.0)
        w_u = pywt.swtn(u, wavelet_name, level=l)
        
        if debug:
            crit = numpy.linalg.norm(u.flatten() - u_old.flatten(), 2.0) / numpy.linalg.norm(u_old.flatten(), 2.0)
            print("Criterion: {}".format(crit))
            
        d = [{k:
            w_u[j][k] / c[j][k] + v[j][k] - numpy.maximum(numpy.minimum(w_u[j][k] / c[j][k] + v[j][k], alpha / mu * mask[j][k] / c[j][k] * phi), -alpha / mu * mask[j][k] / c[j][k] * phi)
        for k in keys} for j in range(l)]
        
        v = [{k:
            v[j][k] + eta * (w_u[j][k] / c[j][k] - d[j][k])
        for k in keys} for j in range(l)]
        
        del(w_u)
        
        print("Iteration {} finished".format(it))
    
    return u
