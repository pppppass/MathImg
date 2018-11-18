import numpy
import scipy.fftpack
import skimage.filters


def calc_degrade(image, sigma, eta, style="conv", seed=1):
    i = image
    n_x, n_y = i.shape
    if sigma is not None:
        if style == "conv":
            i_blur = skimage.filters.gaussian(i, sigma=sigma)
        elif style == "fft":
            c_x, c_y = n_x // 2, n_y // 2
            i_x, i_y = numpy.indices((n_x, n_y))
            k = numpy.exp(-1.0 / 2.0 / sigma**2 * ((i_x - c_x)**2 + (i_y - c_y)**2))
            k = numpy.roll(k, (-c_x, -c_y), axis=(0, 1))
            k_ = scipy.fftpack.fft2(k)
            k_ /= k_[0, 0]
            i_ = scipy.fftpack.fft2(i)
            i_blur = scipy.fftpack.ifft2(k_ * i_).real
        elif style == "dct":
            i_x, i_y = numpy.indices((n_x, n_y))
            k = numpy.exp(-1.0 / 2.0 / sigma**2 * (i_x**2 + i_y**2))
            k_ = scipy.fftpack.dctn(k, type=1)
            k_ /= k_[0, 0]
            i_ = scipy.fftpack.dctn(i, norm="ortho", type=2)
            i_blur = scipy.fftpack.idctn(k_ * i_, norm="ortho", type=2)
        elif style == "dst":
            i_x, i_y = numpy.indices((n_x, n_y))
            k = numpy.exp(-1.0 / 2.0 / sigma**2 * (i_x**2 + i_y**2))
            k_ = scipy.fftpack.dctn(k, type=1)
            k_ /= k_[0, 0]
            i_ = scipy.fftpack.dstn(i, norm="ortho", type=2)
            i_blur = scipy.fftpack.idstn(k_ * i_, norm="ortho", type=2)
    else:
        i_blur = i
    if eta is not None:
        numpy.random.seed(seed)
        xi = eta * numpy.random.randn(n_x, n_y)
        i_degr = i_blur + xi
    else:
        i_degr = i_blur
    return i_degr
