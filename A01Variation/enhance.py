import time
import argparse
import numpy
import skimage
import tv

parser = argparse.ArgumentParser(description="Enhance a degraded image")
parser.add_argument("input", type=str, help="Input filename")
parser.add_argument("output", type=str, help="Output filename")
parser.add_argument("--sigma", type=float, default=2.0, help="Size of Gaussian kernel")
parser.add_argument("--lamda", type=float, default=2.0e-6, help="Regularization coefficient")
parser.add_argument("--rho", type=float, default=2.0e-6, help="Step size of ADMM")
parser.add_argument("--iters", type=int, default=1000, help="Maximum number of iterations")
parser.add_argument("--eps", type=float, default=1.0e-3, help="Tolerance")
parser.add_argument("--inv", type=str, default="dct", help="Method of calculating inverse")
parser.add_argument("--tv", type=str, default="iso", help="Type of total variation")
parser.add_argument("--alpha", type=float, default=1.618, help="Regularization coefficient")
parser.add_argument("--truth", type=str, default=None, help="Ground-truth image")

args = parser.parse_args()

i_degr = skimage.io.imread(args.input) / 255.0
start = time.time()
u, _ = tv.opt_tv_admm(i_degr, args.sigma, args.lamda, args.rho, args.iters, args.eps, inv=args.inv, tv=args.tv, alpha=args.alpha)
end = time.time()
skimage.io.imsave(args.output, numpy.clip(u, 0.0, 1.0))
print("Time: {:.5f}".format(end - start))
if args.truth is not None:
    i = skimage.io.imread(args.truth) / 255.0
    print("PSNR before: {:.5f}".format(skimage.measure.compare_psnr(i, i_degr)))
    print("PSNR after: {:.5f}".format(skimage.measure.compare_psnr(i, u)))
    print("SSIM before: {:.5f}".format(skimage.measure.compare_ssim(i, i_degr)))
    print("SSIM after: {:.5f}".format(skimage.measure.compare_ssim(i, u)))
