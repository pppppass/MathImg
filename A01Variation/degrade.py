import argparse
import numpy
import skimage
import tv

parser = argparse.ArgumentParser(description="Degrade an image")
parser.add_argument("input", type=str, help="Input filename")
parser.add_argument("output", type=str, help="Output filename")
parser.add_argument("--sigma", type=float, default=2.0, help="Size of Gaussian kernel")
parser.add_argument("--eta", type=float, default=5.0/255.0, help="Strength of Gaussian noise")
parser.add_argument("--style", type=str, default="conv", help="Method of blurs")
parser.add_argument("--seed", type=int, default=1, help="Random seed")

args = parser.parse_args()

i = skimage.io.imread(args.input) / 255.0
i_degr = tv.calc_degrade(i, args.sigma, args.eta, style=args.style, seed=args.seed)
skimage.io.imsave(args.output, numpy.clip(i_degr, 0.0, 1.0))
