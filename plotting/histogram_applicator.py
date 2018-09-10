from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import argparse
import os

def get_img_array(image):
    img = Image.open(image)
    return np.asarray(img)

def histo_lim_calc(img, value_freq_percent=0.005):
    freqs, bins = np.histogram(img, bins=256)
    # Need to reduce the number of bins by one. Chop of the highest one
    bins = bins[:-1]
    min_freq = np.amax(freqs) * (value_freq_percent/100)
    more_than_min_bins = bins[freqs > min_freq]
    return (np.amin(more_than_min_bins), np.amax(more_than_min_bins))

def plot_img(img, name=None, outlier_fraction=0.005, histo_clims=None):
    if name ==None:
        name = os.path.splitext(img)[0]
    img_arr = get_img_array(img)

    plt.figure(tight_layout=True)
    if histo_clims:
        # Convert user supplied limits to a tuple of floats
        histo_clims = tuple(map(float, histo_clims))
    else:
        # No colormap limits provided, so calculate out own
        histo_clims = histo_lim_calc(img_arr, outlier_fraction)

    # List of colormaps available in matplotlib:
    # https://matplotlib.org/users/colormaps.html
    plt.imshow(img_arr, cmap="Blues_r", clim=tuple(histo_clims))
    plt.colorbar()

    #Set ticks for both axes to run from 0 to 2048 in steps of 500
    axes = plt.gca()
    axes.xaxis.set_ticks(np.arange(0,2048,500))
    axes.yaxis.set_ticks(np.arange(0,2048,500))

    #Save the image as a 300dpi png
    print("Saving {}.png...".format(name))
    plt.savefig(str(name)+".png", dpi=300)

if __name__ == '__main__':
    # are we working with a directory, a list or a  single file?
    parser = argparse.ArgumentParser(description='Applies a colormap to one, a list or a directory of 2d diffraction images. By default, outliers are selected using the same criteria as DAWN')
    parser.add_argument('-f', '--file', dest='filename', default=None)
    parser.add_argument('-l', '--list', nargs='*', dest='filelist', default=None)
    parser.add_argument('-d', '--directory', dest='dirname', default=None)
    parser.add_argument('-o', '--outliers', dest='outlier_frac', default=0.005)
    parser.add_argument('--histo-lims', dest='histo_clims', nargs=2)

    args = parser.parse_args()

    if args.outlier_frac:
        outliers = args.outlier_frac

    if args.filename:
        plot_img(args.filename, outlier_fraction=outliers, histo_clims=args.histo_clims)
    elif args.filelist:
        for imgfile in args.filelist:
            if os.path.exists(imgfile):
                plot_img(imgfile, outlier_fraction=outliers, histo_clims=args.histo_clims)
            else:
                raise Exception('Cannot find file {}'.format(imgfile))
    elif args.dirname:
        files_in_dir = os.listdir(args.dirname)
        for imgfile in files_in_dir:
            full_imgfile = os.path.join(args.dirname, imgfile)
            if os.path.splitext(full_imgfile)[1] not in ['.tif', '.tiff']:
                continue
            plot_img(full_imgfile, outlier_fraction=outliers, histo_clims=args.histo_clims)
    else:
        raise Exception('No image source provided.')
