import fabio
import h5py
import numpy as np
import os
import sys
import time

import argparse

__hdf_ext = ['.h5', '.hdf', '.nxs']


'''
From a given file path/name, return a numpy dataset
'''
def get_data(file_name, dset_path=None):

    if os.path.splitext(file_name)[1] in __hdf_ext and dset_path:
        with h5py.File(file_name, 'r') as data_file:
            return data_file.get(dset_path)
    else:
        # There is no close function in fabio, so can't use a with statement and I don't see how try/except would be useful
        data_file = fabio.open(file_name)
        return data_file.data



def merge_files(file_list, window=None):

    n_img = len(file_list)
    dataset = np.zeros(shape=(n_img, 1, 1))
    for i in range(n_img):
        print('Reading {}...'.format(file_list[i]))
        img = get_data(file_list[i])
        if dataset.size == n_img:
            dataset.resize((n_img, img.shape[-2], img.shape[-1]))
        dataset[i] = img

    return merge_frames(dataset, window=window)


def merge_frames(dataset, bounds=None, window=None):

    if window:
        if dataset.shape[0]%window != 0:
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('WARNING: Dataset axis does not divide by {}'.format(window))
            print('         Last frames will not be windowed')
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

        print('Dataset shape: {}'.format(dataset.shape))
        n_windows = dataset.shape[0] // window
        avgd = np.zeros(shape=(n_windows, dataset.shape[-2], dataset.shape[-1]))
        sumd = np.zeros(shape=(n_windows, dataset.shape[-2], dataset.shape[-1]))

        for i in range(n_windows):
            print('Merging window {0} of {1}...'.format(i+1, n_windows))
            bounds = (window*i, (window*i)+window-1)
            windowed_data = merge_frames(dataset, bounds)
            avgd[i] = windowed_data['avg']
            sumd[i] = windowed_data['sum']

    else:
        if bounds:
            # Are we starting at the end of the file and working backwards?
            if bounds[0] > bounds[1]:
                frame_list = range(bounds[0], bounds[1] - 1, -1)
            else:
                frame_list = range(bounds[0], bounds[1] + 1)
        else:
            frame_list = range(0, len(dataset))

        # The first run, we need to create our datasets to sum/average against
        # Having this as a separate block avoids an if-statement in the for-loop
        i, j = frame_list[0], 1
        first_frame = dataset[i]
        avgd = first_frame
        sumd = first_frame
        print('Merging: {}/{} (Frame number: {})'.format(j, len(frame_list), i))

        # Here we actually do the merging.
        # i needs to start from the second (index 1) position in frame_list
        for i in frame_list[1:]:
            j += 1
            frame = dataset[i]
            avgd = (avgd + frame) / 2
            sumd = sumd + frame
            print('Merging: {}/{} (Frame number: {})'.format(j, len(frame_list), i))

    return {"avg": avgd, "sum": sumd}


def build_file_path(path, basename, file_index, file_ext, frame_separator='_', zero_fill=5):
    filename = '{0}{1}{2:0{fill}}.{3}'.format(basename, frame_separator, file_index, file_ext, fill = zero_fill)
    return os.path.join(path, filename)


'''
TODO This would be better done with regex:
- identify files with the basename in the given directory
- assert structure consists of basename+sep+file_index+.+file_ext
- extract the sep based on the same regex
- print warning if it's not in possible_separators
'''
def get_frame_nr_separator(path, basename, file_index, file_ext, zero_fill=5):
    possible_separators = ['_', '-']
    test_filenames = []
    for sep in possible_separators:
        filename = build_file_path(path, basename, file_index, file_ext, frame_separator=sep, zero_fill=zero_fill)
        test_filenames.append(filename)
        if os.path.exists(filename):
            return sep
    else:
        test_filenames_prinable = '\n'.join(test_filenames)
        raise Exception('Could not find any of the files:\n{1}\n\nCannot determine the character which separates filename from the frame index.'.format(path, test_filenames_prinable))


def create_merged_hdf(out_file_path, datasets_to_write):

    with h5py.File(out_file_path, 'w') as out_data_file:
        out_data_file.create_dataset("data/averaged", data=datasets_to_write['avg'], chunks=True)
        out_data_file.create_dataset("data/summed", data=datasets_to_write['sum'], chunks=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Merge a set of files as summed and averaged datasets in an hdf5 file.")
    parser.add_argument("basename", metavar="file base name", type=str)
    parser.add_argument("file_ext", metavar='file_ext', type=str)
    parser.add_argument("-n", "--number", dest="n_files", action="store", type=int, default=None, help="Number of files to process (should be an integer!)")
    parser.add_argument("-s", "--start-at", dest="init_n", action="store", type=int, default=None, help="Number to start counting the sequence of file numbers at")
    parser.add_argument('-f', '--finish-at', dest='final_n', action="store", type=int, default=None, help="Number to stop counting the sequence of file numbers at")
    parser.add_argument("-i", "--in-path", dest="in_path", action="store", type=str, default='', help="Directory containing input files")
    parser.add_argument("-o", "--out-path", dest="out_path", action="store", type=str, default='', help="Directory where output hdf5 will be written")
    parser.add_argument('-w', '--window-size', dest='window', action='store', type=int, default=None, help='Width of window for window avergaing')
    parser.add_argument("--exclude", dest="excl", action="store", nargs="*", type=int, default=None, help="File numbers to be excluded")
    parser.add_argument("--include", dest="incl", action="store", nargs="*", type=int, default=None, help="File numbers to be explicitly included")
    parser.add_argument('--dset', dest='dset_path', action='store', type=str, default=None, help='Path to dataset in hdf file')
    parser.add_argument('--dset-start', dest='dset_start', action='store', type=str, default=None, help='First frame in an hdf dataset to merge')
    parser.add_argument('--dset-finish', dest='dset_end', action='store', type=str, default=None, help='First frame in an hdf dataset to merge')

    args=parser.parse_args()

    start = 0
    end = 0

    if (args.init_n is args.final_n is args.n_files is None) or (args.init_n is args.final_n is args.n_files is not None):
        print('Please specify only two of --start-at, --finish-at and --number')
        sys.exit(1)

    if args.init_n is not None and args.final_n is not None:
        start = args.init_n
        end = args.final_n + 1
    elif args.init_n is not None and args.n_files is not None:
        start = args.init_n
        end = start + args.n_files
    elif args.final_n is not None and args.n_files is not None:
        end = args.final_n + 1
        start = end - args.n_files
    else:
        print('Could not determine file numbers to merge. Did you give two of --start-at, --finish-at and --number?')
        sys.exit(1)

    file_numbers = list(range(start, end))
    if args.excl:
        for nr in args.excl:
            file_numbers.remove(nr)
    if args.incl:
        for nr in args.incl:
            file_numbers.append(nr)

    in_path = os.path.normpath(args.in_path)
    out_path = os.path.normpath(args.out_path)

    #Find out what character to use ofr the frame number separator:
    frame_sep = get_frame_nr_separator(in_path, args.basename, file_numbers[0], args.file_ext)

    if len(file_numbers) == 1 and args.file_ext in __hdf_ext:
        dataset = get_data(build_file_path(in_path, args.basename, file_numbers[0], args.file_ext, frame_separator=frame_sep), args.dset_path)
        datasets_to_write = merge_frames(dataset, (args.dset_start, args.dset_end), window=args.window)
    else:
        file_list = []
        for i in file_numbers:
            file_list.append(build_file_path(in_path, args.basename, i, args.file_ext, frame_separator=frame_sep))

        #TODO: What about multiple hdf files? dset path, start, end???
        datasets_to_write = merge_files(file_list, window = args.window)

    runtime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_file_name = "{0}_{1}_{2}".format(runtime, args.basename, file_numbers[0])
    if len(file_numbers) > 1:
        out_file_name = '{0}-{1}'.format(out_file_name, file_numbers[-1])
    if args.dset_start or args.dset_end:
        #TODO: Needs testing with an hdf file
        out_file_name = '{0}_Frames({1}-{2})'.format(out_file_name, args.dset_start, args.dset_end)
    out_file_name = '{0}.{1}'.format(out_file_name, 'hdf')
    out_file_name = os.path.join(out_path, out_file_name)
    create_merged_hdf(out_file_name, datasets_to_write)
