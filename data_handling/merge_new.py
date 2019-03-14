import argparse
import fabio
import h5py
import numpy as np
import os


'''
# TODO:
- add additional arguments
- add frame separator function (c.f. merge.py)
- add argument p021 to allow automatic merging at beamline:
  set /gpfs/current/processed/merged as output dir
  create /gpfs/current/processed/merged if does not exist
  set /gpfs/current/raw/basename as the input dir
  set tif as the file_ext
- add argument p021_local to allow merging on own machine, assuming same
directory structure
'''

'''
Identifies the index separator used to

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
        test_filenames_printable = '\n'.join(test_filenames)
        raise Exception('Could not find any of the files:\n{1}\n\nCannot determine the character which separates filename from the frame index.'.format(path, test_filenames_printable))


'''
Generates full input path + basename and merged path + merged filename

TODO Move this into a main function
'''
def set_up_paths(basename, n_files, in_path=None, out_path=None):
    merge_filename = '{}_{:d}-merged.hdf5'.format(basename, n_files)
    if out_path:
        merge_filename = os.path.join(out_path, merge_filename)

    # Get rid of existing merge file for dataset 'basename'
    if os.path.exists(merge_filename):
        os.remove(merge_filename)

    # Open the files in sequence. For loop runs in steps of 1; +1 is needed to get the last file
    if in_path:
        basename = os.path.join(in_path, basename)

    return (merge_filename, basename)


'''
Creates final output file. Provide the averaged and summed datasets (could
these be passed in as a dictionary?)

TODO Should this be part of the main function too?
'''
def create_hdf5(merge_filename, avgd_dataset=np.empty((1,)), sumd_dataset=np.empty((1,))):
    # Create/update an hdf5 file which will hold our average & summed datasets
    with h5py.File(merge_filename, 'a') as merged_f:
        # On the first pass create the datasets...
        print('Creating hdf5 file...')
        merged_f.create_dataset('data/averaged', data=avgd_dataset, chunks=True, compression='gzip')
        merged_f.create_dataset('data/summed', data=sumd_dataset, chunks=True, compression='gzip')
        merged_f.flush()


'''
Assembles the path and file and opens the dataset

TODO This should be merged with the get_data function of merge.py
'''
def get_data(basename, file_num, file_ext):
    full_filename = basename + '-' + str(file_num).zfill(5) + '.' + file_ext

    print('Opening {}...'.format(full_filename))
    img = fabio.open(full_filename)
    return img.data


'''
Creates two new numpy datasets (summed and averaged) which are assembled from
the given basenames, numbers and file extensions.

Window average/summing is achieved by the functions calling itself with a subset
of the file numbers to be merged which corresponds to the window.

TODO Adds bounds argument (c.f. merge.py merge_frames)
'''
def merge(basename, file_nums, file_ext, window=False):
    first_run = True
    for i in range(len(file_nums)):
        num = file_nums[i]
        if window:
            avg_n_dset, sum_n_dset = merge(basename, num, file_ext)
            if first_run:
                dset_shape = (len(file_nums), avg_n_dset.shape[0], avg_n_dset.shape[1])
                avg_dset = np.empty(dset_shape)
                sum_dset = np.empty(dset_shape)
                first_run = False
            avg_dset[i, ...] = avg_n_dset
            sum_dset[i, ...] = sum_n_dset
        else:
            next_data = get_data(basename, num, file_ext)

            # Process the data
            if first_run:
                avg_dset = sum_dset = next_data
                first_run = False
            else:
                avg_dset = (avg_dset + next_data) / 2
                sum_dset = sum_dset + next_data

    return avg_dset, sum_dset

def main():
    parser = argparse.ArgumentParser(description='Merge a set of files as summed and averaged datasets in an hdf5 file.')
    parser.add_argument('basename', metavar='base', type=str, help='File base name')
    parser.add_argument('file_ext', action='store', type=str, help='File extension (no .) of the files to be processed')
    parser.add_argument('-i', '--in-path', dest='in_path', action='store', type=str, default=os.getcwd(), help='Directory containing input files')
    parser.add_argument('-o', '--out-path', dest='out_path', action='store', type=str, default=os.getcwd(), help='Directory where output hdf5 will be written')
    parser.add_argument('-r', '--range', dest='file_num_lims', type=int, nargs=2)
    parser.add_argument('-l', '--list', dest='file_num_list', type=int, nargs='*')
    parser.add_argument('--exclude', type=int, nargs='*')
    parser.add_argument('-w', '--window-size', dest='window_size', type=int, default=1)

        # parser.add_argument('-n', '--number', dest='n_files', action='store', type=int, default=None, help='Number of files to process (should be an integer!)')
        # parser.add_argument('-s', '--start-at', dest='init_n', action='store', type=int, default=None, help='Number to start counting the sequence of file numbers at')
        # parser.add_argument('-f', '--finish-at', dest='final_n', action='store', type=int, default=None, help='Number to stop counting the sequence of file numbers at')
        # parser.add_argument('--include', dest='incl', action='store', nargs='*', type=int, default=None, help='File numbers to be explicitly included')
        # parser.add_argument('--dset', dest='dset_path', action='store', type=str, default=None, help='Path to dataset in hdf file')
        # parser.add_argument('--dset-start', dest='dset_start', action='store', type=str, default=None, help='First frame in an hdf dataset to merge')
        # parser.add_argument('--dset-finish', dest='dset_end', action='store', type=str, default=None, help='First frame in an hdf dataset to merge')


    args = parser.parse_args()

    if args.file_num_list:
        file_list = args.file_num_list
    elif args.file_num_lims:
        # Calculate range. Need +1 to ensure we get the last number too!
        file_list = list(range(args.file_num_lims[0], args.file_num_lims[1] + 1))

    if args.exclude:
        for num in args.exclude:
            file_list.remove(num)

    # Set up paths
    merge_file_path_name, basename = set_up_paths(args.basename, len(file_list), args.in_path, args.out_path)

    # Reshape list depending on window size
    if args.window_size > 1:
        if len(file_list) % args.window_size != 0:
            print('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\nWARNING: Number of files does not divide by {d} an integer number of times.\n The last window will be for a smaller set\n'.format(args.window_size))

        window_file_list = []
        for i in range(len(file_list)):
            if i % args.window_size == 0:
                window_file_list.append([file_list[i]])
            else:
                window_file_list[-1].append(file_list[i])
        file_list = window_file_list

    avg_dset, sum_dset = merge(basename, file_list, args.file_ext, window=args.window_size > 1)
    create_hdf5(merge_file_path_name, avg_dset, sum_dset)


if __name__ == '__main__':
    main()
