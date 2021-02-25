import os
import logzero
import argparse
from tqdm import tqdm



def format_for_align(files_to_format, output_dir):
    abs_files = [os.path.abspath(file) for file in files_to_format]
    output_dir = os.path.abspath(output_dir) 

    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir) 

    dashes = 89 * '-' + '\n'
    print(dashes)
    for file_name in tqdm(abs_files):
        file = open(file_name, 'r').read().remove(dashes)
        print(file)

if __name__ == "__main__":
    # pass all the files that need to be put into formated

    # pass were they should be outputter

    parser = argparse.ArgumentParser(
        description="Adds indexes to passed in files and separates the index and the line by a tab.")

    parser.add_argument("files_to_format", metavar="N", type=str, nargs="+", help="Files that will be formated")
    parser.add_argument("--output_dir", help="Directory where the formated files will be saved")

    args = parser.parse_args()

    files_to_format = args.files_to_format
    output_dir = args.output_dir

    format_for_align(files_to_format, output_dir)

    print(files_to_format)
    print(output_dir)