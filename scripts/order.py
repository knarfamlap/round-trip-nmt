# grab all folder in a directory


# iterate through them and look for forward and backwards models.
# an example is: if we find en-de, we look for de-en
# if found, put that pair inside the target folder


from pathlib import Path
from shutil import copy
import argparse
import os
import logging
import errno

def filter_models(input_dirr):
    cur_dir = os.getcwd() 
    subdirs_with_paths = [os.path.abspath(f.path) for f in os.scandir(input_dirr) if f.is_dir()]
    
    filtered  = []
    parent_path = Path(subdirs_with_paths[0]).parent
    for subdir in subdirs_with_paths:
        if '+' in subdir:
            continue

        name_dir = os.path.split(subdir)[-1]
        src_lang = name_dir.split('-')[0]
        trg_lang = name_dir.split('-')[1]
        
        trg_src = os.path.join(parent_path, trg_lang + "-" + src_lang) 
        if trg_src in subdirs_with_paths and trg_src not in filtered:
            filtered.append(trg_src)
            filtered.append(subdir)

            logging.debug('Round Trip Pretrained Model Found for {}'.format(trg_src))


    return filtered


def move_filtered_models(output_dir, filtered_dirs):

    for dirr in filtered_dirs:
        dest = os.path.join(output_dir, os.path.basename(dirr))
        dest_file = os.path.join(dest, "README.md") 
        file_dirr = os.path.join(dirr, "README.md")

        logging.debug("Copying {} to {}".format(file_dirr, dest))
        os.mkdir(dest)
    
        with open(file_dirr, 'rb') as src, open(dest_file, 'wb') as dst: dst.write(src.read())

        logging.debug("Successfully copied {} to {}".format(file_dirr, dest)) 
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find all the forward and backward models and put them into a desire location.")

    parser.add_argument('--input', help='directory that contains all models')
    parser.add_argument('--output', help="directory where you want to save the forward and backward models") 

    args = parser.parse_args() 

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) 

    logging.debug('Looking at directiories in: {}'.format(args.input))
    logging.debug('Moving directories to: {}'.format(args.output)) 
    
    output_dir = os.path.abspath(args.output)
    os.mkdir(output_dir)

    move_filtered_models(output_dir, filter_models(args.input))
