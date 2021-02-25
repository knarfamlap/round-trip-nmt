import os
import logzero
import argparse
import spacy
from tqdm import tqdm
from simalign import SentenceAligner

nlp = spacy.load("en_core_web_sm")

def tokenize_sent(sent):
    doc = nlp(sent) 

    tokens = [token.text for token in doc]
    return tokens



def format_for_align(files_to_format, output_dir):
    abs_files = [os.path.abspath(file) for file in files_to_format]
    output_dir = os.path.abspath(output_dir) 

    # if not os.path.exists(output_dir):
    #     os.mkdir(output_dir) 

    dashes = 89 * '-' + '\n'
    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    for file_name in tqdm(abs_files):
        # remove all dashed lines 
        file_data = open(file_name, 'r').read().split('\n')
        basename = os.path.basename(file_name)
        # get number of translation per src sentence
        top = int(basename[basename.find("top")+3: basename.find("top") + 3 + 1])
        test_sentence_str = "Test Sentence: " 
        for i, line in enumerate(file_data):
            if line == dashes:
                continue
            elif  test_sentence_str in line:
                src_sent = tokenize_sent(line[len(test_sentence_str):])
                idx_beg = i + 2 # index of first generated sent
                idx_end = i + top + 1 # last index of the last generate sent

                for idx in range(idx_beg, idx_end + 1):
                    # get tokenized sent genereate round trip
                    rt_sent = tokenize_sent(file_data[idx])
                    aligments = aligner.get_word_aligns(src_sent, rt_sent)

                    




                    



        
        print(file_data)

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