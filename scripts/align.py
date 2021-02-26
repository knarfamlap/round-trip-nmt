import os
import argparse
import spacy
import torch
from tqdm import tqdm
from simalign import SentenceAligner
from logzero import logger

nlp = spacy.load("en_core_web_sm")


def load_sentence_aligner(device, model="bert", token_type="bpe", matching_methods="main"):
    aligner = SentenceAligner(model=model, token_type=token_type,
                              matching_methods=matching_methods, device=device)

    return aligner


def tokenize_sent(sent):
    doc = nlp(sent)

    tokens = [token.text for token in doc]
    return tokens


def alignments_to_wrds(aligments, src_sent_arr, trans_sent_arr, method="mwmf"):
    # arr with tuples alignment of src sent and traans sent
    aligned_indices = aligments[method]
    aligned_words = []

    for i, j in aligned_indices:
        aligned_words.append((src_sent_arr[i], trans_sent_arr[j]))

    return aligned_words


def align(files_to_align, output_dir, method, device):
    files_to_align = [os.path.abspath(file) for file in files_to_align]
    output_dir = os.path.abspath(output_dir)
    # making sure the output dir exits
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    # init aligner
    aligner = load_sentence_aligner(device)
    # Constant strings
    TEST_SENTENCE_STR = "Test Sentence: "
    TAB = "\t"
    # iterate all the files that were passed in
    for file_path in tqdm(files_to_align):
        # read the file
        file_basename = os.path.basename(file_path)
        # open the file where everything will be saved
        output_file = open(os.path.join(output_dir, "{}.align".format(
            file_basename)), "w", encoding='utf-8')
        last_src_sent_arr = ""

        for line in open(file_path, 'r'):
            # check to see if the line contains the source sentence
            # if so, save to use it later
            if TEST_SENTENCE_STR in line:
                line_copy = line.strip().rstrip("\n")
                last_src_sent_arr = tokenize_sent(
                    line[len(TEST_SENTENCE_STR):])
                output_file.write(line)
            # check if see if the line contains a TAB
            # if so, this is a translated sentence and it must be aligned
            elif TAB in line:
                # remove the tab from the line
                line_copy = line.strip().rstrip("\n")
                try:
                    # tokenize the trans sentence
                    trans_sent_arr = tokenize_sent(line_copy)
                    # get the alignments in index form
                    alignments_indeces = aligner.get_word_aligns(
                        last_src_sent_arr, trans_sent_arr)
                    # convert the alignments in word form
                    alignments_wrds = alignments_to_wrds(
                        alignments_indeces, last_src_sent_arr, trans_sent_arr, method)
                    # save the alignments into the output file
                    output_file.write("\t{}\n".format(alignments_wrds))
                except:
                    # if there was an error, make note
                    output_file.write("\t{}\n".format(
                        "CANNOT BE ALIGN: {}".format(line_copy)))
            # just copy the line into the output file
            else:
                output_file.write(line)

        logger.info("Wrote alignments into: {}".format(output_file.name))
        output_file.close()


if __name__ == "__main__":
    # pass all the files that need to be put into formated

    # pass were they should be outputter

    parser = argparse.ArgumentParser(
        description="Aligned the passed in files")

    parser.add_argument("files_to_align", metavar="N",
                        type=str, nargs="+", help="Files that will be aligned")
    parser.add_argument(
        "--output_dir", default="./aligned", help="Directory where the aligned files will be saved")
    parser.add_argument(
        "--device", type=str, default="0", help="Specify which gpu to use. If no gpu available then will use cpu"
    )
    parser.add_argument("--method", type=str, default="mwmf",
                        help="Method for alignment. Can be the string mwmf, inter, or itermax")
    args = parser.parse_args()

    files_to_align = args.files_to_align
    output_dir = args.output_dir
    method = args.method
    device = torch.device(
        f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    logger.info("Device in use: {}".format(device))
    align(files_to_align, output_dir, method, device)
