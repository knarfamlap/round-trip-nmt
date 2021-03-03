import argparse
import os
import torch
from logzero import logger
from utils import load_model_and_tokenizer, save_nbest
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


def translate_from_trg_to_src(
        src,
        trg,
        src_to_trg_translations_dict,
        nbest,
        device='cuda:0'):

    logger.info('Getting {}-{} model and its tokenizer'.format(trg, src))
    model, tokenizer = load_model_and_tokenizer(
        trg, src, device=device)

    trg_to_src_translations = []
    for src_to_trg_translations in tqdm(src_to_trg_translations_dict.values()):

        for translation in src_to_trg_translations:

            trg_to_src_translations.append([])
            tokens_dict_src = tokenizer.prepare_seq2seq_batch(
                translation, return_tensors="pt"
            )

            # Translate from src to pivot language. Sentences are encoded
            trg_to_src_encoded = model.generate(
                input_ids=tokens_dict_src['input_ids'].to(device),
                attention_mask=tokens_dict_src['attention_mask'].to(device),
                num_return_sequences=nbest,
                num_beams=nbest,
                do_sample=True,
                top_k=20,
                temperature=2.0,
            )

            # Decode src to pivot tranlations
            trg_to_src_decoded = [
                tokenizer.decode(t, skip_special_tokens=True)
                for t in trg_to_src_encoded
            ]

            torch.cuda.empty_cache()

            trg_to_src_translations[-1].extend(trg_to_src_decoded)

    return trg_to_src_translations


def parse_file(forward_file):
    forward_file_path = os.path.abspath(forward_file)
    TEST_SENTENCE_STR = "Test Sentence: "
    TAB = "\t"
    last_src_sent = ""
    src_to_trg = {}

    for line in open(forward_file_path, 'r'):
        if TEST_SENTENCE_STR in line:
            line_copy = line.strip().rstrip("\n")
            src_to_trg[line_copy] = []
            last_src_sent = line_copy
        elif TAB in line:
            # get the prev tranlations
            prev_translations = src_to_trg[last_src_sent]
            # add the curr translation and clean it
            translation = line.strip().rstrip("\n")
            prev_translations.append(translation)
            src_to_trg[last_src_sent] = prev_translations

    return src_to_trg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate from forward model to pivot model and back given n number of hypothesis"
    )

    parser.add_argument('--src', help='Name of language for the forward model')
    parser.add_argument('--trg', help='Name of language for the pivot model')
    parser.add_argument(
        '--nbest',
        help='Number of sequences to return from the backward model')
    parser.add_argument('--output',
                        help='Directory where the sequences will be saved')
    parser.add_argument('--src_to_trg_translations', help='Data to produce translation from trg to src')
    parser.add_argument('--device', help='GPU where the script should run')
    args = parser.parse_args()

    src = args.src
    trg = args.trg
    nbest = int(args.nbest)
    output_dir = args.output
    test_data_loc = args.src_to_trg_translations

    if args.device == "cpu":
        device = "cpu"
    else:
        device = torch.device(
            f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    logger.info('Getting test data from: {}'.format(test_data_loc))
    # load all test data
    test_data = parse_file(test_data_loc)

    logger.info("Device in use: {}".format(device))

    logger.info('Translating test data from {} to {}'.format(trg, src))
    src_to_trg_translations = translate_from_trg_to_src(src, trg, test_data,
                                        nbest, device)

    logger.info('Saving translations to {}'.format(
        os.path.abspath(
            os.path.join(
                output_dir,
                "{}-{}-top{}translations.txt".format(src, trg,
                                                     nbest * nbest)))))
    save_nbest(src_to_trg_translations, test_data, src, trg, nbest, output_dir)
