import argparse
import os
import torch
from logzero import logger
from utils import load_model_and_tokenizer, save_nbest
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


def translate(
        src,
        trg,
        sentences,
        nbest,
        device='cuda:0'):

    logger.info('Getting {}-{} model and its tokenizer'.format(src, trg))
    model, tokenizer = load_model_and_tokenizer(
        src, trg, device=device)

    src_to_trg_translations = []
    for src_sent in tqdm(sentences):
        src_to_trg_translations.append([])
        tokens_dict_src = tokenizer.prepare_seq2seq_batch(
            src_sent, return_tensors="pt"
        )

        # Translate from src to pivot language. Sentences are encoded
        src_to_trg_encoded = model.generate(
            input_ids=tokens_dict_src['input_ids'].to(device),
            attention_mask=tokens_dict_src['attention_mask'].to(device),
            num_return_sequences=nbest,
            num_beams=nbest,
            do_sample=True,
            top_k=20,
            temperature=2.0,
        )

        # Decode src to pivot tranlations
        src_to_trg_decoded = [
            tokenizer.decode(t, skip_special_tokens=True)
            for t in src_to_trg_encoded
        ]

        torch.cuda.empty_cache()

        src_to_trg_translations[-1].extend(src_to_trg_decoded)

    return src_to_trg_translations

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
    parser.add_argument('--test', help='Test data to produce translation')
    parser.add_argument('--device', help='GPU where the script should run')
    args = parser.parse_args()

    src = args.src
    trg = args.trg
    nbest = int(args.nbest)
    output_dir = args.output
    test_data_loc = args.test

    if args.device == "cpu":
        device = "cpu"
    else:
        device = torch.device(
            f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    test_data_loc = os.path.abspath(test_data_loc)
    logger.info('Getting test data from: {}'.format(test_data_loc))
    # load all test data
    test_data = open(test_data_loc, 'r').read().split('\n')

    logger.info("Device in use: {}".format(device))

    logger.info('Translating test data from {} to {}'.format(src, trg))
    src_to_trg_translations = translate(src, trg, test_data,
                                        nbest, device)

    logger.info('Saving translations to {}'.format(
        os.path.abspath(
            os.path.join(
                output_dir,
                "{}-{}-top{}translations.txt".format(src, trg,
                                                     nbest * nbest)))))
    save_nbest(src_to_trg_translations, test_data, src, trg, nbest, output_dir)