import argparse
import os
import torch
from logzero import logger
from utils import load_model_and_tokenizer, save_nbest
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer


def translate_sent_to_nbest(nbest, sentence, model, tokenizer, device="cuda:0"):
    """ 
    nbest: the number of translation to produce per given sentenc 
    sentence: A sentence that we wish to tranlate
    translations: arr where the generated tranlations will be appended to
    model: model for nmt tranlation
    tokenizer: tokinzer for preprocessing the given sentence
    """

    translations = []

    tokenized_sent = tokenizer.prepare_seq2seq_batch(
        sentence, return_tensors="pt"
    )

    encoded_translations = model.generate(
        input_ids=tokenized_sent['input_ids'].to(device),
        attention_mask=tokenized_sent['attention_mask'].to(device),
        num_return_sequences=nbest,
        num_beams=nbest,
        do_sample=True,
        top_k=20,
        temperature=2.0,
    )

    decoded_translations = [
        tokenizer.decode(t, skip_special_tokens=True) for t in encoded_translations
    ]

    translations.extend(decoded_translations)

    torch.cuda.empty_cache()

    return translations


def translate(src, trg, sentences, nbest, direction="f", device="cuda:0"):

    logger.info('Getting {}-{} model and its tokenizer'.format(src, trg))
    model, tokenizer = load_model_and_tokenizer(
        src, trg, device=device)

    translations = []

    if direction == "f":
        for sent in tqdm(sentences):
            translations.append([])
            translations[-1].extend(translate_sent_to_nbest(nbest,
                                                            sent, model, tokenizer, device))
    elif direction == "b":
        for trg_sent_batch in tqdm(sentences):
            translations.append([])
            for sent in tqdm(trg_sent_batch):
                translations[-1].extend(translate_sent_to_nbest(nbest,
                                                                sent, model, tokenizer, device))

    return translations


def parse_file_for_backward(forward_file):
    forward_file_path = os.path.abspath(forward_file)
    TEST_SENTENCE_STR = "Test Sentence: "
    TAB = "\t"
    last_src_sent = ""
    src_to_trg = {}

    for line in open(forward_file_path, 'r'):
        if TEST_SENTENCE_STR in line:
            line_copy = line.strip().rstrip("\n")[len(TEST_SENTENCE_STR):]
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

def init_job(src, trg, nbest, direction, test_sentences_loc, device="cuda:0"):

    if direction == "f":
        # load test_sentence for forward translations
        test_sentences_path = os.path.abspath(test_sentences_loc)
        test_sentences = open(test_sentences_path, 'r').read().split('\n')

        logger.info('Translating test sentences from {} to {}'.format(src, trg))
        translations = translate(
            src, trg, test_sentences, nbest, direction="f", device=device)
        translations_file_name = "{}-{}-top{}translations.txt".format(src, trg,
                                                                      nbest * nbest)

    elif direction == "b":
        # load test sentences for backward tranlations
        test_sentences = parse_file_for_backward(test_sentences_loc)
        trg_sentences_in_batches = test_sentences.values()

        logger.info('Translating test sentences from {} to {}'.format(trg, src))
        translations = translate(
            trg, src, trg_sentences_in_batches, nbest, direction="b", device=device)
        translations_file_name = "{}-{}-{}-top{}translations.txt".format(src, trg, src,
                                                                         nbest * nbest)

    logger.info('Saving translations to {}'.format(
        os.path.abspath(
            os.path.join(
                output_dir,
                translations_file_name))))
    save_nbest(translations, test_sentences, nbest,
               translations_file_name, output_dir)


    return translations_file_name



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translate from src model to trg model and back given n number of hypothesis for each hypothesis in src_to_trg_translations"
    )

    parser.add_argument(
        '-s', '--src', type=str, help='Name of language for the forward model')
    parser.add_argument(
        '-t', '--trg', action="append", type=list, help='Name of language for the pivot model')
    parser.add_argument(
        '--nbest',
        help='Number of sequences to return from the backward model')
    parser.add_argument('--output',
                        help='Directory where the sequences will be saved')
    parser.add_argument('--sentences',
                        help='Sentences to produce translation from trg to src')
    parser.add_argument('--device', help='GPU where the script should run')
    parser.add_argument('--forward', action="store_true",
                        help="If true then it will evaluate tranlations for src to trg")
    parser.add_argument('--backward', action="store_true",
                        help="If true then it will evaluate translations for trg to src")
    parser.add_argument("--both", action="store_true", help="If true then it will translate for both directions")
    args = parser.parse_args()

    src = args.src
    trg_langs = args.trg
    nbest = int(args.nbest)
    output_dir = args.output
    test_sentences_loc = args.sentences
    translate_forward = args.forward
    translate_backward = args.backward
    translate_both = args.both

    if args.device == "cpu":
        device = "cpu"
    else:
        device = torch.device(
            f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    logger.info("Device in use: {}".format(device))
    logger.info('Getting test sentences from: {}'.format(test_sentences_loc))
    
    if translate_both:
        for trg in trg_langs:
            translation_file_path = init_job(src, trg, nbest, "f", test_sentences_loc, device) 
            init_job(src, trg, nbest, "b", translation_file_path, device)

    elif translate_forward:
        for trg in trg_langs:
            init_job(src, trg, nbest, "f", test_sentences_loc, device) 
    elif translate_backward:
        for trg in trg_langs:
            init_job(src, trg, nbest, "b", test_sentences_loc, device) 