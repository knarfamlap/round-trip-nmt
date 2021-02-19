from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from logzero import logger
import os
import torch
import argparse


def get_num_params(model):
    # get total number of parameters in model
    return sum(param.numel() for param in model.parameters()
               if param.requires_grad)


def load_model_and_tokenizer(src, trg, device='cuda'):
    # parse model name
    model_name = 'Helsinki-NLP/opus-mt-{}-{}'.format(src, trg)
    # load the tokenizer for the model
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    # load the model
    model = MarianMTModel.from_pretrained(model_name)
    # change to cuda if possible
    model = model.to(device)
    # calculate number of params
    num_params = get_num_params(model)

    logger.info('Number of Parameters in Model: {}'.format(num_params))
    return model, tokenizer


def translate(fwrd_model,
              fwrd_tokenizer,
              bwrd_model,
              bwrd_tokenizer,
              sentences,
              nbest,
              device='cuda'):
    rt_translations = []

    for te in tqdm(sentences):
        rt_translations.append([])
        tokens_dict_src = fwrd_tokenizer.prepare_seq2seq_batch(
            te, return_tensors="pt")
        # Translate from src to pivot language. Sentences are encoded
        src_to_pivot = fwrd_model.generate(
            input_ids=tokens_dict_src['input_ids'].to(device),
            attention_mask=tokens_dict_src['attention_mask'].to(device),
            num_return_sequences=nbest,
            num_beams=nbest,
            do_sample=True,
            top_k=20,
            temperature=2.0,
        )
        # Decode src to pivot tranlations
        src_to_pivot_txt = [
            fwrd_tokenizer.decode(t, skip_special_tokens=True)
            for t in src_to_pivot
        ]

        for tr in tqdm(src_to_pivot_txt):
            tokens_dict_pivot = bwrd_tokenizer.prepare_seq2seq_batch(
                tr, return_tensors="pt")
            # Translate pivot to src. Sentences are encoded
            pivot_to_src = bwrd_model.generate(
                input_ids=tokens_dict_pivot['input_ids'].to(device),
                attention_mask=tokens_dict_pivot['attention_mask'].to(device),
                num_return_sequences=nbest,
                num_beams=nbest,
                do_sample=True,
                top_k=20,
                temperature=2.0,
            )
            # Decode pivot to src translations
            pivot_to_src_txt = [
                bwrd_tokenizer.decode(t, skip_special_tokens=True)
                for t in pivot_to_src
            ]
            # add to round trip tranlation list
            rt_translations[-1].extend(pivot_to_src_txt)

    logger.info("Number of round trip translations {}".format(
        len(rt_translations)))
    return rt_translations


def save_nbest(rt_translations, test_sents, src, trg, nbest, output_dir):
    output_dir = os.path.abspath(output_dir)
    assert os.path.exists(output_dir)

    with open(os.path.join(
            output_dir,
            "{}-{}-top{}translations.txt".format(src, trg, nbest * nbest)),
              'w',
              encoding='utf-8') as f:
        for test, translations in zip(test_sents, rt_translations):
            f.write(89 * '-' + '\n')
            f.write('Test Sentence: {}\n'.format(test))
            f.write('Top {} Translations\n'.format(nbest * nbest))
            for translation in translations:
                f.write('\t{}\n'.format(translation))
            f.write(89 * '-' + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Translate from forward model to pivot model and back given n number of hypothesis"
    )

    parser.add_argument('--src', help='Name of language for the forward model')
    parser.add_argument('--trg', help='Name of language for the pivot model')
    parser.add_argument(
        '--nbest',
        help='Number of sequences to return from the backward model')
    parser.add_argument('--output',
                        help='Directory where the sequences will be saved')
    parser.add_argument('--test', help='Test data to produce translation')
    args = parser.parse_args()

    src = args.src
    trg = args.trg
    nbest = int(args.nbest)
    output_dir = args.output
    test_data_loc = args.test

    test_data_loc = os.path.abspath(test_data_loc)
    logger.info('Getting test data from: {}'.format(test_data_loc))
    # load all test data
    test_data = open(test_data_loc, 'r').read().split('\n')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device is use: {}".format(device))

    logger.info('Getting {}-{} model and its tokenizer'.format(src, trg))
    src_trg_model, src_trg_tokenizer = load_model_and_tokenizer(src, trg, device)

    logger.info('Getting {}-{} model and its tokenizer'.format(trg, src))
    trg_src_model, trg_src_tokenizer = load_model_and_tokenizer(trg, src, device)

    logger.info('Translating test data')
    rt_translations = translate(src_trg_model, src_trg_tokenizer,
                                trg_src_model, trg_src_tokenizer, test_data,
                                nbest, device)

    logger.info('Saving translations to {}'.format(
        os.path.abspath(
            os.path.join(
                output_dir,
                "{}-{}-top{}translations.txt".format(src, trg,
                                                     nbest * nbest)))))
    save_nbest(rt_translations, test_data, src, trg, nbest, output_dir)
