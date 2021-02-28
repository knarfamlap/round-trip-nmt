import argparse
import os
import torch
from logzero import logger
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer

def get_num_params(model):
    # get total number of parameters in model
    return sum(param.numel() for param in model.parameters()
               if param.requires_grad)

def load_model_and_tokenizer(src, trg, device='cuda:0'):
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