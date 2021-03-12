import argparse
import os
import ast
import torch
from logzero import logger
from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from typing import List



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


def save_nbest(rt_translations, test_sents, nbest, file_name, output_dir):
    output_dir = os.path.abspath(output_dir)
    assert os.path.exists(output_dir)

    with open(os.path.join(
            output_dir,
            file_name),
        'w',
            encoding='utf-8') as f:
        for test, translations in zip(test_sents, rt_translations):
            f.write(89 * '-' + '\n')
            f.write('Test Sentence: {}\n'.format(test))
            f.write('Top {} Translations\n'.format(nbest * nbest))
            for translation in translations:
                f.write('\t{}\n'.format(translation))
            f.write(89 * '-' + '\n')
    # return the path where the translations were saved
    return os.path.join(output_dir, file_name)


def find_retrieval(li_a: List[str], li_b: List[str]) -> int:
    """ 
    Compute Retrieval of gold distractors. li_a is the list of gold distractors 
    and li_b is the list of generated distractors
    """
    intersection = set(li_a) & set(li_b)

    return len(intersection)


def extract_generated_distractors_from_file(file_path, keyword):
    """ 
    file_path: location where the file of aligned tokens is stored. Should have extension .aligned
    keyword: The solution of the cloze item
    
    Returns: A list of generated distractors. The distractors that are matched with the keyword
     """
    TAB = "\t"

    distractors = []
    for line in open(file_path, 'r'):
        if TAB in line:

            aligned_tokens_as_str = line.strip().strip(
                "\n")
            
            aligned_tokens_as_tuples = ast.literal_eval(aligned_tokens_as_str)

            for src_token, trg_token in aligned_tokens_as_tuples:
                if src_token == keyword:
                    distractors.append(trg_token)
                    break

    return distractors


    