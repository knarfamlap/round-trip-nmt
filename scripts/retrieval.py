import argparse
import os
import numpy as np
from logzero import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import extract_generated_distractors_from_file, find_retrieval


def plot_retrieval(num_retrieved_list, num_best_list):
    # plot

    cols = list(mcolors.TABLEAU_COLORS.values())
    plt.figure(figsize=(9, 6), dpi=160)
    plt.plot(num_best_list, num_retrieved_list['en-fr-en'],
             marker='^', markersize=10, color=cols[0], label='en-de-en')
    plt.plot(num_best_list, num_retrieved_list['en-zh-en'],
             marker='s', markersize=10, color=cols[1], label='en-ru-en')
    plt.plot(num_best_list, num_retrieved_list['both'],
             marker='p', markersize=10, color=cols[2], label='Both')
    plt.xlabel('Number of round-trip translations used', fontsize=24)
    plt.ylabel('# Gold distractors retrieved', fontsize=24)
    plt.title('Retrieval of distractors', fontsize=18)
    plt.xticks(num_best_list, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.4)
    legend = plt.legend(title='Round trip translation', fontsize=24)
    legend.get_title().set_fontsize('24')
    plt.tight_layout()
    plt.savefig('[PLOT]num_retrieved_esl_lounge_exercises_endeen_enruen.pdf')
    plt.show()
    plt.close()


def plot_retrieval_and_num_cands(
        _system_names,
        _num_retrieved_list,
        _retrieval_save_path="./plot.pdf",
        _num_distractors_save_path=None,
):
    # plot
    plt.figure(figsize=(6, 5), dpi=160)
    plt.bar(np.arange(len(_system_names)), _num_retrieved_list, color='xkcd:gold')
    # plt.bar([np.arange(len(_system_names))[-1]], [_num_retrieved_list[-1]], color='xkcd:gold', hatch='x')
    for i in np.arange(len(_system_names)):
        plt.text(i, _num_retrieved_list[i], f'({100 * _num_retrieved_list[i]/(144 * 3):0.1f}%)\n{_num_retrieved_list[i]:0.0f}',
                 horizontalalignment='center',
                 verticalalignment='bottom',
                 fontsize=14)
    plt.xlabel('Round-trip system', fontsize=14)
    plt.ylabel('# Gold dstractors retrieved', fontsize=14)
    # plt.title('Retrieval of distractors\n(no filtering)', fontsize=18)
    plt.xticks(np.arange(len(_system_names)), _system_names, rotation=30, fontsize=14, horizontalalignment='right')
    plt.ylim([0, max(_num_retrieved_list) + 30])
    plt.grid(alpha=0.4, axis='y')
    plt.tight_layout()
    if _retrieval_save_path is not None:
        plt.savefig(_retrieval_save_path)
    plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Creates retrieval graph for a given set of tranlations from multiple langugages.")
    parser.add_argument('-d', "--gold_distractors",
                        help="location of gold distractors")
    parser.add_argument(
        '-k', "--keywords", help="location where the solutions are for each clozer item")
    parser.add_argument('-t', "--translations", type=list,
                        nargs="*", help="location of translations")
    parser.add_argument(
        '-p', "--prefix", type=list, nargs="*", help="Prefix of each file to look for. Must be the same length as translation flag")
    parser.add_argument(
        '-s', "--save", help="Location of where to save the plot")
    args = parser.parse_args()

    gold_distractors_path = os.path.abspath(args.gold_distractors)
    solutions_path = os.path.abspath(args.keywords)
    paths_to_translations = list(map(lambda x: "".join(x), args.translations))
    prefixs = list(map(lambda x: "".join(x), args.prefix))
    save_dir = os.path.abspath(args.save)

    logger.info("Fetching solutions from: {}".format(solutions_path))
    # process solutions
    solutions = []
    for line in open(solutions_path, 'r'):
        solutions.append(line.strip().strip('\n'))

    logger.info("Fetching gold distractors from: {}".format(
        gold_distractors_path))
    # process gold distractors
    gold_distractors = []
    for line in open(gold_distractors_path, 'r'):
        gold_distractors.extend(line.strip().split(','))

    # preprocess paths for translations
    paths_to_translations = list(
        map(lambda x: os.path.abspath(x), paths_to_translations))

    num_cloze_items = len(solutions)

    # calculate the number of retrivals
    retrieved = {}
    for prefix, path in zip(prefixs, paths_to_translations):
        generated_distractors = []
        logger.info("Calculating retrival from: {}".format(path))

        for i in range(0, num_cloze_items):
            file_path = os.path.join(
                path, prefix + "-top100translations-{}.txt.align".format(i + 1))
            generated_distractors.extend(extract_generated_distractors_from_file(
                file_path, solutions[i]))
        # save tuple: (prefix, num_retrieval)
        retrieved[prefix] = find_retrieval(
            gold_distractors, generated_distractors)

    retrieved['both'] = retrieved['en-fr-en'] + retrieved['en-zh-en']
    retrieved = [retrieved['en-fr-en'], retrieved['en-zh-en']]
    logger.info("Creating Plot")
    plot_retrieval_and_num_cands(['en-fr-en-top100', 'en-zh-en-top100'], retrieved)
    logger.info("Done!")
