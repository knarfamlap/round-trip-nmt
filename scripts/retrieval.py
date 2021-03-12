import argparse
import os
from logzero import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from utils import extract_generated_distractors_from_file, find_retrieval


def plot_retrieval(num_retrieved_list, num_best_list):
    # plot

    cols = list(mcolors.TABLEAU_COLORS.values())
    plt.figure(figsize=(9, 6), dpi=160)
    plt.plot(num_best_list, num_retrieved_list['en-fr-en'], marker='^', markersize=10, color=cols[0], label='en-de-en')
    plt.plot(num_best_list, num_retrieved_list['en-zh-en'], marker='s', markersize=10, color=cols[1], label='en-ru-en')
    plt.plot(num_best_list, num_retrieved_list['both'], marker='p', markersize=10, color=cols[2], label='Both')
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
        solutions.append(line.strip())

    logger.info("Fetching gold distractors from: {}".format(gold_distractors_path))
    # process gold distractors
    gold_distractors = []
    for line in open(gold_distractors_path, 'r'):
        gold_distractors.append(line.strip().split(','))
    
    # preprocess paths for translations
    paths_to_translations = list(
        map(lambda x: os.path.abspath(x), paths_to_translations))

    num_cloze_items = len(gold_distractors)
    # calculate the number of retrivals
    retrieved = {}
    for prefix, path in zip(prefixs, paths_to_translations):
        generated_distractors = []
        logger.info("Calculating retrival from: {}".format(path))
        for i in range(1, num_cloze_items + 1):
            file_path = os.path.join(path, prefix + "-top100translations-{}.txt".format(i))
            generated_distractors.append(
                extract_generated_distractors_from_file(file_path, solutions[i - 1]))
        # save tuple: (prefix, num_retrieval)
        retrieved[prefix] = find_retrieval(gold_distractors, generated_distractors)

    retrieved['Both'] = retrieved['en-fr-en'] + retrieved['en-zh-en']
    logger.info("Creating Plot")
    plot_retrieval(retrieved, [10])
    logger.info("Done!")

    

            
