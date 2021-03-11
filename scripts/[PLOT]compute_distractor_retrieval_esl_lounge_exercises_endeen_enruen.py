import json
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

if __name__ == "__main__":

    PREPOSITION_SAMPLES_IDS = [9, 122]
    sentences_file_path = os.path.join(
        '../data',
        'cloze_task_data',
        'esl_lounge',
        'all_sentences.txt'
    )
    solutions_file_path = os.path.join(
        '../data',
        'cloze_task_data',
        'esl_lounge',
        'all_solutions.txt'
    )
    distractors_file_path = os.path.join(
        '../data',
        'cloze_task_data',
        'esl_lounge',
        'all_distractors.txt'
    )

    assert os.path.isfile(sentences_file_path)
    assert os.path.isfile(solutions_file_path)
    assert os.path.isfile(distractors_file_path)

    data_dict = {}
    for file_path, name in zip(
            [sentences_file_path, solutions_file_path, distractors_file_path],
            ['sentences', 'solutions', 'distractors']
    ):
        with open(file_path, 'r', encoding='utf-8') as f:
            data_dict[name] = list(map(
                str.strip, f.readlines()
            ))
            if name == 'distractors':
                data_dict[name] = list(map(
                    lambda x: x.split(','),
                    data_dict[name]
                ))

    generated_distractors = {
        'endeen': [],
        'enruen': []
    }

    """
    en-de-en
    """
    for idx in range(72):
        generated_distractors_file_path = os.path.join(
            f'../results',
            f'sample_wise_results_part_a',
            f'aligned_tokens_exercises_idx{idx}_endeen.json'
        )
        print(generated_distractors_file_path)
        assert os.path.isfile(generated_distractors_file_path)

        with open(generated_distractors_file_path, 'r', encoding='utf-8') as f:
            generated_distractors['endeen'].append(json.load(f))

    for idx in range(72):
        generated_distractors_file_path = os.path.join(
            f'../results',
            f'sample_wise_results_part_b',
            f'aligned_tokens_exercises_idx{idx}_endeen.json'
        )
        print(generated_distractors_file_path)
        assert os.path.isfile(generated_distractors_file_path)

        with open(generated_distractors_file_path, 'r', encoding='utf-8') as f:
            generated_distractors['endeen'].append(json.load(f))

    """
    en-ru-en
    """
    generated_distractors_file_path = os.path.join(
        f'../results',
        f'aligned_tokens_exercises_enruen.json'
    )
    print(generated_distractors_file_path)
    assert os.path.isfile(generated_distractors_file_path)

    with open(generated_distractors_file_path, 'r', encoding='utf-8') as f:
        generated_distractors['enruen'] = json.load(f)

    assert len(generated_distractors['endeen']) == \
           len(generated_distractors['enruen'])

    num_best_list = list(range(100, 1000 + 1, 100))
    num_retrieved_lists = {
        'endeen': [],
        'enruen': [],
        'both': [],
    }
    num_distractors_lists = {
        'endeen': [],
        'enruen': [],
        'both': [],
    }
    solution_found_lists = {
        'endeen': [],
        'enruen': [],
        'both': [],
    }
    atleast1_lists = {
        'endeen': [],
        'enruen': [],
        'both': [],
    }
    atleast2_lists = {
        'endeen': [],
        'enruen': [],
        'both': [],
    }
    atleast3_lists = {
        'endeen': [],
        'enruen': [],
        'both': [],
    }

    for k in num_best_list:
        num_retrieved = {
            'endeen': 0,
            'enruen': 0,
            'both': 0
        }
        num_distractors = {
            'endeen': 0,
            'enruen': 0,
            'both': 0
        }
        solution_found = {
            'endeen': True,
            'enruen': True,
            'both': True
        }

        atleast = {
            'endeen': [],
            'enruen': [],
            'both': []
        }

        for sample_idx in range(len(generated_distractors['endeen'])):
            if sample_idx in PREPOSITION_SAMPLES_IDS:
                continue
            solution = data_dict['solutions'][sample_idx]
            aligned_tokens_dicts = {
                'endeen': generated_distractors['endeen'][sample_idx]['aligned_tokens_dict'],
                'enruen': generated_distractors['enruen'][sample_idx]['aligned_tokens_dict']
            }
            aligned_tokens = {
                'endeen': aligned_tokens_dicts['endeen'][solution]['aligned_tokens'],
                'enruen': aligned_tokens_dicts['enruen'][solution]['aligned_tokens']
            }
            # sanity check
            for aligned_tokens_dict in aligned_tokens_dicts.values():
                if len(aligned_tokens_dict) != 1:
                    raise AssertionError
                if solution != list(aligned_tokens_dict.keys())[0]:
                    raise AssertionError
            for al_toks in aligned_tokens.values():
                if k > len(al_toks):
                    raise AssertionError
            num_retrieved['endeen'] += len(
                set(aligned_tokens['endeen'][:k]) & set(data_dict['distractors'][sample_idx]))
            atleast['endeen'].append(len(
                set(aligned_tokens['endeen'][:k]) & set(data_dict['distractors'][sample_idx])))
            num_retrieved['enruen'] += len(
                set(aligned_tokens['enruen'][:k]) & set(data_dict['distractors'][sample_idx]))
            atleast['enruen'].append(len(
                set(aligned_tokens['enruen'][:k]) & set(data_dict['distractors'][sample_idx])))
            num_retrieved['both'] += len(set(aligned_tokens['endeen'][:k] + aligned_tokens['enruen'][:k]) & set(
                data_dict['distractors'][sample_idx]))
            atleast['both'].append(len(set(aligned_tokens['endeen'][:k] + aligned_tokens['enruen'][:k]) & set(
                data_dict['distractors'][sample_idx])))
            num_distractors['endeen'] += len(set(aligned_tokens['endeen'][:k]))
            num_distractors['enruen'] += len(set(aligned_tokens['enruen'][:k]))
            num_distractors['both'] += len(set(aligned_tokens['endeen'][:k] + aligned_tokens['enruen'][:k]))
            solution_found['endeen'] = solution_found['endeen'] and (solution in set(aligned_tokens['endeen'][:k]))
            solution_found['enruen'] = solution_found['enruen'] and (solution in set(aligned_tokens['enruen'][:k]))
            solution_found['both'] = solution_found['both'] and (
                    solution in set(aligned_tokens['endeen'][:k] + aligned_tokens['enruen'][:k]))
        for item in ['endeen', 'enruen', 'both']:
            num_retrieved_lists[item].append(num_retrieved[item])
            num_distractors_lists[item].append(num_distractors[item])
            solution_found_lists[item].append(solution_found[item])
            atleast1_lists[item].append(
                100 * len(list(filter(lambda x: x >= 1, atleast[item]))) / (len(generated_distractors['endeen'])
                                                                            - len(PREPOSITION_SAMPLES_IDS)))
            atleast2_lists[item].append(
                100 * len(list(filter(lambda x: x >= 2, atleast[item]))) / (len(generated_distractors['endeen'])
                                                                            - len(PREPOSITION_SAMPLES_IDS)))
            atleast3_lists[item].append(
                100 * len(list(filter(lambda x: x >= 3, atleast[item]))) / (len(generated_distractors['endeen'])
                                                                            - len(PREPOSITION_SAMPLES_IDS)))
    print('num_best_list:', num_best_list)
    print('num_retrieved_list:', num_retrieved_lists)
    print('num_distractors_list:', num_distractors_lists)
    print(solution_found_lists)

    cols = list(mcolors.TABLEAU_COLORS.values())

    # plot
    plt.figure(figsize=(9, 6), dpi=160)
    plt.plot(num_best_list,
             [num / (len(generated_distractors['endeen']) - len(PREPOSITION_SAMPLES_IDS)) for num in num_distractors_lists['endeen']],
             marker='^', markersize=10, color=cols[0], label='en-de-en')
    plt.plot(num_best_list,
             [num / (len(generated_distractors['enruen']) - len(PREPOSITION_SAMPLES_IDS)) for num in num_distractors_lists['enruen']],
             marker='s', markersize=10, color=cols[1], label='en-ru-en')
    plt.plot(num_best_list, [num / (len(generated_distractors['endeen']) - len(PREPOSITION_SAMPLES_IDS)) for num in num_distractors_lists['both']],
             marker='p', markersize=10, color=cols[3], label='Both')
    plt.xlabel('Number of round-trip translations used', fontsize=24)
    plt.ylabel('# Automatic \ndistractors / question', fontsize=24)
    # plt.title('Number of candidate items', fontsize=16)
    plt.xticks(num_best_list, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.4)
    legend = plt.legend(title='Round trip translation', fontsize=24)
    legend.get_title().set_fontsize('24')
    plt.tight_layout()
    plt.savefig('[PLOT]num_candidate_items_esl_lounge_exercises_endeen_enruen.pdf')
    plt.show()
    plt.close()

    plt.figure(figsize=(9, 6), dpi=160)
    plt.plot(num_best_list, num_retrieved_lists['endeen'], marker='^', markersize=10, color=cols[0], label='en-de-en')
    plt.plot(num_best_list, num_retrieved_lists['enruen'], marker='s', markersize=10, color=cols[1], label='en-ru-en')
    plt.plot(num_best_list, num_retrieved_lists['both'], marker='p', markersize=10, color=cols[3], label='Both')
    plt.xlabel('Number of round-trip translations used', fontsize=24)
    plt.ylabel('# Gold distractors retrieved', fontsize=24)
    # plt.title('Retrieval of distractors', fontsize=18)
    plt.xticks(num_best_list, fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(alpha=0.4)
    legend = plt.legend(title='Round trip translation', fontsize=24)
    legend.get_title().set_fontsize('24')
    plt.tight_layout()
    plt.savefig('[PLOT]num_retrieved_esl_lounge_exercises_endeen_enruen.pdf')
    plt.show()
    plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=160)
    plt.plot(num_best_list, atleast1_lists['endeen'], color=cols[0], linestyle='-', marker='^', markersize=10,
             label='en-de-en, Atleast 1')
    plt.plot(num_best_list, atleast1_lists['enruen'], color=cols[1], linestyle='-', marker='s', markersize=10,
             label='en-ru-en, Atleast 1')
    plt.plot(num_best_list, atleast1_lists['both'], color=cols[3], linestyle='-', marker='p', markersize=10,
             label='Both, Atleast 1')
    plt.plot(num_best_list, atleast2_lists['endeen'], color=cols[0], linestyle='--', marker='^', markersize=10,
             label='en-de-en, Atleast 2')
    plt.plot(num_best_list, atleast2_lists['enruen'], color=cols[1], linestyle='--', marker='s', markersize=10,
             label='en-ru-en, Atleast 2')
    plt.plot(num_best_list, atleast2_lists['both'], color=cols[3], linestyle='--', marker='p', markersize=10,
             label='Both, Atleast 2')
    plt.plot(num_best_list, atleast3_lists['endeen'], color=cols[0], linestyle=':', marker='^', markersize=10,
             label='en-de-en, All 3')
    plt.plot(num_best_list, atleast3_lists['enruen'], color=cols[1], linestyle=':', marker='s', markersize=10,
             label='en-ru-en, All 3')
    plt.plot(num_best_list, atleast3_lists['both'], color=cols[3], linestyle=':', marker='p', markersize=10,
             label='Both, All 3')
    plt.xlabel('Number of round-trip translations used', fontsize=24)
    plt.ylabel('Cloze items (%)', fontsize=24)
    # plt.title('Percentage of items', fontsize=18)
    plt.xticks(num_best_list, fontsize=16)
    plt.yticks(range(0, 101, 10), [f'{val:0.0f}%' for val in range(0, 101, 10)], fontsize=16)
    plt.grid(alpha=0.4)
    plt.legend(ncol=3, fontsize=13, loc='upper center')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('[PLOT]atleast_esl_lounge_exercises_endeen_enruen.pdf')
    plt.show()
    plt.close()
