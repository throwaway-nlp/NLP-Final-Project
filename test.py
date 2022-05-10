import typing

import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification, \
    AutoModelForQuestionAnswering, Trainer, TrainingArguments, HfArgumentParser
from helpers import prepare_dataset_nli, prepare_train_dataset_qa, \
    prepare_validation_dataset_qa, QuestionAnsweringTrainer, compute_accuracy, in_sentence
import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='eval/shuffle_1.00_8/squad')
parser.add_argument('--cw', action='store_false')
parser.add_argument('--g', action='store_true')
args = parser.parse_args()
print("Selected: ", args.dir)

if args.g:
    data = datasets.load_dataset('squad')

    train_answers = []
    train_lengths = []

    for ex in data['train']:
        answers_start = ex['answers']['answer_start']
        total = 0.0
        for i in range(len(answers_start)):
            a_start = answers_start[i]
            context = ex['context']
            actual_sentence, _ = in_sentence(context, a_start)
            total += actual_sentence
        train_answers.append((total/len(answers_start)))
        train_lengths.append(_)

    train_answers = np.array(train_answers)
    train_lengths = np.array(train_lengths)

    eval_answers = []
    eval_lengths = []

    for ex in data['validation']:
        answers_start = ex['answers']['answer_start']
        total = 0.0
        for i in range(len(answers_start)):
            a_start = answers_start[i]
            context = ex['context']
            actual_sentence, _ = in_sentence(context, a_start)
            total += actual_sentence
        eval_answers.append((total/len(answers_start)))
        eval_lengths.append(_)

    eval_answers = np.array(eval_answers)
    eval_lengths = np.array(eval_lengths)

    file_name = 'eval_metrics.json'
    dir = args.dir
    file_path = os.path.join(dir, file_name)

    data = json.load(open(file_path))
    eval_predicted_answers_w = np.array(data['eval_predicted_answers_w'])
    eval_predicted_answers_c = np.array(data['eval_predicted_answers_c'])
    corrected_answers = np.array(data['eval_corrected_answers'])

    # for i in range(len(eval_answers)):
    #     mean = np.mean(np.array(eval_answers[i]))
    #     eval_answers[i] = mean
    # eval_answers = np.array(eval_answers)

    def stats(arr):
        mean, std = np.mean(np.array(arr)), np.std(np.array(arr))
        return mean, std


    fig, axs = plt.subplots(2,1,tight_layout=True)


    def distribution(arr, axs, i, title, label=None):
        # color = 'Blue'
        # if i > 1:
        #     color = 'Orange'
        # i = i % 2
        if label:
            axs[i].hist(arr, bins=10, label=label, range=(1, 11), rwidth=1,)
        else:
            axs[i].hist(arr, bins=10, range=(1, 11), rwidth=1,)
        axs[i].title.set_text(title)
        # axs[i].set_xlim(0, 10)
        axs[i].set_xticks((np.arange(10) + 1))
    #
    plt.xlabel('Sentence containing correct answer')
    fig.text(0.015, .5, 'Frequency', ha='center', va='center', rotation='vertical')

    eval_set_mean, eval_set_std = stats(eval_answers)
    eval_length_mean, eval_length_std = stats(eval_lengths)
    distribution(eval_answers, axs, 0, "SQuAD Test Set Correct Sentence Distribution")
    train_set_mean, train_set_std = stats(train_answers)
    train_length_mean, train_length_std = stats(train_lengths)
    distribution(train_answers, axs, 1, "SQuAD Train Set Correct Sentence Distribution")

    plt.savefig(os.path.join(args.dir, 'set_dists.png'), dpi=100)
    plt.clf()

    w_set_mean, w_set_std = stats(eval_predicted_answers_w)
    c_set_mean, c_set_std = stats(eval_predicted_answers_c)
    fig, axs = plt.subplots(2,1,tight_layout=True)
    distribution(eval_predicted_answers_w, axs, 0, "Incorrect Predicted Sentence Distribution", "predicted")
    # distribution(eval_predicted_answers_c, axs, 1, "Correct Predicted Sentence Distribution")
    distribution(corrected_answers, axs, 1, "Actual Sentence Distribution", "actual")
    plt.xlabel('Sentence containing answer')
    fig.text(.01, .5, 'Frequency', ha='center', va='center', rotation='vertical')
    plt.savefig(os.path.join(args.dir, 'predicted_dists.png'), dpi=100)

    results = {
        'Test_Set_Sentences_Mean':eval_set_mean,
        'Test_Set_Sentences_Std': eval_set_std,
        'Test_Set_Length_Mean': eval_length_mean,
        'Test_Set_Length_Std': eval_length_std,

        'Train_Set_Sentences_Mean': train_set_mean,
        'Train_Set_Sentences_Std': train_set_std,
        'Train_Set_Length_Mean': train_length_mean,
        'Train_Set_Length_Std': train_length_std,

        'W_Predicted_Sentences_Mean': w_set_mean,
        'W_Predicted_Sentences_Std': w_set_std,
        'C_Predicted_Sentences_Mean': c_set_mean,
        'C_Predicted_Sentences_Std': c_set_std,
    }

    f = open(os.path.join(args.dir, 'stats.json'), 'w+')
    json.dump(results, f)

    data = datasets.load_dataset('adversarial_qa', 'adversarialQA')

    answers2 = []
    lengths2 = []

    for ex in data['validation']:
        answers_start = ex['answers']['answer_start']
        total = 0.0
        for i in range(len(answers_start)):
            a_start = answers_start[i]
            context = ex['context']
            actual_sentence, _ = in_sentence(context, a_start)
            total += actual_sentence
        answers2.append((total / len(answers_start)))
        lengths2.append(_)

    answers2 = np.array(answers2)
    lengths2 = np.array(lengths2)

    plt.clf()

    fig, axs = plt.subplots(3, 1, tight_layout=True)
    plt.xlabel('Sentence containing correct answer')
    fig.text(0.015, .5, 'Frequency', ha='center', va='center', rotation='vertical')

    aq_set_mean, aq_set_std = stats(answers2)
    aq_length_mean, aq_length_std = stats(lengths2)
    distribution(train_answers, axs, 0, "SQuAD Train Set Correct Sentence Distribution")
    distribution(eval_answers, axs, 1, "SQuAD Test Set Correct Sentence Distribution")
    distribution(answers2, axs, 2, "AdversarialQA Test Set Correct Sentence Distribution")

    plt.savefig(os.path.join(args.dir, 'aq_dists.png'), dpi=100)

    plt.clf()

    fig, axs = plt.subplots(3, 1, tight_layout=True)
    plt.xlabel('Sentence containing correct answer')
    fig.text(0.015, .5, 'Frequency', ha='center', va='center', rotation='vertical')
    distribution(train_lengths, axs, 0, "SQuAD Train Set Context Length Distribution")
    distribution(eval_lengths, axs, 1, "SQuAD Test Set Context Length Distribution")
    distribution(lengths2, axs, 2, "AdversarialQA Test Set Context Length Distribution")

    plt.savefig(os.path.join(args.dir, 'aq_lengths_dists.png'), dpi=100)

    results = {
        'AQTest_Sentences_Mean': aq_set_mean,
        'AQTest_Sentences_Std': aq_set_std,
        'AQTest_Length_Mean': aq_length_mean,
        'AQTest_Length_Std': aq_length_std,
    }

    f = open(os.path.join(args.dir, 'aqstats.json'), 'w+')
    json.dump(results, f)

if args.cw:
    base_read_path = 'eval/base_electra_8/squad/wrong_eval_predictions.jsonl'
    this_read = os.path.join(args.dir, 'wrong_eval_predictions.jsonl')
    base_list = list(open(base_read_path))
    this_list = list(open(this_read))
    base_result = []
    for jstr in base_list:
        base_result.append(json.loads(jstr))
    this_result = []
    for jstr in this_list:
        this_result.append(json.loads(jstr))

    base_ids = [base_result[i]['id'] for i in range(len(base_result))]
    this_ids = [this_result[i]['id'] for i in range(len(this_result))]

    new_wrong = 0
    same = 0
    for i in range(len(this_ids)):
        if this_ids[i] not in base_ids:
            new_wrong += 1
        else:
            same += 1

    improved = 0
    for i in range(len(base_ids)):
        if base_ids[i] not in this_ids:
            improved += 1

    print('total wrong previously: ', len(base_result))
    print('total wrong now: ', len(this_result))
    print('number of newly wrong examples', new_wrong)
    print('number of same examples still wrong: ', same)
    print('improvement: ', improved)
    print("\% Improvement: ", improved/len(base_result))
    print("\% Newly wrong: ", new_wrong/len(this_result))