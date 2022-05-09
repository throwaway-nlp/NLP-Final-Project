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


data = datasets.load_dataset('squad')

actual_answers = []
lengths = []

for ex in data['train']:
    answers_start = ex['answers']['answer_start']
    a_start = answers_start[0]
    context = ex['context']
    actual_sentence, _ = in_sentence(context, a_start)
    actual_answers.append(actual_sentence)
    lengths.append(_)

file_name = 'eval_metrics.json'
dir = 'eval_text/'
file_path = os.path.join(dir, file_name)

data = json.load(open(file_path))
eval_predicted_answers = data['eval_predicted_answers']
eval_answer_sentences = data['eval_answer_sentences']
eval_lengths = data['eval_lengths']


for i in range(len(eval_answer_sentences)):
    mean = torch.mean(torch.tensor(eval_answer_sentences[i]).float()).item()
    eval_answer_sentences[i] = mean


def stats(arr):
    mean, std = torch.std_mean(torch.tensor(arr).float())
    mean, std = mean.item(), std.item()
    return mean, std

fig, axs = plt.subplots(1,2,tight_layout=True)

def distribution(arr, i):
    maxx = max(arr)
    if (type(maxx) == typing.List):
        maxx = maxx[0]
    bins = torch.arange(maxx)
    axs[i].hist(arr, bins)

print(stats(eval_answer_sentences))
print(stats(eval_lengths))
distribution(eval_answer_sentences, 0)

print(stats(actual_answers))
print(stats(lengths))
distribution(actual_answers, 1)

plt.show()


correct = []
wrong = []
for i in range(len(eval_predicted_answers)):
    print(eval_predicted_answers[i], eval_answer_sentences[i])
    if eval_predicted_answers[i] == eval_answer_sentences[i]:
        correct.append(eval_predicted_answers[i])
    else:
        wrong.append(eval_predicted_answers[i])