import os
import re
import fasttext
import pickle
import pandas as pd
import numpy as np

def createModel(trainfile):
    hyper_params = {"lr":1.0,"epoch": 25,"wordNgrams": 2, "bucket":20000, "dim": 300,"thread":2,"loss":'softmax'}
    model = fasttext.train_supervised(input=trainfile, **hyper_params)
    model.save_model("model_SG5.bin")
    print("Model trained with the hyperparameter \n {}".format(hyper_params))

def print_results(N, p, r):
    print("N\t" + str(N))
    print("P@{}\t{:.3f}".format(1, p))
    print("R@{}\t{:.3f}".format(1, r))

def evalModel(fileName):
    model = fasttext.load_model("model_SG5.bin")
    print_results(*model.test(fileName))


def getPredictions(testFile):
    model = fasttext.load_model("model_SG5.bin")
    # model = fasttext.load_model("model_auto5.bin")
    lines = []
    with open(testFile, 'r') as f:
        line = f.readline()
        lines.append(line)

        while line:
            line = f.readline()
            if line != '':
                lines.append(line)

    sentences_with_labels = {}
    for sent_label in lines:
        sent_label = sent_label.replace('__label__1', '__label__1')
        sent_label = sent_label.replace('__label__0', '__label__2')
        # sent = re.sub('__label__\d+', '', sent_label)
        sent = sent_label.replace('__label__2', '').strip()
        sent = sent.replace('__label__1', '').strip()
        sent = sent.replace('\n', '')
        actuallabels = str(re.findall('__label__\d+', sent_label)).replace("['__label__", "").replace("']", "").strip()
        if actuallabels == '1':
            actuallabels = 1
        if actuallabels == '2':
            actuallabels = 0

        predictions_per_sent = model.predict(sent)
        pred_label = ""

        if str(predictions_per_sent[0]).replace("('", "").replace("',)", "").strip() == '__label__0':
            # pred_label = '2'
            pred_label = 0
        if str(predictions_per_sent[0]).replace("('", "").replace("',)", "").strip() == '__label__1':
            pred_label = '1'
            pred_label = 1

        sentences_with_labels[sent] = [actuallabels, pred_label]

    return sentences_with_labels

def evaluateModel(sentences_with_labels):
    tp = {}
    fp = {}
    fn = {}
    tn = {}
    count_1_tp = 0
    count_1_fp = 0
    count_1_fn = 0
    count_1_tn = 0

    count_2_tp = 0
    count_2_fp = 0
    count_2_fn = 0
    count_2_tn = 0
    for s in sentences_with_labels.keys():
        actuallabels = sentences_with_labels[s][0]
        predictedlabels = sentences_with_labels[s][1]
        if actuallabels == 1 and predictedlabels == 1:
            count_1_tp = count_1_tp + 1

        if actuallabels == 1 and predictedlabels == 0:
            count_1_fn = count_1_fn + 1
            # print("--------------------FALSE NEGATIVE-------------------------------")
            # print(s)

        if actuallabels == 0 and predictedlabels == 1:
            count_1_fp = count_1_fp + 1
            # print("--------------------FALSE POSITIVES--------------------")
            # print(s)

        if actuallabels == 0 and predictedlabels == 0:
            count_1_tn = count_1_tn + 1

    print("tp: ", count_1_tp)
    print("fp: ", count_1_fp)
    print("fn: ", count_1_fn)
    print("tn: ", count_1_tn)
    precision = float(count_1_tp) / (count_1_tp + count_1_fp)
    recall = float(count_1_tp) / (count_1_tp + count_1_fn)
    f1 = 2*(precision*recall)/(precision+recall)
    accuracy = float(count_1_tp + count_1_tn) / (count_1_tp + count_1_fp + count_1_tn + count_1_fn)
    print("precision: ",precision)
    print("recall: ",recall)
    print("accuracy: ",accuracy)
    print("-----------------")
    return precision,recall,f1,accuracy

def Average(lst):
    return sum(lst) / len(lst)

def main():
    cvscores_p = []
    cvscores_r = []
    cvscores_f1 = []
    cvscores_a = []
    files = [['./fastText_kfold/wildlifeSubset_train_1.txt','./fastText_kfold/wildlifeSubset_test_1.txt'],
             ['./fastText_kfold/wildlifeSubset_train_2.txt','./fastText_kfold/wildlifeSubset_test_2.txt'],
             ['./fastText_kfold/wildlifeSubset_train_3.txt','./fastText_kfold/wildlifeSubset_test_3.txt'],
             ['./fastText_kfold/wildlifeSubset_train_4.txt','./fastText_kfold/wildlifeSubset_test_4.txt'],
             ['./fastText_kfold/wildlifeSubset_train_5.txt','./fastText_kfold/wildlifeSubset_test_5.txt'],
             ['./fastText_kfold/wildlifeSubset_train_6.txt','./fastText_kfold/wildlifeSubset_test_6.txt'],
             ['./fastText_kfold/wildlifeSubset_train_7.txt','./fastText_kfold/wildlifeSubset_test_7.txt'],
             ['./fastText_kfold/wildlifeSubset_train_8.txt','./fastText_kfold/wildlifeSubset_test_8.txt'],
             ['./fastText_kfold/wildlifeSubset_train_9.txt','./fastText_kfold/wildlifeSubset_test_9.txt'],
             ['./fastText_kfold/wildlifeSubset_train_10.txt','./fastText_kfold/wildlifeSubset_test_10.txt']]

    for i in range(0,len(files)):
        print("========================================================")
        print("TRAIN FILE NAME: ",files[i][0])
        print("TEST FILE NAME: ", files[i][1])
        createModel(files[i][0])
        evalModel(files[i][1])
        sentences_with_labels = getPredictions(files[i][1])
        p,r,f1,a = evaluateModel(sentences_with_labels)
        cvscores_p.append(p * 100)
        cvscores_r.append(r * 100)
        cvscores_f1.append(f1 * 100)
        cvscores_a.append(a * 100)
        print("========================================================")
    print("precision: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_p), np.std(cvscores_p)))
    print("recall: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_r), np.std(cvscores_r)))
    print("F1: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_f1), np.std(cvscores_f1)))
    print("accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores_a), np.std(cvscores_a)))

main()