from bayesian_combination.ibcc import BCCWords
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

def getBCCWordsModel(num_of_classes=3, num_of_workers=3, max_iter=3):
    return BCCWords(L=num_of_classes, K=num_of_workers, max_iter=max_iter, eps=1e-4,
             alpha0_diags=1.0, alpha0_factor=1.0, beta0_factor=1.0,
             verbose=False, use_lowerbound=False)

def getAnnotationMatrix(file_path):
    df = pd.read_csv(file_path,header=None)
    df['label'] = pd.factorize(df[2],sort=True)[0]
    l = []
    n_workers = df[0].unique().shape[0]
    for i in range(df[1].unique().shape[0]):
        t = [-1] * n_workers
        for index, row in df[df[1] == i].iterrows():
            t[row[0]] = row['label']
        l.append(t)
    return np.array(l)

def getBCCWordsFeatures(number_of_items):
    words = ['w1'] * number_of_items
    return np.array(words)

def getGroundTruth(file_path):
    df = pd.read_csv(file_path)
    ground_truth = pd.factorize(df['label'], sort=True)[0]
    return ground_truth

def getGoldLabels(supervision_rate, ground_truth):
    to_delete = ground_truth.shape[0] - int(ground_truth.shape[0] * supervision_rate)
    gold_labels = ground_truth.copy()
    gold_labels[-to_delete:] = -1
    return gold_labels

def getClassifierFeatures(file_path):
    classifier_features = pd.read_csv(file_path, header=None).values
    return classifier_features

def exp_iter(epochs, supervision_rate, ground_truth, aij, classifier_features, bccwords_features, file_out, iters=[]):
    true_labels_pr = pd.get_dummies(ground_truth).values
    x_train, x_test, y_train, y_test = train_test_split(classifier_features, true_labels_pr,
                                                        test_size=(round(1 - supervision_rate, 1)), shuffle=False)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
    class_num = y_train.shape[1]
    input_dim = x_train.shape[1]
    gold_labels = getGoldLabels(supervision_rate,ground_truth)
    l = []
    for i in iters:
        test_accuracy = []
        val_accuracy = []
        test_auc = []
        val_auc = []
        for e in range(epochs):
            model = getBCCWordsModel(y_train.shape[1],aij.shape[1],i)
            probs, z_i, un = model.fit_predict(C=aij, features=bccwords_features, gold_labels=gold_labels, input_dim=input_dim,
                                           class_num=class_num, x_train=x_train, y_train=y_train, x_val=x_val,
                                           y_val=y_val, classifier_features=classifier_features)
            test_accuracy.append(accuracy_score(ground_truth[-x_test.shape[0]:], z_i[-x_test.shape[0]:]))
            val_accuracy.append(accuracy_score(ground_truth[x_train.shape[0]:-x_test.shape[0]], z_i[x_train.shape[0]:-x_test.shape[0]]))
            test_auc.append(roc_auc_score(true_labels_pr[-x_test.shape[0]:, :], probs[-x_test.shape[0]:, :], multi_class="ovo",
                                     average="macro"))
            val_auc.append(roc_auc_score(true_labels_pr[x_train.shape[0]:-x_test.shape[0], :],
                                    probs[x_train.shape[0]:-x_test.shape[0], :], multi_class="ovo",
                                    average="macro"))
        dct = {}
        dct['epochs'] = epochs
        dct['iter'] = i
        dct['test_accuracy'] = np.mean(test_accuracy)
        dct['test_auc'] = np.mean(test_auc)
        dct['val_accuracy'] = np.mean(val_accuracy)
        dct['val_auc'] = np.mean(val_auc)
        l.append(dct)
    pd.DataFrame(l).to_csv(file_out)

def exp_supervision(epochs, iter, ground_truth, aij, classifier_features, bccwords_features, file_out, supervision=[]):
    l = []
    true_labels_pr = pd.get_dummies(ground_truth).values
    for supervision_rate in supervision:
        test_accuracy = []
        val_accuracy = []
        test_auc = []
        val_auc = []
        gold_labels = getGoldLabels(supervision_rate, ground_truth)
        x_train, x_test, y_train, y_test = train_test_split(classifier_features, true_labels_pr,
                                                        test_size=(round(1 - supervision_rate, 1)), shuffle=False)
        x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, shuffle=False)
        class_num = y_train.shape[1]
        input_dim = x_train.shape[1]
        for e in range(epochs):
            model = getBCCWordsModel(y_train.shape[1], aij.shape[1], iter)
            probs, z_i, un = model.fit_predict(C=aij, features=bccwords_features, gold_labels=gold_labels, input_dim=input_dim,
                                           class_num=class_num, x_train=x_train, y_train=y_train, x_val=x_val,
                                           y_val=y_val, classifier_features=classifier_features)


            test_accuracy.append(accuracy_score(ground_truth[-x_test.shape[0]:], z_i[-x_test.shape[0]:]))
            val_accuracy.append(accuracy_score(ground_truth[x_train.shape[0]:-x_test.shape[0]], z_i[x_train.shape[0]:-x_test.shape[0]]))

            test_auc.append(roc_auc_score(true_labels_pr[-x_test.shape[0]:, :], probs[-x_test.shape[0]:, :], multi_class="ovo",
                                     average="macro"))
            val_auc.append(roc_auc_score(true_labels_pr[x_train.shape[0]:-x_test.shape[0], :],
                                    probs[x_train.shape[0]:-x_test.shape[0], :], multi_class="ovo",
                                    average="macro"))
        dct = {}
        dct['epochs'] = epochs
        dct['iter'] = iter
        dct['supervision_rate'] = supervision_rate
        dct['test_accuracy'] = np.mean(test_accuracy)
        dct['test_auc'] = np.mean(test_auc)
        dct['val_accuracy'] = np.mean(val_accuracy)
        dct['val_auc'] = np.mean(val_auc)
        l.append(dct)
    pd.DataFrame(l).to_csv(file_out)

if __name__ == '__main__':
    ### influencer experiment:
    ground_truth = getGroundTruth('../data/influencer_ground_truth.csv')
    aij = getAnnotationMatrix('../data/influencer_aij.csv')
    bccwords_features = getBCCWordsFeatures(aij.shape[0])
    classifier_features = getClassifierFeatures('../data/influencer_features.csv')
    #exp_iter(10, 0.6, ground_truth, aij, classifier_features, bccwords_features,'exp_iter_influencer.csv',[1,2])
    #exp_supervision(10,5,ground_truth,aij,classifier_features,bccwords_features,'exp_supervision_influencer.csv',[0.6,0.7])


    ### sentiment experiment:
    ground_truth = getGroundTruth('../data/sentiment_ground_truth.csv')
    aij = getAnnotationMatrix('../data/sentiment_aij.csv')
    bccwords_features = getBCCWordsFeatures(aij.shape[0])
    classifier_features = getClassifierFeatures('../data/sentiment_features.csv')
    #exp_iter(10, 0.6, ground_truth, aij, classifier_features, bccwords_features,'exp_iter_sentiment.csv',[1,2])
    #exp_supervision(10, 3, ground_truth, aij, classifier_features, bccwords_features, 'exp_supervision_sentiment.csv',[0.6, 0.7])

    ### sentiment sparse experiment:
    ground_truth = getGroundTruth('../data/sentiment_ground_truth.csv')
    aij = getAnnotationMatrix('../data/sentiment_sparse_aij.csv')
    bccwords_features = getBCCWordsFeatures(aij.shape[0])
    classifier_features = getClassifierFeatures('../data/sentiment_features.csv')
    #exp_iter(10, 0.6, ground_truth, aij, classifier_features, bccwords_features,'exp_iter_sentiment_sparse.csv',[1,2])
    #exp_supervision(10, 1, ground_truth, aij, classifier_features, bccwords_features, 'exp_supervision_sentiment_sparse.csv',[0.6, 0.7])
