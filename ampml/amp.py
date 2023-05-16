import sys,math
import os,re
import argparse
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import sklearn.utils
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import svm
#from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
#from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score,RocCurveDisplay,roc_curve
class DottableDict(dict):
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.__dict__ = self
    def allowDotting(self, state=True):
        if state:
            self.__dict__ = self
        else:
            self.__dict__ = dict()
def train(args):
    try:
        if args.representation == 'CTDD':
            positive_df = CTDD(args.positive)
        elif args.representation == 'PAAC':
            positive_df = PAAC(args.positive)
        elif args.representation == 'AAC':
            positive_df = AAC(args.positive)
        else:
            print("Please check method name.")
            sys.exit(1)
        positive_df['classi'] = 1
    except FileNotFoundError:
        print(f"AMP positive fasta sequence file: {args.positive} not found!", file=sys.stderr)
        sys.exit(1)

    try:
        if args.representation == 'CTDD':
            negative_df = CTDD(args.negative)
        elif args.representation == 'PAAC':
            negative_df = PAAC(args.negative)
        elif args.representation == 'AAC':
            negative_df = AAC(args.negative)
        else:
            print("Please check method name.")
            sys.exit(1)
        negative_df['classi'] = 0
    except FileNotFoundError:
        print(f"AMP negative fasta sequence file: {args.negative} not found!", file=sys.stderr)
        sys.exit(1)

    feature_drop_list = []
    if args.drop_feature:
        try:
            with open(args.drop_feature, "r") as drop_data:
                for feature in drop_data:
                    feature = feature.strip()
                    feature_drop_list.append(feature)
        except FileNotFoundError:
            print(f"Feature drop file: {args.drop_feature} not found!", file=sys.stderr)
            sys.exit(1)
    feature_drop_list.append("classi")
    training_df = pd.concat([positive_df, negative_df])
    training_df = sklearn.utils.shuffle(training_df, random_state=args.seed)
    X = training_df.drop(columns=feature_drop_list)
    y = training_df.classi
    if args.tree_test:
        min_estimators = 50
        max_estimators = 150
        print("n_estimators\toob_error\toob_balanced_error")
        for i in range(min_estimators, max_estimators + 1):
            clf_ = RandomForestClassifier(n_estimators=i, oob_score=True,
                                          random_state=args.seed,
                                          n_jobs=4)
            clf_.fit(X, y)
            oob_error = 1 - clf_.oob_score_
            pred_train = np.argmax(clf_.oob_decision_function_, axis=1).tolist()
            oob_balanced_accuracy = metrics.balanced_accuracy_score(y, pred_train)
            print(f"{i}\t{oob_error}\t{1-oob_balanced_accuracy}")

    if args.feature_importance:
        from rfpimp import importances,plot_importances
        clf = RandomForestClassifier(n_estimators=args.num_trees, min_samples_split=10,min_samples_leaf=5,oob_score=True,
                                random_state=args.seed,
                                n_jobs=4)
        clf.fit(X, y)
        imp = importances(clf, X, y,n_samples=-1)
        imp.to_csv('feature_importances.txt',sep='\t')
        viz = plot_importances(imp)
        viz.view()
        #oob_dropcol_importances(clf, X, y, args.seed, 4)

    if args.method=='SVM':
        clf = svm.SVC(C=1, kernel='rbf',probability=True)
        Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.2,random_state=1024)
        clf.fit(Xtrain.values, Ytrain)
        pred_test_y = clf.predict(Xtest)
        CM=metrics.confusion_matrix(Ytest, pred_test_y)
        ROC = RocCurveDisplay.from_estimator(clf, Xtest, Ytest)
        plt.savefig('ROC_SVM.png')
        cv_scores = cross_val_score(clf,Xtest,Ytest,cv=10,scoring='accuracy')
        with open(f"model_SVM_score.txt",'w') as OUT:
            OUT.write(f'accuracy: {clf.score(Xtest, Ytest)}\n')
            OUT.write(f'precision: {precision_score(Ytest, pred_test_y)}\n')
            OUT.write(f'recall: {recall_score(Ytest, pred_test_y)}\n')
            OUT.write(f'f1: {f1_score(Ytest, pred_test_y)}\n')
            OUT.write(f'cv 10 scores: {sum(cv_scores)/len(cv_scores)}\n')
        try:
            with open(f"AMPpred_{args.representation}.model", "wb") as model_pickle:
                pickle.dump(clf, model_pickle)
        except IOError:
            print("Error in writing model to file!", file=sys.stderr)
        return sum(cv_scores)/len(cv_scores),CM
    elif args.method=='RF':
        clf = RandomForestClassifier(n_estimators=args.num_trees, min_samples_split=10,max_depth=19,min_samples_leaf=5,oob_score=True,
                                random_state=args.seed,
                                n_jobs=4) # CTDD 111 PAAC 143
        Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.1,random_state=1024)
        clf.fit(Xtrain.values, Ytrain)
        pred_train = np.argmax(clf.oob_decision_function_, axis=1).tolist()
        y_predprob = clf.predict_proba(Xtest)[:,1]
        pred_test_y = clf.predict(Xtest)
        CM=metrics.confusion_matrix(Ytest, pred_test_y)
        fpr, tpr, thresholds = roc_curve(Ytest, y_predprob)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        ROC = RocCurveDisplay.from_estimator(clf, Xtest, Ytest)
        plt.savefig('ROC_RF.png')
        with open(f"model_RF_score.txt",'w') as OUT:
            OUT.write(f'accuracy: {clf.score(Xtest, Ytest)}\n')
            OUT.write(f'precision: {precision_score(Ytest, pred_test_y)}\n')
            OUT.write(f'recall: {recall_score(Ytest, pred_test_y)}\n')
            OUT.write(f'f1: {f1_score(Ytest, pred_test_y)}\n')
            OUT.write(f'Out-of-bag accuracy:{clf.oob_score_}\n')
            OUT.write(f'Out-of-bag balanced accuracy:{metrics.balanced_accuracy_score(Ytrain, pred_train)}\n')
            OUT.write(f'AUC Score:: {metrics.roc_auc_score(Ytest, y_predprob)}\n')
            OUT.write(f'The optimal threshold for classification:: {optimal_threshold}\n')

        #param_test1 = {'max_depth':range(3,20,2), 'min_samples_split':range(10,111,10)}
        #gsearch1 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=100,min_samples_leaf=5,oob_score=True,
        #                        random_state=args.seed,n_jobs=4), param_grid = param_test1, scoring='roc_auc',verbose=1,iid=False,cv=5,n_jobs=-1)
        #gsearch1.fit(X,y)
        #print(gsearch1.best_params_, gsearch1.best_score_)

        try:
            with open(f"AMPpred_{args.representation}.model", "wb") as model_pickle:
                pickle.dump(clf, model_pickle)
        except IOError:
            print("Error in writing model to file!", file=sys.stderr)
        return clf.oob_score_,metrics.balanced_accuracy_score(Ytrain, pred_train),metrics.roc_auc_score(Ytest, y_predprob),CM
    elif args.method=='GT':
        clf = GradientBoostingClassifier(n_estimators=145, learning_rate=0.1,min_samples_split=78,max_depth=10,subsample=0.8,random_state=args.seed)
        Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.1,random_state=1024)
        clf.fit(Xtrain.values, Ytrain)
        pred_test_y = clf.predict(Xtest)
        cv_scores = cross_val_score(clf,Xtest,Ytest,cv=10,scoring='accuracy')
        CM=metrics.confusion_matrix(Ytest, pred_test_y)
        ROC = RocCurveDisplay.from_estimator(clf, Xtest, Ytest)
        plt.savefig('ROC_GT.png')
        with open(f"model_GT_score.txt",'w') as OUT:
            OUT.write(f'accuracy: {clf.score(Xtest, Ytest)}\n')
            OUT.write(f'precision: {precision_score(Ytest, pred_test_y)}\n')
            OUT.write(f'recall: {recall_score(Ytest, pred_test_y)}\n')
            OUT.write(f'f1: {f1_score(Ytest, pred_test_y)}\n')
            OUT.write(f'Out-of-bag accuracy:: {clf.oob_score_}\n')
            OUT.write(f'Out-of-bag balanced accuracy:: {metrics.balanced_accuracy_score(Ytrain, pred_train)}\n')
            OUT.write(f'AUC Score:: {metrics.roc_auc_score(Ytest, y_predprob)}\n')
        try:
            with open(f"AMPpred_{args.representation}.model", "wb") as model_pickle:
                pickle.dump(clf, model_pickle)
        except IOError:
            print("Error in writing model to file!", file=sys.stderr)
        return sum(cv_scores)/len(cv_scores),CM
    elif args.method=='bayes':
        clf = MultinomialNB()
        Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size=0.2,random_state=1024)
        clf.fit(Xtrain.values, Ytrain)
        pred_test_y = clf.predict(Xtest)
        CM=metrics.confusion_matrix(Ytest, pred_test_y)
        cv_scores = cross_val_score(clf,Xtest,Ytest,cv=10,scoring='accuracy')
        ROC = RocCurveDisplay.from_estimator(clf, Xtest, Ytest)
        plt.savefig('ROC_bayes.png')
        with open(f"model_bayes_score.txt",'w') as OUT:
            OUT.write(f'accuracy: {clf.score(Xtest, Ytest)}\n')
            OUT.write(f'precision: {precision_score(Ytest, pred_test_y)}\n')
            OUT.write(f'recall: {recall_score(Ytest, pred_test_y)}\n')
            OUT.write(f'f1: {f1_score(Ytest, pred_test_y)}\n')
            OUT.write(f'cv 10 scores: {sum(cv_scores)/len(cv_scores)}\n')
        try:
            with open(f"AMPpred_{args.representation}.model", "wb") as model_pickle:
                pickle.dump(clf, model_pickle)
        except IOError:
            print("Error in writing model to file!", file=sys.stderr)
        return sum(cv_scores)/len(cv_scores),CM
    else:
        print("Please check ML method name.")
        sys.exit(1)

def predict(args):
    try:
        with open(args.model, "rb") as model_handle:
            clf = pickle.load(model_handle)
    except FileNotFoundError:
        print(f"Model file: {args.model} not found!", file=sys.stderr)
        sys.exit(1)

    try:
        if args.representation == 'CTDD':
            classify_df = CTDD(args.seq_file)
        elif args.representation == 'PAAC':
            classify_df = PAAC(args.seq_file)
        elif args.representation == 'AAC':
            classify_df = AAC(args.seq_file)
        else:
            print("Please check method name.")
            sys.exit(1)
        classify_df.to_csv('feature.txt',sep='\t',index_label="seq_name")
    except FileNotFoundError:
        print(f"Sequence file: {args.seq_file} not found!", file=sys.stderr)
        sys.exit(1)

    classify_output = open('AMPpred.tsv', "w")

    feature_drop_list = []
    if args.drop_feature:
        try:
            with open(args.drop_feature, "r") as drop_data:
                for feature in drop_data:
                    feature = feature.strip()
                    feature_drop_list.append(feature)
        except FileNotFoundError:
            print(f"Feature drop file: {args.drop_feature} not found!", file=sys.stderr)
            sys.exit(1)

    if feature_drop_list:
        classify_df = classify_df.drop(columns=feature_drop_list)
    id_info = classify_df.index.tolist()

    print("probability_nonAMP\tprobability_AMP\tpredicted\tseq_id", file=classify_output)
    preds = clf.predict_proba(classify_df)
    for i, pred in enumerate(preds):
        pred_list = pred.tolist()
        if args.threshold:
            if pred[1] > float(args.threshold):
                predicted = "AMP"
            else:
                predicted = "nonAMP"
            output_line = "{}\t{}\t{}".format("\t".join([str(y) for y in pred_list]),
                                          predicted,
                                          id_info[i])
        else:
            if clf.predict(classify_df.loc[id_info[i], :].to_numpy().reshape(1, -1))[0] == 1:
                predicted = "AMP"
            else:
                predicted = "nonAMP"
            output_line = "{}\t{}\t{}".format("\t".join([str(y) for y in pred_list]),
                                          predicted,
                                          id_info[i])
        print(output_line, file=classify_output)
    classify_output.close()

def parseFasta(filename): #seq_api
    fas = {}
    idlis = []
    id = None
    with open(filename, 'r') as fh:
        for line in fh:
            if line[0] == '>':
                header = line[1:].rstrip() # remove >
                id = header.split()[0]
                idlis.append(id)
                fas[id] = []
            else:
                fas[id].append(line.rstrip())
        for id, seq in fas.items():
            fas[id] = ''.join(seq)
    return fas
def Count(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code
def AAC(fasta):
    AA = 'ARNDCQEGHILKMFPSTWYV'
    encodings = []
    header = []
    names= []
    for i in AA:
        header.append(i)
    fastas = parseFasta(fasta)
    for seq_id,sequence in fastas.items():
        code = []
        names.append(seq_id)
        sequence = re.sub('[^ARNDCQEGHILKMFPSTWYV]', '',sequence)
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
    Property_dataframe = pd.DataFrame.from_dict(encodings)
    Property_dataframe.columns = header
    Property_dataframe.index = names
    return Property_dataframe
def CTDD(fasta):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity':        'LIFWCMVY',
        'polarizability':  'GASDT',
        'charge':          'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess':   'ALFCGIVW'
    }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity':        'PATGS',
        'polarizability':  'CPNVEQIL',
        'charge':          'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess':   'RKQEND'
    }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity':        'HQRKNED',
        'polarizability':  'KMHFRYW',
        'charge':          'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess':   'MSPTHY'
    }

    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    encodings = []
    header = []
    names = []
    fastas = parseFasta(fasta)
    for p in property:
        for g in ('1', '2', '3'):
            for d in ['0', '25', '50', '75', '100']:
                header.append(p + '.' + g + '.residue' + d)
    for seq_id,sequence in fastas.items():
        names.append(seq_id)
        code = []
        sequence = re.sub('[^ARNDCQEGHILKMFPSTWYV]', '',sequence)
        for p in property:
            code = code + Count(group1[p], sequence) + Count(group2[p], sequence) + Count(group3[p], sequence)
        encodings.append(code)
    Property_dataframe = pd.DataFrame.from_dict(encodings)
    Property_dataframe.columns = header
    Property_dataframe.index = names
    return Property_dataframe

def Rvalue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)
def PAAC(fasta, lambdaValue=9, w=0.05):
    Hydrophobicity = {
        'A':0.62,'R':-2.53,'N':-0.78,'D':-0.9,'C':0.29,"Q":-0.85,'E':-0.74,'G':0.48,'H':-0.4,'I':1.38,
        'L':1.06,'K':-1.5,'M':0.64,'F':1.19,'P':0.12,'S':-0.18,'T':-0.05,'W':0.81,'Y':0.26,'V':1.08 }
    Hydrophilicity = {
        'A':-0.5,'R':3.0,'N':0.2,'D':3.0,'C':-1.0,"Q":0.2,'E':3.0,'G':0.0,'H':-0.5,'I':-1.8,
        'L':-1.8,'K':3.0,'M':-1.3,'F':-2.5,'P':0.0,'S':0.3,'T':-0.4,'W':-3.4,'Y':-2.3,'V':-1.5 }
    SideChainMass = {
        'A':15.0,'R':101.0,'N':58.0,'D':59.0,'C':47.0,"Q":72.0,'E':73.0,'G':1.0,'H':82.0,'I':57.0,
        'L':57.0,'K':73.0,'M':75.0,'F':91.0,'P':42.0,'S':31.0,'T':45.0,'W':130.0,'Y':107.0,'V':43.0 }
    pI = {
        'A':6.11,'R':10.76,'N':10.76,'D':2.98,'C':5.02,"Q":5.65,'E':3.08,'G':6.06,'H':7.64,'I':6.04,
        'L':6.04,'K':9.47,'M':5.74,'F':5.91,'P':6.30,'S':5.68,'T':5.60,'W':5.88,'Y':5.63,'V':6.02 }
    AA = list(Hydrophobicity.keys())
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAProperty.append(list(Hydrophobicity.values()))
    AAProperty.append(list(Hydrophilicity.values()))
    AAProperty.append(list(SideChainMass.values()))
    #AAProperty.append(list(pI.values()))
    AAProperty1 = []
    for i in AAProperty: #z-score
        meanI = sum(i) / 20
        std = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
        AAProperty1.append([(j-meanI)/std for j in i])

    encodings = []
    header = []
    names = []
    fastas = parseFasta(fasta)
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))

    for seq_id,sequence in fastas.items():
        names.append(seq_id)
        code = []
        theta = []
        sequence = re.sub('[^ARNDCQEGHILKMFPSTWYV]', '',sequence)
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
    Property_dataframe = pd.DataFrame.from_dict(encodings)
    Property_dataframe.columns = header
    Property_dataframe.index = names
    return Property_dataframe

def oob_dropcol_importances(rf, X_train, y_train, seed, n_jobs):
    """
    Compute drop-column feature importances for scikit-learn.
    Given a RandomForestClassifier or RandomForestRegressor in rf
    and training X and y data, return a data frame with columns
    Feature and Importance sorted in reverse order by importance.
    A clone of rf is trained once to get the baseline score and then
    again, once per feature to compute the drop in out of bag (OOB)
    score.
    return: A data frame with Feature, Importance columns
    SAMPLE CODE
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1, oob_score=True)
    X_train, y_train = ..., ...
    rf.fit(X_train, y_train)
    imp = oob_dropcol_importances(rf, X_train, y_train)
    """
    rf_ = clone(rf)
    rf_.random_state = seed
    rf_.oob_score = True
    rf_.n_jobs = n_jobs
    rf_.fit(X_train, y_train)
    baseline = rf_.oob_score_
    imp = []
    for col in X_train.columns:
        rf_ = clone(rf)
        rf_.random_state = seed
        rf_.oob_score = True
        rf_.n_jobs = n_jobs
        rf_.fit(X_train.drop(col, axis=1), y_train)
        drop_in_score = baseline - rf_.oob_score_
        imp.append(drop_in_score)
    imp = np.array(imp)
    I = pd.DataFrame(data={'Feature':X_train.columns, 'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    I.to_csv("feature.importances.dropcol.oob.csv", index=True)
