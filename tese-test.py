# coding=utf-8
import csv
import math
import numpy as np
import statistics
import random
import os
from copy import copy
import copy
import pandas as pd
import seaborn as sns
from __future__ import division
from RegscorePy import *
idx = pd.IndexSlice

# curva ROC
import matplotlib
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

os.environ['R_HOME'] = 'C:\Program Files\R\R-3.4.1'
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

stats = importr('stats')

from scipy import stats, interp
from sklearn import svm, datasets, linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
import statsmodels.api as sm

mydata = pd.read_csv('clean_data_v2.csv', sep=';', na_values=[' '])
mydata = mydata.drop(['N Processo'],axis=1)

new_data = mydata[mydata['Latest_SCQ_MCHAT_Date'] <2157]# 1204, 1521,1839,2157

# ##############################################################################################
#                          IMPUTAÇÃO DE DADOS- REGRESSAO LIENAR
# ##############################################################################################
features=[]

features = new_data.copy()

features = features.drop(['Grupo'],axis=1)  # elimina a última coluna que corresponde tipo de grupo - autista ou não

print 'Número de variaveis iniciais- ', len(features.columns)  # NUMERO DE variaveis ANTES DE ELIMINAR POR MISSINGS
print 'Número de individuos- ', len(features)

compl_features = [] # lista que conterá o número das features sem missings
position_del = []  # vai conter o número das variáveis a eliminar por ter mais de 30% de missings
position_nan = []  # vai conter posição das variaveis com missings

total_nan = 0

for i in features.columns:
    m = 0  # lista que conterá o numero de missings por variável
    m = features[i].isnull().sum().sum()  # soma os missings em cada coluna - variável

    if (m * 1.0) / len(features) > 0.30:  # se tem mais de 30% de missings
        position_del.append(i)  # adiciona à lista o numero das variáveis
    elif (m * 1.0) == 0:  # se não tem missings
        compl_features.append(i)  # adiciona à lista o numero das variáveis
    else:  # restantes que correspondem às variaveis com missings entre 0 e 30%
        position_nan.append(i)  # adiciona à lista o numero das variáveis
        total_nan = total_nan + m

print 'Número total de missings- ', total_nan
print 'Número de variáveis com > 30% missings- ', len(position_del)  # NUMERO DE  variaveis A ELIMINAR com mais de 30% de missings
print 'Número de variáveis sem missings- ', len(compl_features)
print 'Número de variáveis com < 30% missings- ', len(position_nan)

print 'Variáveis com < 30% missings: '
print position_nan # imprime o nome das variáveis com menos de 30% missings

features = features.drop(position_del,axis=1)  # elimina da matriz principal as colunas das variáveis com mais de 30 % de missings
features_pred = features.copy()

correlation = features.corr(method='spearman')  # cria a matriz das correlações (usado pandas porque julgo que o numpy nao aceita missings)
correlation = correlation.abs()  # faz o modulo dos valores
correlation.values[[np.arange(len(correlation))]*2] = np.NaN  # substitui a diagonal por zeros para a propria variável nao ser identificada como muito correlacionada

# correlation = correlation.where(np.triu(np.ones(correlation.shape)).astype(np.bool))  # torna nulos os valores da parte de baixo do triangulo

#sns.heatmap(correlation)
#plt.show()

correlation = correlation.fillna(0)

quantitativas_iniciais = ['Idade paterna','Idade materna','Primeiras palavras','Primeiras Frases','Marcha','Cont Esf Diurno','Cont Esf Nocturno','Latest_SCQ_MCHAT_Date']

for k in range(len(position_nan)):

    position_var = []  # conterá o nome das variáveis correlacionadas
    sorted_corr = sorted(correlation[position_nan[k]], reverse=True)[:3]  # seleciona-se as 3 variáveis mais correlacionadas

    position_var.append(correlation[position_nan[k]][correlation[position_nan[k]] == sorted_corr[0]].index.tolist()[0])
    position_var.append(correlation[position_nan[k]][correlation[position_nan[k]] == sorted_corr[1]].index.tolist()[0])
    position_var.append(correlation[position_nan[k]][correlation[position_nan[k]] == sorted_corr[2]].index.tolist()[0])

    position_var = [x for x in position_var if x not in position_nan]

    if position_nan[k] in quantitativas_iniciais:
        print 'quantitativas----', position_nan[k], '----'

        features_train = features[~features[position_nan[k]].isnull()]  # separa as linhas da matriz total de dados que não tenham missings - na variável com missing e em todas
        features_test = features[features[position_nan[k]].isnull()]  # separa as linhas da matriz total de dados que tenham missings - na variável com missing e em todas
        index_values = features_test.index.values  # adiciona a posição dos missings na matriz
        features_X_train = features_train[position_var]  # cria matriz com os dados das variáveis correlacionadas com a variável dos missings
        features_y_train = features_train[position_nan[k]]  # cria matriz coluna com os dados sem missings da variável em análise
        features_X_test = features_test[position_var]  # cria matriz com os dados das variáveis correlacionadas correspondentes aos dados dos missings

        # Create linear regression object
        regr = LinearRegression()

        # Train the model using the training sets
        regr.fit(features_X_train, features_y_train)

        # Make predictions using the testing set
        features_y_pred = regr.predict(features_X_test)

        features_y_pred = np.around(features_y_pred)

        features_pred.loc[idx[index_values, position_nan[k]]] = features_y_pred

    else:  # position_nan[k] not in quantitativas_iniciais

        moda = features[position_nan[k]].mode()
        features_pred[position_nan[k]] = features_pred[position_nan[k]].fillna(moda[0])


# ##############################################################################################
#                          SELEÇÃO DE FEATURES
# ##############################################################################################


all = features_pred.columns.tolist()
quantitativas = [x for x in quantitativas_iniciais if x in all]
qualitativas = [x for x in all if x not in quantitativas]

features_pred['Grupo'] = new_data['Grupo']
training_features= features_pred.copy()

features_0 = features_pred.loc[features_pred['Grupo'] == 0]
features_1 = features_pred.loc[features_pred['Grupo'] == 1]

p_value = pd.DataFrame(index=['p-value'])

for i in qualitativas:

    freq_rel_0 = []
    freq_rel_1 = []

    freq_absoluta = pd.DataFrame(index=features_pred[i].value_counts(sort=False).index.tolist())
    freq_relativa = pd.DataFrame(index=features_pred[i].value_counts(sort=False).index.tolist())

    freq_absoluta['Grupo 0'] = features_0[i].value_counts(sort=False)
    freq_absoluta['Grupo 1'] = features_1[i].value_counts(sort=False)

    freq_absoluta = freq_absoluta.fillna(0)

    # Calculo do metodo de Ficher
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr

    stats = importr('stats')

    freq_absoluta_list = freq_absoluta.values.tolist()
    freq_absoluta_list = np.transpose(freq_absoluta_list)

    if len(freq_absoluta_list[0]) < 2:
        freq_absoluta_list = np.insert(freq_absoluta_list, 1, 0, axis=1)

    v = robjects.IntVector(np.reshape(freq_absoluta_list, len(freq_absoluta_list[0]) * 2))
    m = robjects.r['matrix'](v, nrow=len(freq_absoluta_list[0]))
    res = stats.fisher_test(m)
    p_value[i] = (float("{0:.19f}".format(res[0][0])))

    for j in freq_absoluta['Grupo 0']:
        # print j
        # print freq_absoluta['Grupo 0'].sum()
        freq_rel_0.append(float("{0:.2f}".format( ( j*1.0 / (freq_absoluta['Grupo 0'].sum()) ) * 100)))
        # print 'freq_rel', freq_rel_0

    freq_relativa['Grupo 0'] = freq_rel_0

    for j in freq_absoluta['Grupo 1']:
        # print j
        # print freq_absoluta['Grupo 1'].sum()
        freq_rel_1.append(float("{0:.2f}".format( ( j*1.0 / (freq_absoluta['Grupo 1'].sum()) ) * 100)))
        # print 'freq_rel', freq_rel_1

    freq_relativa['Grupo 1'] = freq_rel_1

    ################### Elimnar vairaveis com o p>0.05
    if p_value.loc['p-value', i] < 0.05:
        position_del.append(i)
    else:  # Elimina o conteudo das células dos individuos correspondentes às colunas das condiçoes contrarias do if anterior
        training_features = training_features.drop(i, axis=1)
        print 'eliminada p<-0.05 - ', i

mediana = pd.DataFrame(index=['Grupo 0', 'Grupo 1'])
percentil_25 = pd.DataFrame(index=['Grupo 0', 'Grupo 1'])
percentil_75 = pd.DataFrame(index=['Grupo 0', 'Grupo 1'])
shapiro_test = pd.DataFrame(index=['Grupo 0', 'Grupo 1'])
levene = pd.DataFrame(index=['Result'])

for i in quantitativas:

    # CALCULO DA MEDIANA
    mediana[i] = [statistics.median(features_0[i]),statistics.median(features_1[i])]

    # CALCULO DO PERCENTIL
    test_0 = features_0[i]
    test_1 = features_1[i]

    test_0 = test_0.sort_values()
    test_1 = test_1.sort_values()

    percentil_25[i] = [np.percentile(test_0, 25), np.percentile(test_1, 25)]
    percentil_75[i] = [np.percentile(test_0, 75), np.percentile(test_1, 75)]

    # VERIFICAR NORMALIDADE - TEST SHAPIRO WILK
    from scipy import stats
    shapiro_test[i] = [stats.shapiro(features_0[i])[1],stats.shapiro(features_1[i])[1]]  # o 1 corresponde ao p-value porque os outputs do shapiro sao (W : float, p-value : float)

    if shapiro_test.loc['Grupo 0', i] > 0.05 and shapiro_test.loc['Grupo 1', i] > 0.05:  # Verificar se é Normalmente distribuida

        levene[i] = [stats.levene(features_0[i], features_1[i], center='mean')[1]]

        if levene.loc['Result',i] < 0.05:
            p_value[i] = stats.ttest_ind(features_0[i], features_1[i], equal_var=False)[1]  # TESTE T-STUDENT
            print 'n individuos diferentes- ',i
        else:
            p_value[i] = stats.ttest_ind(features_0[i], features_1[i])[1]
            print 'n individuos iguais- ',i

    else:
        p_value[i] = stats.mannwhitneyu(features_0[i], features_1[i], use_continuity=False,alternative='two-sided')[1]  # TESTE MANN WHITNEY

    if p_value.loc['p-value', i] < 0.05:
        position_del.append(i)
    else:  # Elimina o conteudo das células dos individuos correspondentes às colunas das condiçoes contrarias do if anterior
        training_features = training_features.drop(i, axis=1)
        print 'eliminada p<-0.05 - ', i


# ------------------------------------------------------------------------------------------
#                TESTE AIC   e    BIC
# -----------------------------------------------------------------------------------------

sorted_aic = pd.DataFrame()
sorted_bic = pd.DataFrame()

for i in training_features.loc[:, training_features.columns != 'Grupo']:
    print i

    imp_features = training_features.drop(i, axis=1)

    ########### outra forma pela regressao logistica
    #model = LogisticRegression()

    XX = training_features[i] # imp_features.loc[:, imp_features.columns != 'Grupo']
    XX = sm.add_constant(XX)
    # Train the model using the training sets

    model = sm.OLS(imp_features['Grupo'], XX).fit()

    # model.summary()

    sorted_aic[i] = [model.__getattribute__('aic')]
    sorted_bic[i] = [model.__getattribute__('bic')]

    print 'aic- ', model.__getattribute__('aic')
    print 'bic- ', model.__getattribute__('bic')


########### outra forma pela regressao logistica mas dá erro no mchat15
    if i == 'MCHAT15':
        continue
    XX = training_features[i]
    XX = XX.astype('float64')
    YY = imp_features['Grupo'].astype('float64')

    m1 = sm.Logit(YY, XX)

    m1.fit()

    sorted_aic[i] = [m1.fit().aic]
    sorted_bic[i] = [m1.fit().bic]

    print 'aic- ', m1.fit().aic
    print 'bic- ', m1.fit().bic


####### https://github.com/UBC-MDS/RegscorePy
    yy = training_features[i].tolist()
    yy = map(float, yy)
    yypred = training_features['Grupo'].tolist()
    yypred = map(float, yypred)

    print aic.aic(yy, yypred, 1)
    print bic.bic(yy, yypred, 1)




sorted_aic = sorted_aic.transpose()
sorted_bic = sorted_bic.transpose()

sorted_aic = sorted_aic.sort_values(by=[0], ascending=False)
sorted_bic = sorted_bic.sort_values(by=[0], ascending=False)

print sorted_aic
print sorted_bic

fig = plt.figure(figsize=(8,5))
plt.subplot(211)
sing_vals = np.arange(len(sorted_aic)) + 1
plt.plot(sing_vals,sorted_aic, 'ro-', linewidth=2)
plt.subplot(212)
sing_vals1 = np.arange(len(sorted_bic)) + 1
plt.plot(sing_vals1,sorted_bic, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')




















# ##############################################################################################
#                          CLASSIFICAÇÃO
# ##############################################################################################

X_0_subject = []
X_1_subject = []

training_features_0 = training_features.loc[features_pred['Grupo'] == 0]   # lista de elementos nao autistas
training_features_1 = training_features.loc[features_pred['Grupo'] == 1]  # lista de elementos autistas

accuracy_svm = []
sensitivity_svm = []
specificity_svm = []
roc_svm = []

accuracy_rf = []
sensitivity_rf = []
specificity_rf = []
roc_rf = []
rf = RandomForestClassifier(n_estimators=100)

accuracy_lr = []
sensitivity_lr = []
specificity_lr = []
roc_lr = []

# -----------------------------------------------------------------------------------------------
#                                     BALANCEAMENTO
# -----------------------------------------------------------------------------------------------

for i in range(100):  # correr 100x

    new_features = pd.DataFrame(columns=training_features_0.columns.tolist())
    labels = []

    a=0
    b=0

    for j in range(len(training_features_1)):

        c = random.choice(training_features_0.index.tolist())
        new_features = new_features.append(training_features_0.loc[c])

        c = random.choice(training_features_1.index.tolist())
        new_features = new_features.append(training_features_1.loc[c])

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    x_train, x_test, y_train, y_test = train_test_split(new_features.iloc[:, :-1], new_features.iloc[:, -1], test_size=0.30, random_state=None)

    # print "x_train", len(x_train), x_train
    # print "x_test", len(x_test), x_train
    # print "y_train", len(y_train), y_train
    # print "y_test", len(y_test), y_test

    # -----------------------------------------------------------------------------------------------
    #                                     SVM
    # -----------------------------------------------------------------------------------------------

    model = svm.SVC(kernel='linear')
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    confusion = confusion_matrix(y_test, predicted)
    confusion = np.asmatrix(confusion)

    accuracy_svm.append(accuracy_score(y_test, predicted))
    sensitivity_svm.append(float(confusion[0, 0]) / (float(confusion[0, 0]) + float(confusion[0, 1])))
    specificity_svm.append(float(confusion[1, 1]) / (float(confusion[1, 0]) + float(confusion[1, 1])))

    fpr, tpr, threshold = roc_curve(y_test, predicted)
    roc_svm.append(auc(fpr, tpr))

    # ROC Curve
    # plt.title('Receiver Operating Characteristic')
    # plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_svm[i])
    # plt.legend(loc = 'lower right')
    # plt.plot([0, 1], [0, 1],'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

    # -----------------------------------------------------------------------------------------------
    #                                RANDOM FOREST
    # -----------------------------------------------------------------------------------------------

    rf.fit(x_train, y_train)

    results_rf = rf.predict(x_test)

    confusion_rf = confusion_matrix(y_test, results_rf)
    confusion_rf = np.asmatrix(confusion_rf)

    accuracy_rf.append(accuracy_score(y_test, results_rf))
    sensitivity_rf.append(float(confusion_rf[0, 0]) / (float(confusion_rf[0, 0]) + float(confusion_rf[0, 1])))
    specificity_rf.append(float(confusion_rf[1, 1]) / (float(confusion_rf[1, 0]) + float(confusion_rf[1, 1])))

    fpr, tpr, threshold = roc_curve(y_test, results_rf)
    roc_rf.append(auc(fpr, tpr))

    # -----------------------------------------------------------------------------------------------
    #                               LOGISTIC REGRESSION
    # -----------------------------------------------------------------------------------------------

    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    results_lr = lr.predict(x_test)

    confusion_lr = confusion_matrix(y_test, results_lr)
    confusion_lr = np.asmatrix(confusion_lr)

    accuracy_lr.append(accuracy_score(y_test, results_lr))
    sensitivity_lr.append(float(confusion_lr[0,0])/(float(confusion_lr[0,0])+float(confusion_lr[0,1])))
    specificity_lr.append(float(confusion_lr[1,1])/(float(confusion_lr[1,0])+float(confusion_lr[1,1])))

    fpr, tpr, threshold = roc_curve(y_test, results_lr)
    roc_lr.append(auc(fpr, tpr))

accuracy_svm = statistics.mean(accuracy_svm)
sensitivity_svm = statistics.mean(sensitivity_svm)
specificity_svm = statistics.mean(specificity_svm)
roc_svm = statistics.mean(roc_svm)

print 'accuracy svm->', accuracy_svm
print 'sensitivity svm->', sensitivity_svm
print 'specificity svm->', specificity_svm
print 'roc svm->', roc_svm

accuracy_rf = statistics.mean(accuracy_rf)
sensitivity_rf = statistics.mean(sensitivity_rf)
specificity_rf = statistics.mean(specificity_rf)
roc_rf = statistics.mean(roc_rf)

print 'accuracy random->', accuracy_rf
print 'sensitivity random->', sensitivity_rf
print 'specificity random->', specificity_rf
print 'roc random->', roc_rf

accuracy_lr= statistics.mean(accuracy_lr)
sensitivity_lr = statistics.mean(sensitivity_lr)
specificity_lr = statistics.mean(specificity_lr)
roc_lr = statistics.mean(roc_lr)

print 'accuracy logistic->',accuracy_lr
print 'sensitivity logistic->',sensitivity_lr
print 'specificity logistic->',specificity_lr
print 'roc curve logistic->', roc_lr










#
#
#
#
#
# # ##############################################################################################
# #                          CLASSIFICAÇÃO
# # ##############################################################################################
#
# group = np.copy(subject_grupo[1:, -1])
# group = group[:].astype(float)
#
# features_group = pd.DataFrame.as_matrix(features_imputed)
#
# features_group = np.insert(features_group,len(features_group[0]), [group], axis=1)
#
#
# X_0_subject = []
# X_1_subject = []
#
# # 0 --> Nao autista
# # 1 --> Autista
#
# X_0_subject = features_group[features_group[:,-1]==0]   # lista de elementos nao autistas
# X_1_subject = features_group[features_group[:, -1] == 1]  # lista de elementos autistas
#
# # print len(X_0_subject)
# # print len(X_1_subject)
#
# accuracy_svm = []
# sensitivity_svm = []
# specificity_svm = []
# roc_svm = []
#
# accuracy_rf = []
# sensitivity_rf = []
# specificity_rf = []
# roc_rf = []
# rf = RandomForestClassifier(n_estimators=100)
#
# accuracy_lr = []
# sensitivity_lr = []
# specificity_lr = []
# roc_lr = []
#
# for i in range(100):  # correr 100x
#
#     new_subject = []
#     labels = []
#     a=0
#     b=0
#     # -----------------------------------------------------------------------------------------------
#     #                                     BALANCEAMENTO
#     # -----------------------------------------------------------------------------------------------
#
#     while len(new_subject) != len(X_0_subject):  # enquanto a nova lista nao tiver o mesmo numero que o total de nao autista continua o ciclo
#         c = random.randrange(len(features_group))  # procura um elemento aleatorio do suject_grupo
#
#         if features_group[c][len(features_group[c]) - 1] == 0:  # se esse elemento for nao autista adiciona à nova matriz
#             a=a+1
#             new_subject.append(features_group[c][: (len(features_group[c]) - 1)])
#             labels.append(features_group[c][(len(features_group[c]) - 1)])
#
#     while len(new_subject) != 2 * len(X_0_subject):  # enquanto a nova lista nao tiver o mesmo numero de autista como de nao autistas continua o ciclo
#         c = random.randrange(len(features_group))  # procura um elemento aleatorio do suject_grupo
#         if features_group[c][len(features_group[c]) - 1] == 1:  # se esse elemento for autista adiciona à nova matriz
#             b=b+1
#             new_subject.append(features_group[c][: (len(features_group[c]) - 1)])
#             labels.append(features_group[c][(len(features_group[c]) - 1)])
#
#
#     new_subject = np.array(new_subject)
#
#     x_train = []
#     x_test = []
#     y_train = []
#     y_test = []
#
#     x_train, x_test, y_train, y_test = train_test_split(new_subject, labels, test_size=0.30, random_state=None)
#
#     # print "x_train", len(x_train), x_train
#     # print "x_test", len(x_test), x_train
#     # print "y_train", len(y_train), y_train
#     # print "y_test", len(y_test), y_test
#
#     # -----------------------------------------------------------------------------------------------
#     #                                     SVM
#     # -----------------------------------------------------------------------------------------------
#
#     model = svm.SVC(kernel='linear')
#     model.fit(x_train, y_train)
#     # model.score(Xtrain_total, y_train)
#     predicted = model.predict(x_test)
#
#     confusion = confusion_matrix(y_test, predicted)
#     confusion = np.asmatrix(confusion)
#
#     accuracy_svm.append(accuracy_score(y_test, predicted))
#     sensitivity_svm.append(float(confusion[0, 0]) / (float(confusion[0, 0]) + float(confusion[0, 1])))
#     specificity_svm.append(float(confusion[1, 1]) / (float(confusion[1, 0]) + float(confusion[1, 1])))
#
#     fpr, tpr, threshold = roc_curve(y_test, predicted)
#     roc_svm.append(auc(fpr, tpr))
#
#     # ROC Curve
#     plt.title('Receiver Operating Characteristic')
#     plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_svm[i])
#     plt.legend(loc = 'lower right')
#     plt.plot([0, 1], [0, 1],'r--')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.ylabel('True Positive Rate')
#     plt.xlabel('False Positive Rate')
#     plt.show()
#
#     # -----------------------------------------------------------------------------------------------
#     #                                RANDOM FOREST
#     # -----------------------------------------------------------------------------------------------
#
#     rf.fit(x_train, y_train)
#
#     results_rf = rf.predict(x_test)
#
#     confusion_rf = confusion_matrix(y_test, results_rf)
#     confusion_rf = np.asmatrix(confusion_rf)
#
#     accuracy_rf.append(accuracy_score(y_test, results_rf))
#     sensitivity_rf.append(float(confusion_rf[0, 0]) / (float(confusion_rf[0, 0]) + float(confusion_rf[0, 1])))
#     specificity_rf.append(float(confusion_rf[1, 1]) / (float(confusion_rf[1, 0]) + float(confusion_rf[1, 1])))
#
#     fpr, tpr, threshold = roc_curve(y_test, results_rf)
#     roc_rf.append(auc(fpr, tpr))
#
#     # -----------------------------------------------------------------------------------------------
#     #                               LOGISTIC REGRESSION
#     # -----------------------------------------------------------------------------------------------
#
#     lr = LogisticRegression()
#     lr.fit(x_train, y_train)
#
#     results_lr = lr.predict(x_test)
#
#     confusion_lr = confusion_matrix(y_test, results_lr)
#     confusion_lr = np.asmatrix(confusion_lr)
#
#     accuracy_lr.append(accuracy_score(y_test, results_lr))
#     sensitivity_lr.append(float(confusion_lr[0,0])/(float(confusion_lr[0,0])+float(confusion_lr[0,1])))
#     specificity_lr.append(float(confusion_lr[1,1])/(float(confusion_lr[1,0])+float(confusion_lr[1,1])))
#
#     fpr, tpr, threshold = roc_curve(y_test, results_lr)
#     roc_lr.append(auc(fpr, tpr))
#
#
# print 'Número de individuos considerados no classificador (autistas e não autistas)- ', a, b
#
# accuracy_svm = statistics.mean(accuracy_svm)
# sensitivity_svm = statistics.mean(sensitivity_svm)
# specificity_svm = statistics.mean(specificity_svm)
# roc_svm = statistics.mean(roc_svm)
#
# print 'accuracy svm->', accuracy_svm
# print 'sensitivity svm->', sensitivity_svm
# print 'specificity svm->', specificity_svm
# print 'roc svm->', roc_svm
#
# accuracy_rf = statistics.mean(accuracy_rf)
# sensitivity_rf = statistics.mean(sensitivity_rf)
# specificity_rf = statistics.mean(specificity_rf)
# roc_rf = statistics.mean(roc_rf)
#
# print 'accuracy random->', accuracy_rf
# print 'sensitivity random->', sensitivity_rf
# print 'specificity random->', specificity_rf
# print 'roc random->', roc_rf
#
# accuracy_lr= statistics.mean(accuracy_lr)
# sensitivity_lr = statistics.mean(sensitivity_lr)
# specificity_lr = statistics.mean(specificity_lr)
# roc_lr = statistics.mean(roc_lr)
#
# print 'accuracy logistic->',accuracy_lr
# print 'sensitivity logistic->',sensitivity_lr
# print 'specificity logistic->',specificity_lr
# print 'roc curve logistic->', roc_lr
#
