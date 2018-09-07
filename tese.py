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

idx = pd.IndexSlice
# curva ROC
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
#

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

values_SVM = []

with open('clean_data_v2.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=';', quoting=csv.QUOTE_NONE)

    y = []
    y_total = []
    subject = []  # corresponde aos dados de cada individuo de teste
    var_total = []  # criar lista com 77 listas

    # Fazer estudo por patamares de idade
    for line in reader:

        if line[12] == 'Latest_SCQ_MCHAT_Date':
            subject.append(line)
        elif int(line[12]) < 1204:  # 1204, 1521,1839,2157
            subject.append(line)
        else:
            continue

    # criar uma lista com todos os dados. todos seguidos por ordem de colunas. ordená-los
    for k in range(len(subject[1])):
        for j in range(len(subject)):
            y.append(subject[j][k])

# transformar os números de strings para int
for i in range(len(y)):

    try:
        y[i] = (float(y[i]))
    except:
        continue

for i in range(len(subject)):
    for j in range(len(subject[0])):

        try:
            subject[i][j] = (float(subject[i][j]))
        except:
            continue

# Criar uma lista em que cada elemento é uma lista com as colunas das variáveis
for j in range(len(subject[1])):
    var_total.append(y[(j * len(subject)):((j + 1) * len(subject))])

# criar um dicionário com a base de dados do tipo:  {'var' : [dados],...}
data = {}
data_group = {}
mediana = {}
percentil_25 = {}
percentil_75 = {}
shapiro_test = {}
levene = {}
p_value = {}
freq_abs = {}
freq_rel = {}
fisher = {}

for i in range(len(subject[1])):  # Adicionar as keys aos dicionarios. As keys passam a ser os nomes das variaveis
    # Criar um dicionário do tipo:   {'N Processo': [ [individuos nao autistas], [individuos autistas] ]......}
    data[var_total[i][0]] = var_total[i][1:len(subject)]
    data_group[var_total[i][0]] = [[], []]
    mediana[var_total[i][0]] = []
    percentil_25[var_total[i][0]] = []
    percentil_75[var_total[i][0]] = []
    shapiro_test[var_total[i][0]] = []
    levene[var_total[i][0]] = []
    p_value[var_total[i][0]] = []
    freq_abs[var_total[i][0]] = [[], []]
    freq_rel[var_total[i][0]] = [[], []]
    fisher[var_total[i][0]] = []

nao_autista = []
autista = []

for i in range(len(subject) - 1):
    if data['Grupo'][i] == 0:
        nao_autista.append(i)
    elif data['Grupo'][i] == 1:
        autista.append(i)

for j in range(len(subject[1])):
    for k in range(len(nao_autista)):
        data_group[var_total[j][0]][0].append(data[var_total[j][0]][nao_autista[k]])

    for k in range(len(autista)):
        data_group[var_total[j][0]][1].append(data[var_total[j][0]][autista[k]])

# ##############################################################################################
#                          ESTUDOS DAS VARIÁVEIS - SELEÇÃO DE FEATURES
# ##############################################################################################

print('{0:25} {1:^19} {2:^26}  {3:^4}'.format('VARIÁVEL', 'NÃO AUTISTA', 'AUTISTA', 'P-VALUE'))
a = []
for key, value in data_group.items():

    # VARIÁVEIS QUANTITATIVAS
    # Idade_paterna; Idade_materna; Primeiras_palavras; Primeiras_Frases; Marcha; Cont_Esf_Diurno; Cont_Esf_Nocturno; Latest_SCQ_MCHAT_Date

    if key == 'Idade paterna' or key == 'Idade materna' or key == 'Primeiras palavras' or key == 'Primeiras Frases' or key == 'Marcha' or key == 'Cont Esf Diurno' or key == 'Cont Esf Nocturno' or key == 'Latest_SCQ_MCHAT_Date':
        data[key] = [x for x in data[key] if type(x) != str]
        for w in [0, 1]:
            data_group[key][w] = [x for x in data_group[key][w] if type(x) != str]  # nao considerar os None ou ''


            # CALCULO DA MEDIANA
            mediana[key].append(statistics.median((data_group[key][w])))
            # CALCULO DO PERCENTIL
            (data_group[key][w]).sort()
            percen25 = np.percentile((data_group[key][w]), 25)
            percentil_25[key].append(percen25)

            percen75 = np.percentile((data_group[key][w]), 75)
            percentil_75[key].append(percen75)

            from scipy import stats

            # VERIFICAR NORMALIDADE - TEST SHAPIRO WILK
            shapiro_test[key].append(stats.shapiro(data_group[key][w])[1])  # o 1 corresponde ao p-value porque os outputs do shapiro sao (W : float, p-value : float)

        if shapiro_test[key][0] > 0.05 and shapiro_test[key][1] > 0.05:  # Verificar se é Normalmente distribuida
            levene[key].append(stats.levene(data_group[key][0], data_group[key][1], center='mean')[1])
            if levene[key][0] < 0.05:
                p_value[key].append(
                    stats.ttest_ind(data_group[key][0], data_group[key][1], equal_var=False)[1])  # TESTE T-STUDENT
            else:
                p_value[key].append(stats.ttest_ind(data_group[key][0], data_group[key][1])[1])
        else:
            p_value[key].append(stats.mannwhitneyu(data_group[key][0], data_group[key][1], use_continuity=False,
                                                   alternative='two-sided')[1])  # TESTE MANN WHITNEY

        print('{0:25} {1:6} ({2:^4};{3:^4})  {4:6} ({5:^4};{6:^4})       {7:5}'.format(key, mediana[key][0],percentil_25[key][0],percentil_75[key][0],mediana[key][1],percentil_25[key][1],percentil_75[key][1],p_value[key][0]))
        print('')
        print('--------------------------------------------------------------------------------------------------')

    elif key == 'N Processo':  # or key=='Grupo' or key == 'SCQ2' or key == 'SCQ3' or key == 'SCQ4' or key == 'SCQ5' or key == 'SCQ6' or key == 'SCQ7':
        continue

    # VARIÁVEIS QUALITATIVAS
    # Idade_paterna; Idade_materna; Primeiras_palavras; Primeiras_Frases; Marcha; Cont_Esf_Diurno; Cont_Esf_Nocturno; Latest_SCQ_MCHAT_Date
    # Sexo,Literacia_Pai,Literacia_Mae,Historia_Familiar_Posi, TODOS MCHAT, TODOS SCQ, Grupo]

    else:
        n_var = []
        data[key] = [x for x in data_group[key] if x != '']
        for w in [0, 1]:
            # Retirar os elementos em branco
            data_group[key][w] = [x for x in data_group[key][w] if x != '']

            # Adicionar à lista n_var o nome das diferentes opçoes de cada variável
            for i in range(len(data_group[key][w])):
                if data_group[key][w][i] not in n_var:
                    n_var.append(data_group[key][w][i])

        # Ordenar os elementos da n_var e calcular a freq absoluta

        n_var.sort()
        if len(n_var) == 1:
            # nos casos em que só tem uma variavel, tipo só tem zeros
            freq_abs[key][0].append(data_group[key][0].count(n_var[0]))
            freq_abs[key][0].append(0)
            freq_abs[key][1].append(data_group[key][1].count(n_var[0]))
            freq_abs[key][1].append(0)

        else:
            for k in range(len(n_var)):
                freq_abs[key][0].append(data_group[key][0].count(n_var[k]))
                freq_abs[key][1].append(data_group[key][1].count(n_var[k]))

        # Calcular a frequencia relativa
        for h in range(len(freq_abs[key][0])):
            freq_rel[key][0].append((((freq_abs[key][0][h]) * 1.0) / ((len(data_group[key][0]))) * 1.0) * 100)
            freq_rel[key][0][h] = float("{0:.2f}".format(freq_rel[key][0][h]))

        for h in range(len(freq_abs[key][1])):
            freq_rel[key][1].append((((freq_abs[key][1][h]) * 1.0) / ((len(data_group[key][1]))) * 1.0) * 100)
            freq_rel[key][1][h] = float("{0:.2f}".format(freq_rel[key][1][h]))

        # Calculo do metodo de Ficher
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr

        stats = importr('stats')

        v = robjects.IntVector(np.reshape(freq_abs[key], len(freq_abs[key][0]) * 2))
        print 'v', v
        m = robjects.r['matrix'](v, nrow=len(freq_abs[key][0]))
        print 'm', m
        res = stats.fisher_test(m)
        #print 'res', res
        p_value[key].append(float("{0:.19f}".format(res[0][0])))

        #Imprimir os dados
        print('{0:25} '.format(key))
        for k in range(len(n_var)):
           if k==0:
               print('{0:<25} {1:6}({2:^4}%)           {3:6}({4:^4}%)         {5:^5}'.format(n_var[k], freq_abs[key][0][k],freq_rel[key][0][k],freq_abs[key][1][k],freq_rel[key][1][k],p_value[key][0]))
           else:
               print('{0:<25} {1:6}({2:^4}%)           {3:6}({4:^4}%)'.format(n_var[k], freq_abs[key][0][k],freq_rel[key][0][k], freq_abs[key][1][k],freq_rel[key][1][k]))
        print('')
        print('--------------------------------------------------------------------------------------------------')

    # -----------------------------------------------------------------------------------------------
    #                                     PREPARAÇÃO DAS VARIÁVEIS PARA CLASSIFICADORES
    # -----------------------------------------------------------------------------------------------
    # Eliminar as colunas do SCQ 1 até SCQ7 porque tem grande falha de dados

    if key == 'SCQ2' or key == 'SCQ3' or key == 'SCQ4' or key == 'SCQ5' or key == 'SCQ6' or key == 'SCQ7':  # Elimina o conteudo das células dos individuos correspondentes às colunas
        del data[key]
        for q in range(len(subject[0]) - 1):
            if key == subject[0][q]:
                for w in range(len(subject)):
                    del subject[w][q]

    elif p_value[key][0] < 0.05 and p_value[key] != []:
        a.append(key)  # a lista 'a' contem o nome das variaveis significativas

    else:  # Elimina o conteudo das células dos individuos correspondentes às colunas das condiçoes contrarias do if anterior
        del data[key]
        for q in range(len(subject[0]) - 1):
            if key == subject[0][q]:
                for w in range(len(subject)):
                    del subject[w][q]  # subject contem agora os dados dos individuos excepto os dados das variaveis nao significativos

subject_grupo = np.copy(subject)

# ##############################################################################################
#                          IMPUTAÇÃO DE DADOS- REGRESSAO LIENAR
# ##############################################################################################
features=[]

features = np.copy(subject)  # fazer uma copia do subject mas resulta um ndarray - trabalhado no numpy
name_features = np.copy(subject[0][1:-1])  # faz uma copia dos nomes das variáveis (primeira linha do subject) e retira o primeio e ultimo ('n processo' e o grupo)

features = np.delete(features,0,1)  # elimina a primeira coluna que corresponde aos nº dos processos
features = np.delete(features,-1,1)  # elimina a última coluna que corresponde tipo de grupo - autista ou não

print 'Número de variaveis iniciais- ', len(features[0])  # NUMERO DE variaveis ANTES DE ELIMINAR POR MISSINGS

compl_features = [] # lista que conterá o número das features sem missings
position_del = []  # vai conter o número das variáveis a eliminar por ter mais de 30% de missings
position_nan = []  # vai conter posição das variaveis com missings

features[features == ""] = np.NaN  # substitui todos os espaços vazios por nan
features = features[1:].astype(float)  # transforma todos os elementos excepto a primeira linha (nome das var) em float

for i in range(len(features[0])):
    m = 0  # lista que conterá o numero de missings por variável
    m = sum(np.isnan(features[:, i]))  # soma os missings em cada coluna - variável
    # print 'variavel- ', i
    # print (m * 1.0) / len(features[:,i])

    if (m * 1.0) / len(features[:,i]) > 0.30:  # se tem mais de 30% de missings
        position_del.append(i)  # adiciona à lista o numero das variáveis
    elif (m * 1.0) == 0:  # se não tem missings
        compl_features.append(i)  # adiciona à lista o numero das variáveis
    else:  # restantes que correspondem às variaveis com missings entre 0 e 30%
        position_nan.append(i)  # adiciona à lista o numero das variáveis

print 'Número de variáveis com > 30% missings- ', len(position_del)  # NUMERO DE  variaveis A ELIMINAR com mais de 30% de missings
print 'Número de variáveis sem missings- ', len(compl_features)
print 'Número de variáveis com < 30% missings- ', len(position_nan)

print 'Variáveis com < 30% missings: '
for i in range(len(position_nan)):
    print name_features[position_nan[i]]  # imprime o nome das variáveis com menos de 30% missings

#  elimina os elementos das listas dos individuos a quem faltam dados
for k in range(len(position_del)):
    features = np.delete(features,position_del[k], 1)  # elimina da matriz principal as colunas das variáveis com mais de 30 % de missings

features_pd = pd.DataFrame(features)  # transforma o ndarray (numpy) num dataframe (pandas)

correlation = features_pd.corr(method='spearman')  # cria a matriz das correlações (usado pandas porque julgo que o numpy nao aceita missings)

correlation.values[[np.arange(len(correlation[0]))]*2] = 0  # substitui a diagonal por zeros para a propria variável nao ser identificada como muito correlacionada

sns.heatmap(correlation)
plt.show()

features_imputed = features_pd.copy()  # cria-se a matriz onde serão substituidos os missings pelos valores gerados

for k in range(len(position_nan)):

    position_var = []  # conterá o numero das variáveis correlacionadas
    sorted_corr = sorted(correlation[position_nan[k]], reverse=True)[:3]  # seleciona-se as 3 variáveis mais correlacionadas

    for i in range(len(correlation)):
        # procura a posição dos valores mais altos de correlação na matriz geral da correlação
        if correlation[k][i] == sorted_corr[0] or correlation[k][i] == sorted_corr[1] or correlation[k][i] == sorted_corr[2]:

            if i in position_nan:  # não adiciona se a variável correlacionada também tiver missings
                continue
            else:
                position_var.append(i)  # adiciona-se o numero das variáveis mais correlacionadas

    features_train = features_pd[~features_pd.iloc[:,position_nan[k]].isnull()]  # separa as linhas da matriz total de dados que não tenham missings - na variável com missing e em todas
    features_test = features_pd[features_pd.iloc[:,position_nan[k]].isnull()]  # separa as linhas da matriz total de dados que tenham missings - na variável com missing e em todas
    index_values = features_test.index.values  # adiciona a posição dos missings na matriz
    features_X_train = features_train.iloc[:,position_var]  # cria matriz com os dados das variáveis correlacionadas com a variável dos missings
    features_y_train = features_train[position_nan[k]]  # cria matriz coluna com os dados sem missings da variável em análise
    features_X_test = features_test.iloc[:,position_var]  # cria matriz com os dados das variáveis correlacionadas correspondentes aos dados dos missings

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(features_X_train, features_y_train)

    # Make predictions using the testing set
    features_y_pred = regr.predict(features_X_test)

    features_imputed.iloc[idx[index_values, position_nan[k]]] = features_y_pred

# ##############################################################################################
#                          CLASSIFICAÇÃO
# ##############################################################################################

group = np.copy(subject_grupo[1:, -1])
group = group[:].astype(float)

features_group = pd.DataFrame.as_matrix(features_imputed)

features_group = np.insert(features_group,len(features_group[0]), [group], axis=1)


X_0_subject = []
X_1_subject = []

# 0 --> Nao autista
# 1 --> Autista

X_0_subject = features_group[features_group[:,-1]==0]   # lista de elementos nao autistas
X_1_subject = features_group[features_group[:, -1] == 1]  # lista de elementos autistas

# print len(X_0_subject)
# print len(X_1_subject)

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

for i in range(100):  # correr 100x

    new_subject = []
    labels = []
    a=0
    b=0
    # -----------------------------------------------------------------------------------------------
    #                                     BALANCEAMENTO
    # -----------------------------------------------------------------------------------------------

    while len(new_subject) != len(X_0_subject):  # enquanto a nova lista nao tiver o mesmo numero que o total de nao autista continua o ciclo
        c = random.randrange(len(features_group))  # procura um elemento aleatorio do suject_grupo

        if features_group[c][len(features_group[c]) - 1] == 0:  # se esse elemento for nao autista adiciona à nova matriz
            a=a+1
            new_subject.append(features_group[c][: (len(features_group[c]) - 1)])
            labels.append(features_group[c][(len(features_group[c]) - 1)])

    while len(new_subject) != 2 * len(X_0_subject):  # enquanto a nova lista nao tiver o mesmo numero de autista como de nao autistas continua o ciclo
        c = random.randrange(len(features_group))  # procura um elemento aleatorio do suject_grupo
        if features_group[c][len(features_group[c]) - 1] == 1:  # se esse elemento for autista adiciona à nova matriz
            b=b+1
            new_subject.append(features_group[c][: (len(features_group[c]) - 1)])
            labels.append(features_group[c][(len(features_group[c]) - 1)])


    new_subject = np.array(new_subject)

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    x_train, x_test, y_train, y_test = train_test_split(new_subject, labels, test_size=0.30, random_state=None)

    # print "x_train", len(x_train), x_train
    # print "x_test", len(x_test), x_train
    # print "y_train", len(y_train), y_train
    # print "y_test", len(y_test), y_test

    # -----------------------------------------------------------------------------------------------
    #                                     SVM
    # -----------------------------------------------------------------------------------------------

    model = svm.SVC(kernel='linear')
    model.fit(x_train, y_train)
    # model.score(Xtrain_total, y_train)
    predicted = model.predict(x_test)

    confusion = confusion_matrix(y_test, predicted)
    confusion = np.asmatrix(confusion)

    accuracy_svm.append(accuracy_score(y_test, predicted))
    sensitivity_svm.append(float(confusion[0, 0]) / (float(confusion[0, 0]) + float(confusion[0, 1])))
    specificity_svm.append(float(confusion[1, 1]) / (float(confusion[1, 0]) + float(confusion[1, 1])))

    fpr, tpr, threshold = roc_curve(y_test, predicted)
    roc_svm.append(auc(fpr, tpr))

    # ROC Curve
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_svm[i])
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

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


print 'Número de individuos considerados no classificador (autistas e não autistas)- ', a, b

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





















# # ##############################################################################################
# #                          CLASSIFICAÇÃO
# # ##############################################################################################
# subject_grupo= copy.copy(subject)
#
#
# X_0_subject = []
# X_1_subject = []
#
# # 0 --> Nao autista
# # 1 --> Autista
#
# for k in range(len(subject_grupo)):
#     if subject_grupo[k][- 1] == 0:
#         X_0_subject.append(subject_grupo[k])  # lista de elementos nao autistas
#     else:
#         X_1_subject.append(subject_grupo[k])  # lista de elementos autistas
#
# print len(X_0_subject)
# print len(X_1_subject)
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
#
# for i in range(1):
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
#         c = random.randrange(len(subject_grupo))  # procura um elemento aleatorio do suject_grupo
#
#         if subject_grupo[c][len(subject_grupo[c]) - 1] == 0:  # se esse elemento for nao autista adiciona à nova matriz
#             a=a+1
#             new_subject.append(subject_grupo[c][: (len(subject_grupo[c]) - 1)])
#             labels.append(subject_grupo[c][(len(subject_grupo[c]) - 1)])
#
#     while len(new_subject) != 2 * len(X_0_subject):  # enquanto a nova lista nao tiver o mesmo numero de autista como de nao autistas continua o ciclo
#         c = random.randrange(len(subject_grupo))  # procura um elemento aleatorio do suject_grupo
#
#         if subject_grupo[c][len(subject_grupo[c]) - 1] == 1:  # se esse elemento for autista adiciona à nova matriz
#             b=b+1
#             new_subject.append(subject_grupo[c][: (len(subject_grupo[c]) - 1)])
#             labels.append(subject_grupo[c][(len(subject_grupo[c]) - 1)])
#
#     print a
#     print b
#
#     x_train = []
#     x_test = []
#     y_train = []
#     y_test = []
#
#     x_train, x_test, y_train, y_test = train_test_split(new_subject, labels, test_size=0.30, random_state=None)
#
#     print "x_train", len(x_train), x_train
#     print "x_test", len(x_test), x_train
#     print "y_train", len(y_train), y_train
#     print "y_test", len(y_test), y_test
#
