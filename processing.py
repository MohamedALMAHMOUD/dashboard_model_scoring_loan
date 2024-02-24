# # I. Importation
# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import requests
import gc
import time
from contextlib import contextmanager
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', 250)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#!mkdir ~/.kaggle

#!cp /Users/mohamedalmahmoud/kaggle.json /Users/mohamedalmahmoud/.kaggle/kaggle.json

#!cd /Users/mohamedalmahmoud/.kaggle/ && ls

#!kaggle competitions download -c home-credit-default-risk

# # Extraction zip
# from zipfile import ZipFile
# dataset = "/Users/mohamedalmahmoud/P7/home-credit-default-risk.zip"

# with ZipFile(dataset, 'r') as zip:
#     zip.extractall()
#     print('Dataset est décompressé')

# # II. Liste des fichiers et des variables 

path = "/Users/mohamedds/Desktop/P7/files/"
datas = []
for file in os.listdir(path):
    print(file)                     # Print the name of file
    globals()['file'] = pd.read_csv(path+file, encoding_errors='ignore') # Generate a pandas variable
    datas.append(globals()['file']) # Add variables to list

# # III. L'exploration et nettoyage de train et test dataframe

# L'exploration et le nettoyage de données va dans le sens de fusionner 8 tables. 4 tables (install_payments, credit_card_balance, POS_CASH_balance, previous_application) peuvent se fusionner selon 2 colonnes reférencielles (SK_ID_PREV et SK_ID_CURR). Or, le schemas préconise la fusion selon l'id SK_ID_PREV. Les 4 dernières tables se fusionnent selon l'id SK_ID_CURR. Dans ce context, je peux supprimer des identifants en trop.

# ## III.1. Table principale
# * Target (variable cible)
# * infos sur le prêt et le demadeur du prêt

# ### III.1.1. Informations générales

# #### Aperçu du dataframe

app_train = datas[5]
app_train

# #### Aperçu général des valeurs manquantes

#pip install missingno
import missingno as msno 
msno.matrix(app_train)

app_train.isna().mean().sort_values()

app_train.describe()

# ### III.1.2. Les variabes non numeriques 

# Loop for plot.pie of no numeric features
col_cat = [col for col in app_train.select_dtypes(include=object)]
#  Categorical Data
#b = 4  # number of columns
#a = len(col_cat)//b  # number of rows
c = 1  # initialize plot counter 
plt.figure(figsize=(20, 15))
for i in col_cat: 
    plt.subplot(4, 4, c)
    app_train[i].value_counts().plot.pie()
    c = c + 1
plt.show()   

# Un constat à chaud montre que les variables suivantes: 
# * EMERGRNCYSTATE_MODE
# * HOUSETY_MODE  
# 
# inutiles car la variation de leurs valeurs est quasi nulle

app_train = app_train.drop(['HOUSETYPE_MODE', 'EMERGENCYSTATE_MODE'], axis=1)

# ### III.1.3. Les variabes numeriques 

# #### Liste des variables numériques

list(app_train.select_dtypes("number"))

# #### Des distributions des variables

app_train.hist(bins=50, figsize=(35,45))
plt.show()

# A chaud également les variables numériques suiantes sont inutiles faute de variation:
#     * AMT_INCOME_TOTAL
#     * FLAG_MOBIL
#     * FLAG_EMAIL
#     * FLAG_CONT_MOBIL
#     * HOUR_APPR_PROCESS_START
#     * REG_REGION_NOT_LIVE_REGION
#     * REG_REGION_NOT_WORK_REGION
#     * LIVE_REGION_NOT_WORK_REGION
#     * REG_CITY_NOT_LIVE_CITY
#     * NONLIVINGAPARTMENTS_AVG
#     * NONLIVINGAPARTMENTS_MEDI
#     * OBS_30_CNT_SOCIAL_CIRCLE
#     * OBS_60_CNT_SOCIAL_CIRCLE
#     * FLAG_DOCUMENT_2
#     * FLAG_DOCUMENT_4
#     * FLAG_DOCUMENT_5
#     * FLAG_DOCUMENT_6
#     * FLAG_DOCUMENT_7
#     * FLAG_DOCUMENT_8
#     * FLAG_DOCUMENT_9
#     * FLAG_DOCUMENT_10
#     * FLAG_DOCUMENT_11
#     * FLAG_DOCUMENT_12
#     * FLAG_DOCUMENT_13
#     * FLAG_DOCUMENT_14
#     * FLAG_DOCUMENT_15
#     * FLAG_DOCUMENT_16
#     * FLAG_DOCUMENT_17
#     * FLAG_DOCUMENT_18
#     * FLAG_DOCUMENT_19
#     * FLAG_DOCUMENT_20
#     * FLAG_DOCUMENT_21
#     * AMT_REQ_CREDIT_BUREAU_HOUR
#     * AMT_REQ_CREDIT_BUREAU_DAY
#     * AMT_REQ_CREDIT_BUREAU_WEEK
#     * AMT_REQ_CREDIT_BUREAU_HOUR
#     * AMT_REQ_CREDIT_BUREAU_QRT
#     

# #### Un diagramme circulaire pour la variable target

plt.rcParams['font.size'] = '12'
plt.figure(figsize=(8,8))
app_train['TARGET'].value_counts().plot.pie(autopct='%.2f%%')
plt.show()

# La variable cible est 0 si le client rembourse le prêt et 1 si le client fait défaut. Il est clair en regardant le graphique que l'ensemble de données est déséquilibré.

# #### Des courbes de comparaison entre le target et les autres variables numériques

sns.displot(data= app_train, x = 'AMT_CREDIT', kind="kde", hue = 'TARGET')
plt.show()

# * La densité est élevée pour le montant inférieur à 10⁶ pour les deux types de demandeurs qui peuvent payer et ont des difficultés à rembourser les prêts.
# * Le graphique semble asymétrique à droite.

sns.displot(data= app_train, x = 'AMT_CREDIT', kind="kde", hue = 'NAME_FAMILY_STATUS')
plt.show()

# Nettoyage possible à faire : fusionner Married et Civil marriage, supprimer Unknown

app_train['NAME_FAMILY_STATUS'] = app_train['NAME_FAMILY_STATUS'].replace(["Civil marriage"], "Married")
app_train = app_train[app_train['NAME_FAMILY_STATUS']!="Unknown"]


sns.displot(data= app_train, x = 'AMT_CREDIT', kind="kde", hue = 'NAME_FAMILY_STATUS')
plt.show()

sns.displot(data= app_train, x = 'AMT_CREDIT', kind="kde", hue = 'NAME_CONTRACT_TYPE')
plt.show()

sns.displot(data= app_train, x = 'AMT_CREDIT', kind="kde", hue = 'CODE_GENDER')
plt.show()

app_train = app_train[app_train['CODE_GENDER']!='XNA']
sns.displot(data= app_train, x = 'AMT_CREDIT', kind="kde", hue = 'CODE_GENDER')
plt.show()

sns.displot(data= app_train, x = 'AMT_INCOME_TOTAL', kind="kde", hue = 'TARGET')
plt.show()

# ### Etudier en détail cette variable de revenu total
# Il y a certainement des valeurs abberantes qui obligent la courbe à agir de telle façon. 

sns.boxplot(data= app_train, x = 'AMT_INCOME_TOTAL')

# On voit que la valeur 1.2*e8 est une valeur très éloignée. Je vérifie si un crédit a été accordé pour ce revenu.

AMT_INCOME_TOTAL_HIGH = app_train.loc[app_train['AMT_INCOME_TOTAL']>1.1*1e8, :]
AMT_INCOME_TOTAL_HIGH

# On voit bien que malgé le revenu important la demande de crédit a été rejetée. Ce qui est encore bizarre est que le demandeur travaille comme ouvrier et dispose ces revenus. Donc, il s'agit d'une valeur abberante à supprimer.

app_train = app_train.loc[app_train['AMT_INCOME_TOTAL']<1.1*1e8]

sns.displot(data= app_train, x = 'AMT_INCOME_TOTAL', kind="kde", hue = 'TARGET')
plt.show()

# Après 0.25*1e6 on constate que la distibution est nulle, donc on essaie de filtrer encore.

app_train = app_train.loc[app_train['AMT_INCOME_TOTAL']<1*1e6]
sns.displot(data= app_train, x = 'AMT_INCOME_TOTAL', kind="kde", hue = 'TARGET')
plt.show()

# Il se voit que les deux courbes ne se chauveuchent pas donc on peut dire que cette variable serait utile

# #### Variables concernant l'âge et l'expérience de job actuel en jours

sns.displot(data= app_train, x = 'DAYS_BIRTH', kind="kde", hue = 'TARGET')
sns.displot(data= app_train, x = 'DAYS_EMPLOYED', kind="kde", hue = 'TARGET')
plt.show()

# * Il y a une majorité de groupe d'âge (30-60) ans qui peuvent rembourser leur prêt.
# * Une valeur aberrante se manifeste à 365000 jours pour la durée d'emploi actuel avant la demande de crédit
# C'est certes une valeur abberante, mais on attend la fusion des tables pour supprimer cette valeur.

app_train['DAYS_EMPLOYED'].describe()

# * Il est visible que la densité des clients qui ont moins de 10 ans d'expérience ont des difficultés à rembourser les prêts.
# * Il n'y a pas de chevauchement dans la densité maximale de deux valeurs cibles. Donc, cette variable est utile.

# #### Quelques graphiques qui montrent les correlations entre variables numériques

plt.figure(figsize=(20, 15))
sns.heatmap(app_train.corr(method='pearson')[['TARGET']].sort_values('TARGET').tail(20),
            vmax=1, vmin=-1, cmap='YlGnBu', annot=True)

plt.figure(figsize=(20, 15))
sns.heatmap(app_train.corr(method='spearman')[['TARGET']].sort_values('TARGET').tail(20),
            vmax=1, vmin=-1, cmap='YlGnBu', annot=True)

# plt.figure(figsize=(20, 15))
# sns.heatmap(app_train.corr(method='kendall')[['TARGET']].sort_values('TARGET').tail(20),
#             vmax=1, vmin=-1, cmap='YlGnBu', annot=True)

# En regardant les heatmaps de correlation, je constate à chaud quelques correlations:
#     * OWN_CAR_AGE
#     * DAYS_BIRTH
#     * DAYS_EMPLOYED. 
# 
# Ces variables sont en lien directe avec le métier des banques et la distribution des prêts. Les autres varibles sont moins mentionnées dans les sites d'accordement du crédit.

app_train.to_csv("filles_after_data_processing/app_train.csv")

# ## III.2. Table de test
# C'est les mêmes variables que la table précédente mais sans target.

app_test = datas[0]
app_test

app_test.describe()

app_test.isna().mean().sort_values()

app_test.to_csv("filles_after_data_processing/app_test.csv")

# ## III.3. Table des soldes mensuels des prêts précédents du client dans le bureau de Crédit Immobilier

POS_CASH_balance = datas[2]
POS_CASH_balance

POS_CASH_balance.describe()

POS_CASH_balance.isna().mean()

POS_CASH_balance['SK_DPD_DEF'].value_counts().sort_values()

POS_CASH_balance.hist(bins=100, figsize=(30,25))
plt.show()

plt.figure(figsize=(15,12))
POS_CASH_balance['NAME_CONTRACT_STATUS'].value_counts().plot.pie(autopct='%.2f%%')
plt.show()

POS_CASH_balance.to_csv("filles_after_data_processing/POS_CASH_balance.csv")

# ## III.4. Table de solde de la carte de crédit

credit_card = datas[3]
credit_card

credit_card.isna().mean()

credit_card.to_csv("filles_after_data_processing/credit_card_balance.csv")

# ## III.5. Table de paiements échelonnés

instal_payments = datas[4]
instal_payments

instal_payments.info()


instal_payments.isna().mean()

instal_payments['SK_ID_PREV'].nunique()

instal_payments.to_csv("filles_after_data_processing/instal_payments.csv")

# ## III.6.  Table de la demande précédente

previous_app = datas[7]
previous_app

previous_app.describe()

round(previous_app.isna().mean(), 2)

# ### III.6.1. Variables categorielles

for i in previous_app.select_dtypes(include=object):
    plt.figure(figsize=(10,10))
    previous_app[i].value_counts().plot.pie(autopct='%1f%%')
    plt.show()

# La variation de la variable "FLAG_LAST_APPL_PER_CONTRACT" est quasi nuelle, donc elle n'est pas utile. Egalement, Ce qui m'intéresse dans la variable NAME_CONTRACT_STATUS est les prêts approuvés et refusés. Les autres prêts, notamment annulés seront filtrés car je suppose que les clients aient annulé leur demande même avant la décision. en outre, faire juste 2 catégories: Approved et refuser. Autrement dit, Ce qui fait recours à une variable target (catégorielle) pour cette table. Concernant la variable NAME_CLIENT_TYPE, une valeur indéterminée : XNA a un taux très faible, donc à filtrer. A vérifier si la vrariable NAME_TYPE_SUITE est utile ou non en comparant avec la variable NAME_CONTRACT_STATUS. A propos de la variable WEEKDAY_APPR_PROCESS_START on voit que les différentes catégories de la variable on un taux quasi identiques (15%), juste le dimanche et c'est logique: c'est un jour de repos. Donc la variation n'est pas significative, à ignorer.

previous_app = previous_app.drop(['FLAG_LAST_APPL_PER_CONTRACT', 'WEEKDAY_APPR_PROCESS_START'], axis=1)

previous_app = previous_app[previous_app['NAME_CONTRACT_STATUS']!='Canceled']
previous_app.shape

previous_app = previous_app[previous_app['NAME_CONTRACT_STATUS']!='Unused offer']
previous_app['NAME_CONTRACT_STATUS'] = previous_app['NAME_CONTRACT_STATUS'].astype('category')


previous_app['NAME_CONTRACT_STATUS'].value_counts().plot.pie(autopct='%1f%%')
plt.show()

# Si on compare cette variable à la variable target de la table application_train, on constate que les demandes de prêts ont reçu davantage un avis défavorable: 21% ici contre 9% dans l'autre.

previous_app = previous_app[previous_app['NAME_CLIENT_TYPE']!='XNA']
previous_app.shape

# On cherche à trouver des corrélations entre les variables catégorielles.

# Visualize the relationship between categorical variables
for i in previous_app.select_dtypes(include=object):
    previous_app_croostab = pd.crosstab(index=previous_app[i],
                                        columns=previous_app['NAME_CONTRACT_STATUS'])
    previous_app_croostab.plot.bar(figsize=(9,6))

# #### Quelques résultats
# D'après les graphiques ci-dessus, on constate que certaines variables ne sont pas corrélées à la variable 'NAME_CONTRACT_STATUS', notamment lorsque la variation des valeurs sont quasi identiques comme la variable 'NAME_GOODS_CATEGORY', 'NAME_TYPE_SUITE' et 'NAME_PAYMENT_TYPE'. Donc, ces variables ne sont pas à utiles à supprimer.
# D'ailleurs, on distingues les valeurs qui recoivent des avis favorables et défavorables:
# * les 'consumer loans' reçoivent davantages des avis favorables.
# * Les XAP sont egalement favorables
# * HC , LIMIT et SCO sont défavorables
# * New clients sont plus favorables que les autres
# * POS: point of sale ou point de vente sont préférés, notamment pos mobile without interest et pos household with interest
# * Country-wide est également approuvé, par contre credit and cash offices ont plus de refus
# * Connectivity and consumer electronics sont plus favorables ques XNA dans la variable NAME_SELLER_INDUSTRY
# 

# Remove useless features 
previous = previous_app.\
drop(['NAME_GOODS_CATEGORY', 'NAME_TYPE_SUITE', 'NAME_PAYMENT_TYPE'], axis=1)
previous.shape

# ### III.6.2. Variables numériques

for i in previous_app.select_dtypes(include= 'number'):
    sns.displot(data= previous_app, x = i, kind="kde", hue = 'NAME_CONTRACT_STATUS')
    plt.show()

# Les graphiques ci-dessus montrent déjà qu'il y a pas mal de outiliers à determiner avec les boites à moustache. Il s'agit toujours des jours (365000) qui coresspondent à 999 ans. Cela représente une valeur aberrante qui va être traitée lors de la fusion des tables.

for i in previous_app.select_dtypes(include= 'number'):
    sns.boxplot(data= previous_app, x = i, y = 'NAME_CONTRACT_STATUS')
    plt.show()

# Avant d'étudier la corrélation entre les variable, j'essaie de traiter les outliers.

previous_app = previous_app[(previous_app['AMT_ANNUITY']<3*1e5)]

for i in previous_app.select_dtypes(include= 'number'):
    sns.catplot(data= previous_app, x = i, y = 'NAME_CONTRACT_STATUS', kind="bar")
    plt.show()



# #### Remarques: 
# toutes variables commancent par AMT a un effet negatif

previous_app.to_csv("filles_after_data_processing/previous_app.csv")

# ## III.7. Table du bureau de crédit

bureau = datas[6]
bureau

# Exemple d'un prêt
bureau[bureau['SK_ID_CURR']==215354]

# On constate que'un client enregistre 11 prêts. 5 ont été remboursés et 6 sont en cours. 

bureau.describe()

bureau.isna().mean()

# La variable AMT_ANNUITY comporte des données manquantes importantes mais elle se trouve dans des autres tables (application train/test, bureau, previous application). 

# Comme la variable contient + 70% des valeurs manquantes et elle se trouve dans des autres tables, j'opte de la supprimer.

bureau = bureau.drop('AMT_ANNUITY', axis=1)

# ### III.7.1. Variables catégorielles

for i in bureau.select_dtypes(include=object):
    plt.figure(figsize=(6,6))
    bureau[i].value_counts().plot.pie(autopct='%.2f%%')
    plt.show()

# La variable CREDIT_CURRENCY n'est pas utile car elle comporte presque une seule valeur: currency 1. Donc à supprimer

bureau_clean = bureau.drop('CREDIT_CURRENCY', axis=1)

# <!-- Il faut distinguer, ici, entre deux types de crédit: garanti et non garanti. Le premier est garanti par l'état, c'est un prêt destiné à une entreprise ou à des professionnels. Cette distingution me permet de comprendre le comportement des clients: physiques ou morals. -->



# A regarder en détail les variables suivantes:

var_bureau = ['CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG', 
              'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT',
              'AMT_CREDIT_SUM_OVERDUE']

bureau[var_bureau].hist(bins=100, figsize=(35,45))
# for i in var_bureau:
#     bureau[i].value_counts().plot.pie(autopct="%.2f%%")
#     plt.show()

# 0.99% des valeurs de ces variables ci-dessus sont égales à 0 dans la variation est presque nulle, donc elle n'est pas utile. A supprimer

bureau = bureau.drop(['CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG', 'AMT_CREDIT_SUM_OVERDUE'], axis=1)
bureau

sns.displot(data= bureau, x = 'AMT_CREDIT_SUM', kind="kde", hue = 'CREDIT_ACTIVE')
plt.show()

sns.boxplot(data = bureau, x= 'AMT_CREDIT_SUM')
plt.show()

# On constate selon les deux garphiques ci-dessus qu'il y a des valeurs aberrantes dans cette variable à partir de 0,2 e8

bureau = bureau[bureau['AMT_CREDIT_SUM']<0.5*1e6]
bureau

sns.displot(data= bureau, x = 'AMT_CREDIT_SUM', kind="kde", hue = 'CREDIT_ACTIVE')
plt.show()

sns.boxplot(data = bureau, x= 'AMT_CREDIT_SUM')
plt.show()

bureau.to_csv("filles_after_data_processing/bureau.csv")

# ## III.8. Table de solde au bureau de crédit

bureau_balance = datas[8]
bureau_balance

bureau_balance['STATUS'].value_counts()

# STATUS : C(closed ou fermé), 0 (pas de retard de remboursement du prêt), 1( un mois ou 30 jours de retard), 2 (2 mois ou 60 jours de retard), etc. X (status inconnu). Il serait plus utile de regrouper et/ou aggréger les données en fonction de la variable SK_ID_BUREAU. C'est pour ça l'one hote est plus utile.

sns.displot(data= bureau_balance, x= 'MONTHS_BALANCE')
plt.show()

sns.boxplot(data= bureau_balance, x= 'MONTHS_BALANCE')
plt.show()

# MONTHS_BALANCE : -1 (un mois avant la date de la demande), -2 (2 mois avant la date de la demande) ainsi de suite. -60 veut dire 5 ans avant la date de la demande du prêt. Normalement, on se base sur les 5 dernières années. Donc, je prends les valeurs supérieures à -60 car en regardant la boxplot, on veut que 90% des données sont inférieures à -60.

bureau_balance = bureau_balance[bureau_balance['MONTHS_BALANCE']>-60]
sns.displot(data= bureau_balance, x= 'MONTHS_BALANCE')
plt.show()

bureau_balance['STATUS'].value_counts()

bureau_balance.to_csv("filles_after_data_processing/bureau_balance.csv")

sample_submission = datas[9]
sample_submission

# # IV. Fusion des dataframes

# ## IV.1. Fonctions

# Define a timer
@contextmanager
def timer(title):
    t0 = time.time()
    yield # generator of object
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows = None, nan_as_category = False):
    # Read data and merge
    df = pd.read_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/app_train.csv', nrows= num_rows)
    test_df = pd.read_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/app_test.csv', nrows= num_rows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index() # Concatination of train and test
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature]) # [['M', 'F', ...], ['N', 'Y', ...], ['Y', 'N', ...] ==>
                                                                # df[bin_feature] = array([0, 1, ...], [0, 1,..], [0, 1, ...])
                                                                # unique = ['M', 'F', 'Y', 'N']
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    del test_df
    gc.collect() # Garbage Collection
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance(num_rows = None, nan_as_category = True):
    bureau = pd.read_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/bureau.csv', nrows = num_rows)
    bb = pd.read_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/bureau_balance.csv', nrows = num_rows)
    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        #'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        #'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        #'AMT_ANNUITY': ['max', 'mean'],
        #'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']}
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg, bureau
    gc.collect()
    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications(num_rows = None, nan_as_category = True):
    prev = pd.read_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/previous_app.csv', nrows = num_rows)
    prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        #'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash(num_rows = None, nan_as_category = True):
    pos = pd.read_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/POS_CASH_balance.csv', nrows = num_rows)
    pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg

# Preprocess installments_payments.csv
def installments_payments(num_rows = None, nan_as_category = True):
    ins = pd.read_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/instal_payments.csv', nrows = num_rows)
    ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
        'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']}
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance(num_rows = None, nan_as_category = True):
    cc = pd.read_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/credit_card_balance.csv', nrows = num_rows)
    cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg

def main(debug = False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    with timer("Process bureau and bureau_balance"):
        bureau = bureau_and_balance(num_rows)
        print("Bureau df shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Process previous_applications"):
        prev = previous_applications(num_rows)
        print("Previous applications df shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Process POS-CASH balance"):
        pos = pos_cash(num_rows)
        print("Pos-cash balance df shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Process installments payments"):
        ins = installments_payments(num_rows)
        print("Installments payments df shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Process credit card balance"):
        cc = credit_card_balance(num_rows)
        print("Credit card balance df shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        df[df==np.inf] = np.nan 
        del cc
        gc.collect()
        df.to_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/df.csv', index=False)
    
if __name__ == "__main__":
    submission_file_name = "submission_sample.csv" # Save prediction
    with timer("Full model run"):
        main()




