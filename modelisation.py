# Import necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import joblib
from lightgbm import LGBMClassifier
from yellowbrick.classifier import DiscriminationThreshold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, fbeta_score, make_scorer
from sklearn.model_selection import KFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
import time
from contextlib import contextmanager
from PIL import Image

df = pd.read_csv('/Users/mohamedds/Desktop/P7/filles_after_data_processing/df.csv')
df

df = df.drop(['index', 'Unnamed: 0'], axis=1)

df = df.drop_duplicates(subset=['SK_ID_CURR'])
df

df.isna().mean()

# Comme on a plus de 750 variables, il faut mieux supprimer celles dont elles ont plus de 50% de valeurs manquantes.

# Function for filtring mising data features
def fct_col_isna(df, per_nan):
    col_isna = []
    for i in df.columns:
        if df[i].isna().mean()>per_nan:
            col_isna.append(i)
    return col_isna
feat_drop = fct_col_isna(df, 0.51)
len(feat_drop)

df = df.drop(feat_drop, axis=1)
df

y = df.TARGET
x= df.drop('TARGET', axis=1)

# # Imputation

imput = SimpleImputer()
x_imputed = imput.fit_transform(x)
x_dataframe= pd.DataFrame(x_imputed, columns= x.columns)

x_dataframe

# # Conversion des valeurs négatives

x_pos = abs(x_dataframe)
x_pos.head()

# # Conversion de certaines variabes, leurs valeurs données en en jours en années comme DAYS_BIRTH et DAYS_EMPLOYED

x_pos['YEARS_BIRTH'] =  x_pos['DAYS_BIRTH']//365
x_pos['YEARS_EMPLOYED'] =  x_pos['DAYS_EMPLOYED']//365

x_pos = x_pos.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1)

data = pd.concat([x_pos, y], axis=1)
data

# # Verfication des nfinity datas et les noms des variables

np.all(np.isfinite(data))
import re
data = data.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

# # Separation de data

df_train = data[data['TARGET'].notnull()]
X = df_train.iloc[:,: -1]
print("dimensions X:", X.shape)
y = df_train['TARGET']
print("dimensions y:",y.shape)
df_test = data[data['TARGET'].isnull()]
X_test = df_test.iloc[:,:-1]
y_test = df_test['TARGET']
print("dimensions X_test:",X_test.shape)
print("dimensions y_test:",y_test.shape)

# # Modèle
# D'après mes lectures sur la problèmatique, notamment le site de Kaggle où une compétition a eu lieu sur le sujet, on remarque que l'algorithme Lightgbm fonctionne mieux et donne des bons résultats. Donc, ce n'est pas la peine de tester des autres algirithmes comme XGBoost, notamment si on a un dataset inéquilibré comme le nôtre.

# LightGBM GBDT with KFold or Stratified KFold
# Parameters from Tilii kernel: https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt/code
def kfold_lightgbm(df, num_folds, stratified = False, debug= True):
    # Divide in training/validation and test data
    train_df = df_train
    test_df = df_test
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified: # division of a population into smaller sub-groups known as strata
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=0)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=0)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]
        print("Train:", train_idx, "Valid:", valid_idx)
        print("Train lenght:",len(train_idx), "Valid length:", len(valid_idx))

        # LightGBM parameters found by Bayesian optimization
        params = {
            'n_jobs':-1,
            'n_estimators':500,
            'learning_rate':0.02,
            'num_leaves':34,
            'colsample_bytree':0.9497036,
            'subsample':0.8715623,
            'max_depth':8,
            'reg_alpha':0.041545473,
            'reg_lambda':0.0735294,
            'min_split_gain':0.0222415,
            'min_child_weight':39.3259775,
            'silent':-1,
            'verbose':-1
        }
        clf = LGBMClassifier(**params)

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 50, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
        train_df['PROBA']= oof_preds
        train_df.to_csv('train_df_pred.csv', index=False)
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        
        #Get the auc curbe 
        fpr, tpr, thresholds = roc_curve(valid_y, oof_preds[valid_idx])
        auc = roc_auc_score(valid_y, oof_preds[valid_idx])
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, marker='.')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.title('ROC curve', fontsize = 20)
        plt.xlabel('FPR', fontsize=15)
        plt.ylabel('TPR', fontsize=15)
        plt.grid()
        plt.legend(["AUC=%.3f"%auc])
        plt.show()
        
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))

    # Write submission file and plot feature importance
    if not debug:
        test_df['TARGET'] = sub_preds # predict a probability
        test_df[['SK_ID_CURR', 'TARGET']].to_csv(submission_file_name, index= False)
    display_importances(feature_importance_df)
    
    return feature_importance_df

# Define a timer
@contextmanager
def timer(title):
    t0 = time.time()
    yield # generator of object
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Display/plot feature importance
def display_importances(feature_importance_df_):
    global best_features
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    best_features.to_csv('best_features.csv')
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')

with timer("Run LightGBM with kfold"):
        feat_importance = kfold_lightgbm(df, num_folds= 5)

# # Coût fonction
submission = pd.read_csv('/Users/mohamedds/Desktop/P7/submission_sample.csv')
submission.describe()

# Load data with prediction
train_df = pd.read_csv('train_df_pred.csv')
train_df

train_df.PREDICT.describe()
train_df.isna().mean().sort_values()

# # Determiner le seuil à partir duquel on refuse le prêt
# ## Option 1 : Seuil par défaut 0.5
# Premièrement, le seuil à partir duquel la proba calculée se transforme en classe 1 est par défaut 0.5. Du coup, la matrice de confusion est la suivante:
# Test threshold
def test_threshold(thd):
    train_df['pred']= train_df['PROBA'].apply(lambda x: 0 if x<thd else 1)
    cf_matrix = confusion_matrix(train_df['TARGET'], train_df['pred'])
    sns.heatmap(cf_matrix, annot=True)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    print('Rapport: \n', classification_report(train_df['TARGET'], train_df['pred']))
    return train_df['pred']

#Get the confusion matrix with a default threshold 0.5
test_threshold(0.5)

# Là, on remarque qu'il y a 24000 faux négatifs contre 590 faux négatifs. C'est-à-dire que FN est 40 fois plus grand que FP. Autrement dit, on va accorder un prêt à 24000 clients qui ne peuvent pas rembourser. C'est énorme et ne covient pas à la problémtique du métier. Ce dernier considère qu'un faux négatif est environ 10 fois plus couteux qu'un faux positif.  
# ## Option 2: Seuil à partir de l'observation du data test et ses statistiques
# En revanche, notre hypothèse basée sur la prédiction de data_test et son 3e quartile:
# * à partir de describe(), on constate que 75% des clients ont une probabilité inférieur à 0.098 et comme 91% de clients ont reçu un avis favorable au prêt (classe 0). Donc, on peut dire que le seuil de score pour ne pas accorder le prêt est autour de 0.1
# Ce nouvau seuil peut fonctionner mieux que celui par défaut. Le graphe ci-dessous nous montre les nouvaux scores.

# Thresholde 0.09
test_threshold(0.09)

# Ici, on constate 8500 faux négatifs contre 69000 faux positifs. C'est-à-dire que le FP est environ 8 fois plus que le FN. Donc, cette méthode donne des résultats plus aventageux que celle par défaut(0.5).
# # Option 3 : Comparaison de faux négatifs et de faux positifs pour dégager un seuil

def find_threshold(y_true, y_pred_proba):
    values_FP =[]
    values_FN = []
    values_i = []
    for i in np.arange(0,0.5, 0.01):
        values_i.append(i)
        y_pred = y_pred_proba.apply(lambda x: 0 if x<i else 1)
        CM = confusion_matrix(y_true, y_pred)
        FN = CM[1][0] # False Negative
        FP = CM[0][1] # Salse Positive
        values_i.append(i)
        values_FP.append(FP)
        values_FN.append(FN)
        FN_dic = {}
        FP_dic = {}
    for key, value in zip(values_i, values_FN):
        FN_dic[key] = value
    for key, value in zip(values_i, values_FP):
        FP_dic[key] = value
    df_FN = pd.DataFrame(list(FN_dic.items()), columns=['threshold', 'nb_FN'])
    df_FP = pd.DataFrame(list(FP_dic.items()), columns=['threshold', 'nb_FP'])
    df_FP_FN = pd.concat([df_FN, df_FP])
    plt.figure(figsize=(12, 12))
    lines = df_FP_FN.plot.line()
#     #plt.subplot(2, 2, 1)    
#     sns.scatterplot(data = df_FN, x= 'threshold', y = 'nb_FN')
#     #plt.subplot(2, 2, 2)
#     sns.scatterplot(data = df_FP, x= 'threshold', y = 'nb_FP')
    plt.show()
    

find_threshold(train_df['TARGET'], train_df['PROBA'])

# A partir de garphique ci-dessus, on remarque que le seuil qui fait un équilibre entre FN et FP est autour de 0.1

# # Option 4: Un appui sur le context du métier
# Minimiser les faux négatifs qui valent 10 fois plus un faux positif. Donc, on s'appuie sur fbeta_score avec un beta =10 c'est à dire minimiser 10 fois les faux négatifs.

def bank_score(y_true, y_pred, fn_weight=-10, fp_weight=-1, tp_weight=1, tn_weight=1):

    # Matrice de Confusion
    mat_conf = confusion_matrix(y_true,y_pred)
    
    tn = mat_conf[0, 0]
    fn = mat_conf[1, 0]
    fp = mat_conf[0, 1]
    tp = mat_conf[1, 1]
    
    # Calculate of gains
    Gain_tot = tp*tp_weight + tn*tn_weight + fp*fp_weight + fn*fn_weight
    G_max = (fp + tn)*tn_weight + (fn + tp)*tp_weight
    G_min = (fp + tn)*fp_weight + (fn + tp)*fn_weight
    
    G_normalized = (Gain_tot - G_min)/(G_max - G_min)
    
    return G_normalized  

score_gain=[]
for i in np.arange(0,0.5, 0.01):
    score = bank_score(train_df['TARGET'], test_threshold(i), fn_weight=-10, fp_weight=-1, tp_weight=1, tn_weight=1)
    score_gain.append(score)
    print("threshold", i)
    print("Gain", score)
print("Best gain", max(score_gain))

# Donc, le meilleur gain s'identifie au sueil 0.15. Veuillons la matrice de confusion correspondant.

bank_score(train_df['TARGET'], test_threshold(0.15), fn_weight=-10, fp_weight=-1, tp_weight=1, tn_weight=1)

# # Optimisation du modèle
# Pour aller plus vite en terme de temps de calcul, je m'appuie sur l'importance des variables.

best_features_mean = best_features.groupby('feature').agg({'importance': 'mean'})
best_features_mean = best_features_mean.reset_index()
best_features_mean.sort_values(by = 'importance', ascending=False)
best_features_mean.to_csv('best_features.csv', index=False)

best_features = pd.read_csv('best_features.csv')
best_features.sort_values(by = 'importance', ascending=False)

features = best_features[best_features['importance']>170]
features = list(features['feature'].unique())
features.append('SK_ID_CURR')
features

# ## Choix des variables 
# Comme 20 variables, c'est déjà beaucoup, alors il faut faire des choix en s'appuyant sur le contexte du métier. Autrement dit, je me concentrer sur des variables de qualité et compréhensibles. Par exemple, EXT_SOURCE_2 et 3 représentent des scores donnés par des organismes spécialisés dans le scoring de crédit et comme le 1er score est plus important que le 2nd je me limite au premier. L'âge (Years_BIRTH), l'expérience dans le job actuel (YEARS_EMPLOYED), le retard des remboursements (INSTAL_DPD_MEAN), le montant du crédit (AMT_CREDIT), les échances du paiement (AMT_ANNUITY), le pourcentage de revenu par rapport au credit (ANNUITY_INCOME_PERC), la sum des paiements (INSTAL_AMT_PAYMENT_SUM) et l'identifiant des clients (SK_ID_CURR). En tout, on a 9 variables intéressantes.

final_features = ['SK_ID_CURR','YEARS_BIRTH','YEARS_EMPLOYED','ANNUITY_INCOME_PERC',
                  'EXT_SOURCE_2', 'AMT_CREDIT', 'AMT_ANNUITY','INSTAL_DPD_MEAN',
                 'INSTAL_AMT_PAYMENT_SUM']

data_train_selected_features = df_train[final_features]
data_train_selected_features[['SK_ID_CURR','YEARS_BIRTH', 'YEARS_EMPLOYED']] = data_train_selected_features[['SK_ID_CURR','YEARS_BIRTH', 'YEARS_EMPLOYED']].astype(int)

data_train_selected_features.to_csv('app_streamlit/X_train_features.csv', index=False)
data_train_selected_features

X_train, X_valid, y_train, y_valid = train_test_split(X[final_features], y, random_state=0, train_size=0.7)
print("dimensions X_train:",X_train.shape)
print("dimensions X_valid:",X_valid.shape)

df_train[final_features]

# # Le score du modele LGBMClassifier avant l'intégration de la fonction coût 
## Initialization of lgbm and make score
params = {
            'n_jobs':-1,
            'learning_rate':0.02,
            'num_leaves':34,
            'colsample_bytree':0.9497036,
            'subsample':0.8715623,
            'max_depth':8,
            'reg_alpha':0.041545473,
            'reg_lambda':0.0735294,
            'min_split_gain':0.0222415,
            'min_child_weight':39.3259775,
            'silent':-1,
            'verbose':-1
        }


#score = 'precision_recall_fscore_support'
lgbm = LGBMClassifier(random_state=0, n_estimators=3000, **params)
lgbm.fit(X_train, y_train)
print(lgbm.score(X_valid, y_valid))

# On voit bien que le modèle a un très bon score environ 92%. La prise en compte de la fonction coût va vertiablement changer la donne.
# # Prise en compte de la fonction coût dans le modèle avec optimisation

def model_optimisation(model, params, scoring, X, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,scoring=scoring, cv=cv)
    grid_search.fit(X, y)
    return grid_search, grid_search.best_score_,grid_search.best_params_,grid_search.best_estimator_

# make a scoring
custom_score = make_scorer(bank_score, greater_is_better=True)

# Simple parameters model
param_grid = {
    
        'max_depth': (6, 8)    
}
lgbm_optimize = model_optimisation(lgbm, param_grid, custom_score, X_train, y_train)
lgbm_optimize

# Donc, le score tombe à 0.67.
# # Evaluation du modèle

train_df['y_proba'] = lgbm_optimize[0].predict_proba(X[final_features])[:,1]
train_df

train_df['y_pred'] = train_df['y_proba'].apply(lambda x: 0 if x<0.15 else 1) # business threshold 

fpr, tpr, thresholds = roc_curve(train_df['TARGET'], train_df['y_pred'])
auc = roc_auc_score(train_df['TARGET'], train_df['y_pred'])
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, marker='.')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title('ROC curve', fontsize = 20)
plt.xlabel('FPR', fontsize=15)
plt.ylabel('TPR', fontsize=15)
plt.grid()
plt.legend(["AUC=%.3f"%auc])
plt.show()

cf_matrix = confusion_matrix(train_df['TARGET'], train_df['y_pred'])
sns.heatmap(cf_matrix, annot=True)
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
print('Rapport: \n', classification_report(train_df['TARGET'], train_df['y_pred']))

# # Exportation du modèle et des variables au dashboard

joblib.dump(lgbm, 'FASTAPI/model_loans.joblib')
joblib.dump(lgbm, 'app_streamlit/model_loans.joblib')
joblib.dump(final_features, 'FASTAPI/features.joblib')
joblib.dump(final_features, 'app_streamlit/features.joblib')