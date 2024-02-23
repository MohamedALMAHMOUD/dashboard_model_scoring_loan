# Import necessary packages
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import requests
from PIL import Image
import streamlit.components.v1 as components
from lime import lime_tabular
import joblib
import shap
shap.initjs() # JavaScript plots
import matplotlib.pyplot as pl
import time

# Load data
df = pd.read_csv('X_train_features.csv')

# Import the model. 
lgbm = joblib.load(open("model_loans.joblib","rb"))
features = joblib.load(open("features.joblib", "rb"))

# Functions
# Function to display a shap plot
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

# Function for predicting if we accord or no the credit
def get_pred_per_client(df, data):  
    # Calling api
    response = requests.post("https://python.cabane-data.fr/predict", json=data)
    if response.status_code == 200:
        # Charge JSON response
        response_json = response.json()

        # Extraction of "proba" field valeur 
        proba_value = response_json.get("proba",0)

        # Verification of conversion probability in float
        try:
            proba_float = float(proba_value)
            st.success(proba_float)

            if proba_float < 0.15:
                st.write("On accorde le prêt")
            else:
                st.write("On n'accorde pas le prêt")

        except ValueError:
            st.error("Erreur de conversion en float")
    else:
        st.error(f"Erreur de requête : {response.status_code}")

# Function explanation
def get_interpret(df, data):
    # Explain/Interpretability
    interpretor = lime_tabular.LimeTabularExplainer(
        training_data=np.array(df),
        feature_names=df.columns.values.tolist(),
        mode='classification')
    data_conv =pd.DataFrame.from_dict([data])
    data_array = data_conv.values.ravel()
    exp = interpretor.explain_instance(data_row=np.array(data_array),
                                    predict_fn=lgbm.predict_proba,
                                    num_samples=1000,
                                    num_features=6)
    components.html(exp.as_html(), height=700)
# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="Le menu principal",
        options=["Prédiction des clients", "Prédiction des nouvaux clients", 'Comparaison'],
        icons=["house", "person", "people fill"],
        menu_icon="cast",
        default_index=0)
    image = Image.open(r'logo_entreprise.png')

    st.image(image, caption='Prêt à depenser')

# Home page
if selected == "Prédiction des clients":

    html_temp ="""
       <div style="background-color:tomato;padding:10px">
       <h2 style="color:white;text-align:center;">Prêt à dépenser </h2>
       </div>
        """
    st.markdown(html_temp, unsafe_allow_html=True)
    #filtering by client
    client_id = st.selectbox("Choisir un client", df['SK_ID_CURR'].unique())
    df_client = df[df['SK_ID_CURR'] == client_id]
    st.write(df_client)
    st.success(f"L'identifient du client est: {client_id}")
    df_dico = dict(df_client.iloc[-1, :])
    data = {
        'SK_ID_CURR': df_dico['SK_ID_CURR'],
        'YEARS_BIRTH': df_dico['YEARS_BIRTH'],
        'YEARS_EMPLOYED': df_dico['YEARS_EMPLOYED'],
        'ANNUITY_INCOME_PERC': df_dico['ANNUITY_INCOME_PERC'],
        'EXT_SOURCE_2': df_dico['EXT_SOURCE_2'],
        'AMT_CREDIT': df_dico['AMT_CREDIT'],
        'AMT_ANNUITY': df_dico['AMT_ANNUITY'],
        'INSTAL_DPD_MEAN': df_dico['INSTAL_DPD_MEAN'],
        'INSTAL_AMT_PAYMENT_SUM': df_dico['INSTAL_AMT_PAYMENT_SUM']
    }
    # Predict button
    if st.button("Predict"):
        get_pred_per_client(df_dico, data)
    
    # Explanation button
    if st.button("Interpréter"):
        get_interpret(df, data)

# New client page
if selected == "Prédiction des nouvaux clients":
    st.subheader("Renseignez les champs ci-dessous")
    SK_ID_CURR = st.number_input("Identifient du client", min_value=100001, value=100001, step=1)
    YEARS_BIRTH = st.number_input("L'âge du client",
                                    min_value=20, value=35, step=1)
    YEARS_EMPLOYED = st.number_input("Années d'expériences dans le poste actuel",
                                        min_value=0, value=15, step=1)
    ANNUITY_INCOME_PERC = st.number_input("Le pourcentage des revenus par rapport au montant du crédit",
                                    min_value=0, value=0, step=1)
    EXT_SOURCE_2 = st.slider("Le 2nd score exterieur",
                                min_value=0.014, max_value=0.9, value=0.5, step=0.001)
    AMT_CREDIT = st.number_input("Le montant du prêt demandé",
                                    min_value=0.022, value=0.04, step=0.001)
    AMT_ANNUITY = st.number_input("Les échances du remboursement du prêt",
                                    min_value=1615, value=25000, step=100)
    
    INSTAL_DPD_MEAN = st.number_input("La durée moyenne du retard du prêt",
                                        min_value=0, value=0, step=1)
    INSTAL_AMT_PAYMENT_SUM = st.number_input("La somme des paiements du remboursement des prêts",
                                                min_value=0, value=0, step=1)
    data_new_client = {
            'SK_ID_CURR': SK_ID_CURR,
            'YEARS_BIRTH': YEARS_BIRTH,
            'YEARS_EMPLOYED': YEARS_EMPLOYED,
            'ANNUITY_INCOME_PERC': ANNUITY_INCOME_PERC,
            'EXT_SOURCE_2': EXT_SOURCE_2,
            'AMT_CREDIT': AMT_CREDIT,
            'AMT_ANNUITY': AMT_ANNUITY,
            'INSTAL_DPD_MEAN': INSTAL_DPD_MEAN,
            'INSTAL_AMT_PAYMENT_SUM': INSTAL_AMT_PAYMENT_SUM,            
        }
    if st.button("Predict"):
        get_pred_per_client(df, data_new_client)
    
# Comparaison page
if selected == "Comparaison":
    st.title(f"Quelques comparaisons")
    st.text("Nous cherchons les resultas pour vous")
    
    # Progress bar
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.2)
        my_bar.progress(percent_complete + 1)
    
    st.subheader("Clients similaires")
    # Display stacked Shapley values along clustered observations
    explainer = shap.TreeExplainer(lgbm)
    shap_values = explainer.shap_values(df.sample(1000))
    st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0], features), 400)
    st.subheader("Explication générale")
    
    st.set_option('deprecation.showPyplotGlobalUse', False)
    #summary_plot
    explain = shap.Explainer(lgbm, df.sample(1000))
    shap_values_beeswarm = explain(df.sample(1000), check_additivity=False )
    shap.plots.beeswarm(shap_values_beeswarm)
    st.pyplot()
    st.balloons()