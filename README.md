# Modèle de Scoring pour la Prédiction de Prêts
Ce projet consiste en la création et le déploiement d'un modèle de scoring pour prédire si un client est éligible pour un prêt financier. Le projet comprend plusieurs étapes, allant de l'exploration des données à la création d'un tableau de bord interactif en utilisant Streamlit.

## I. Objectif
L'objectif principal de ce projet est de fournir une solution qui permette aux institutions financières de prendre des décisions éclairées concernant l'octroi de prêts. Le modèle de scoring développé utilise des caractéristiques spécifiques des clients pour évaluer leur probabilité de remboursement. Cette approche vise à minimiser les risques associés aux prêts non remboursés tout en offrant des opportunités aux clients qui répondent aux critères définis.

## II. Contenu du Projet
Le projet est divisé en plusieurs étapes clés :

### 1. Collecte et Nettoyage des Données : 
Un ensemble de données pertinentes est collecté, puis les données sont nettoyées et préparées pour l'analyse ultérieure.

### 2. Exploration des Données : 
Une analyse exploratoire des données est effectuée pour mieux comprendre les tendances, les corrélations et les caractéristiques les plus influentes.

### 3. Feature Engineering : 
Des caractéristiques supplémentaires sont créées à partir des données existantes pour améliorer les performances du modèle.

### 4. Construction du Modèle de Machine Learning : 
Différents modèles de machine learning sont entraînés et évalués en utilisant les données d'entraînement. Le modèle le plus performant est sélectionné pour la prédiction finale.

### 5. Sauvegarde du Modèle : 
Le modèle sélectionné est sauvegardé à l'aide de la bibliothèque joblib pour une utilisation ultérieure.

### 6. Tableau de Bord Interactif : 
Un tableau de bord interactif est créé à l'aide de Streamlit. Ce tableau de bord permet aux utilisateurs de visualiser les clients actuels, d'effectuer des recherches et d'obtenir des prédictions pour de nouveaux clients.

## Utilisation du Tableau de Bord :
Le tableau de bord interactif est conçu pour fournir aux utilisateurs une interface conviviale pour :

Consulter la liste des clients actuels dans la base de données.
Effectuer des recherches spécifiques en fonction de différents critères.
Obtenir des prédictions pour de nouveaux clients en entrant leurs informations pertinentes. Vous pouvez faire tout cela [ICI] (https://dashboard-model-scoring-loan-almahmoud.streamlit.app/)
## III. Prérequis
- Python 3.x
- Bibliothèques Python : pandas, scikit-learn, streamlit, joblib, etc.
- Comment Exécuter le Projet
## IV. Conclusion
Ce projet démontre comment créer et déployer un modèle de scoring pour prédire l'éligibilité des clients à un prêt financier. La combinaison de l'analyse de données, de l'apprentissage automatique et de la création d'un tableau de bord interactif offre une solution complète pour aider les institutions financières à prendre des décisions éclairées et à gérer les risques associés aux prêts.

N'hésitez pas à explorer le code source et à utiliser le tableau de bord pour découvrir davantage le fonctionnement du modèle de scoring.