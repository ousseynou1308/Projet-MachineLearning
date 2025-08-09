# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 19:00:33 2025

@author: nouss
"""

# Import des librairies
import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Détection Faux Billets", page_icon="💵", layout="centered")

st.markdown(
    "<h1 style='text-align: center; color: #00FFAA; font-size: 40px;'>Détection de faux billets 💵</h1>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type="csv", label_visibility="collapsed")
if uploaded_file is not None:
    st.success("✅ Fichier chargé avec succès : " + uploaded_file.name)
    df = pd.read_csv(uploaded_file, sep=";")
    st.write("## Aperçu du fichier :", df.head())

    # Vérification des colonnes attendues
    colonnes_attendues = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
    if not all(col in df.columns for col in colonnes_attendues):
        st.error("⚠️ Le fichier ne contient pas les colonnes attendues.")
        st.stop()

    # --- Style CSS pour le bouton ---
    st.markdown("""
        <style>
        div.stButton > button {
            background-color: #00FFAA;
            color: black;
            border: none;
            padding: 0.75em 1.5em;
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #00DD99;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

    # Vérification des valeurs manquantes
    if df.isnull().values.any():
        st.error("❌ Le fichier contient des valeurs manquantes. Veuillez le corriger avant de lancer la prédiction.")
    else:
        if st.button("Lancer la prédiction"):
            response = requests.post(
                "https://fastapi-production-detectionfraude.up.railway.app/predict/",
                files={"file": uploaded_file.getvalue()}
            )

            if response.status_code == 200:
                predictions = response.json()["predictions"]
                df["Prediction"] = predictions
                st.write("## Résultats des prédictions :", df)

                # Statistiques
                st.write("## Statistiques :")
                st.write(df["Prediction"].value_counts())
                
                st.write(
                    f"**Interprétation :**\n"
                    f"- **1** → Vrais billets\n"
                    f"- **0** → Faux billets"
                )

                # Graphique circulaire
                st.write("## Répartitions des prédictions :")
                fig = px.pie(
                    df,
                    names="Prediction",
                    color_discrete_sequence=px.colors.sequential.Aggrnyl
                )
                st.plotly_chart(fig)
                
                st.write("## Visualisation : Vrais vs Faux billets")
                
                
                fig = px.scatter(
                    df,
                    x="margin_up",
                    y="margin_low",
                    color="Prediction",
                    symbol="Prediction",
                    color_discrete_map={0: "green", 1: "red"},
                    labels={"Prediction": "Type de billet"},
                    title="Répartition des vrais et faux billets"
                )
                
                fig.update_traces(marker=dict(size=10, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig, use_container_width=True)
               
                                 

                # Téléchargement des résultats
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les résultats",
                    data=csv_data,
                    file_name="resultats_predictions.csv",
                    mime="text/csv"
                )

            else:
                st.error("Erreur lors de la requête à l'API.")



