# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:58:01 2025

@author: nouss
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
import pandas as pd
import joblib
import io

app = FastAPI()

# ================================================================== #
#                Appel du modéle predictif                           #
# ================================================================== #

model_path = "model_RegressionLogistic.sav"
scaler_path= "standard_scaler.sav"

model= joblib.load(model_path)
scaler= joblib.load(scaler_path)



# ================================================================== #
#                             UNSEEN DATA                            #
# ================================================================== #


# Colonnes attendues dans l'ordre exact (selon entraînement du modèle)
colonnes_attendues = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

@app.get("/")
def read_root():
    return {"Hello, votre application Détection de fraude fonctionne correctement"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")), sep=';')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur de lecture du fichier : {str(e)}")
    
    # Vérification des colonnes attendues
    if list(df.columns) != colonnes_attendues:
        raise HTTPException(
            status_code=400,
            detail=f"Les colonnes du fichier ne correspondent pas exactement à celles attendues : {colonnes_attendues}"
    )

    # Vérification des valeurs manquantes
    if df.isnull().any().any():
        raise HTTPException(
            status_code=400,
            detail="Le fichier contient des valeurs manquantes. Merci de fournir un fichier complet sans NaN."
        )

    # Préparation des features (exclure la cible)
    X_scaled = scaler.transform(df)
    predictions = model.predict(X_scaled)

    return {"predictions": predictions.tolist()}
