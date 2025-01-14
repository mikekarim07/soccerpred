import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Preprocesamiento de datos
def preprocess_data(data):
    # Filtrar columnas relevantes
    data = data[['HTShortName', 'ATShortName', 'Winner', 'FT_HTGoals', 'FT_ATGoals', 'matchday']]
    
    # Crear variable de resultado
    data['Result'] = data['Winner'].map({'HOME_TEAM': 1, 'DRAW': 0, 'AWAY_TEAM': -1})
    
    # Manejar valores nulos
    data = data.dropna()

    return data

# Entrenar el modelo
def train_model(data):
    # Crear características basadas en equipos
    data = pd.get_dummies(data, columns=['HTShortName', 'ATShortName'], drop_first=True)
    
    # Seleccionar características y etiquetas
    X = data.drop(['Winner', 'Result'], axis=1)
    y = data['Result']

    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo de Random Forest
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, accuracy, report, X

# Predecir resultado
def predict_match(model, local_team, away_team, data_columns):
    # Crear una fila con ceros para las columnas
    input_data = pd.DataFrame(0, index=[0], columns=data_columns)

    # Activar las columnas correspondientes a los equipos seleccionados
    input_data[f'HTShortName_{local_team}'] = 1
    input_data[f'ATShortName_{away_team}'] = 1

    # Predecir resultado
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    return prediction, probabilities

# Cargar archivo de usuario
def main():
    st.title("Predicción de Resultados de Fútbol")

    # Subir archivo
    uploaded_file = st.file_uploader("Sube un archivo de Excel con la información de los partidos", type=["xlsx"])

    if uploaded_file:
        data = pd.read_excel(uploaded_file, sheet_name='Data')
        data = preprocess_data(data)

        st.write("Vista previa de los datos:")
        st.dataframe(data.head())

        # Entrenar modelo
        st.write("Entrenando el modelo...")
        model, accuracy, report, X = train_model(data)

        st.write(f"Precisión del modelo: {accuracy * 100:.2f}%")

        # Obtener nombres únicos de equipos
        local_teams = data['HTShortName'].unique()
        away_teams = data['ATShortName'].unique()

        # Entrada para predicción
        st.write("### Predicción de un partido")
        local_team = st.selectbox("Selecciona el equipo local", options=local_teams)
        away_team = st.selectbox("Selecciona el equipo visitante", options=away_teams)

        if st.button("Predecir Resultado"):
            result, probabilities = predict_match(model, local_team, away_team, X.columns)
            result_text = "Local" if result == 1 else "Empate" if result == 0 else "Visitante"
            st.write(f"Resultado predicho: {result_text}")
            st.write(f"Probabilidades: Local {probabilities[1]:.2%}, Empate {probabilities[0]:.2%}, Visitante {probabilities[2]:.2%}")

if __name__ == "__main__":
    main()
