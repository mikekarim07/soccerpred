import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Preprocesamiento de datos
def preprocess_data(data):
    # Filtrar columnas relevantes
    data = data[['Home Team Short Name', 'Away Team Short Name', 'Winner', 
                 'Full Time Home Team Goals', 'Full Time Away Team Goals',
                 'Home Team Total Points', 'Home Team Home Points','Home Team Matches Played','Home Team Home Games Played','Home Team Home Games Win','Home Team Home Games Draw',
                 'Home Team Home Games Lost','Home Team Home Goals Scored','Home Team Home Goals Received','Home Team Home Points','matchday',
                 'Away Team Total Points', 'Away Team Away Points','Away Team Matches Played','Away Team Away Games Played','Away Team Away Games Win','Away Team Away Games Draw',
                 'Away Team Away Games Lost','Away Team Away Goals Scored','Home Team Home Goals Received','Home Team Home Points']]
    
    # Crear variable de resultado
    data['Result'] = data['Winner'].map({'HOME_TEAM': 1, 'DRAW': 0, 'AWAY_TEAM': -1})
    
    # Manejar valores nulos
    data = data.dropna()

    return data

# Entrenar el modelo
def train_model(data):
    # Crear características basadas en equipos
    data = pd.get_dummies(data, columns=['Home Team Short Name', 'Away Team Short Name'], drop_first=True)
    
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
    input_data[f'Home Team Short Name_{local_team}'] = 1
    input_data[f'Away Team Short Name_{away_team}'] = 1

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
            
        #lineas codigo nuevas
        data = data[data['status']=='FINISHED']
        league = data['LeagueName'].unique()
        st.write("### Elige una liga para realizar el entrenamiento del modelo")
        league_name = st.selectbox("Selecciona la liga", options=league)
        st.dataframe(data)
        #final lineas codigo nuevas
        data = preprocess_data(data)

        st.write("Vista previa de los datos:")
        #st.write(type(data))  # Verifica que sea un DataFrame
        #st.write(data.shape)  # Revisa cuántas filas y columnas tiene

        #st.dataframe(data.head())
        st.dataframe(data)

        # Entrenar modelo
        st.write("Entrenando el modelo...")
        model, accuracy, report, X = train_model(data)

        st.write(f"Precisión del modelo: {accuracy * 100:.2f}%")

        # Obtener nombres únicos de equipos
        local_teams = data['Home Team Short Name'].unique()
        away_teams = data['Away Team Short Name'].unique()

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
