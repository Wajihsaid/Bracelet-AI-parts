import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import joblib
from flask import Flask, jsonify
from flask_cors import CORS
from sklearn.preprocessing import StandardScaler

# Chemins des modèles
current_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.join(current_dir, "model_output", "random_forest_model.pkl")
SCALER_PATH = os.path.join(current_dir, "model_output", "scaler.pkl")
ENCODER_PATH = os.path.join(current_dir, "model_output", "label_encoder.pkl")

# Initialisation de l'application Dash
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
CORS(server)

class PatientMonitor:
    def __init__(self):
        # Charger les modèles
        self.rf_model = joblib.load(MODEL_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        self.encoder = joblib.load(ENCODER_PATH)

        # Configuration initiale
        self.time_step = 1
        self.history_length = 120
        self.prediction_window = 30
        self.data_buffer = []
        
        # Initialisation des données
        self.data = {
            'time': [int(datetime.now().timestamp())],
            'hr': [75.0],
            'spo2': [98.0],
            'temp': [37.0],
            'predictions': [0.0],
            'status': 'normal',
            'last_prediction_time': None,
            'display_status': 'normal'
        }

    def generate_vital_signs(self):
        """Générer des signes vitaux avec une légère variation"""
        hr = float(np.clip(self.data['hr'][-1] + np.random.normal(0, 1.5), 55, 100))
        spo2 = float(np.clip(self.data['spo2'][-1] + np.random.normal(0, 0.2), 95, 100))
        temp = float(np.clip(self.data['temp'][-1] + np.random.normal(0, 0.05), 36, 38.5))
        return hr, spo2, temp

    def predict_patient_status(self, hr, spo2, temp):
        """
        Prédire le statut du patient avec le modèle Random Forest
        """
        # Préparer les données
        features = np.array([[hr, spo2, temp]])
        
        # Mise à l'échelle 
        features_scaled = self.scaler.transform(features)
        
        # Prédiction
        prediction = self.rf_model.predict(features_scaled)
        proba = self.rf_model.predict_proba(features_scaled)
        
        # Convertir la prédiction
        status = self.encoder.inverse_transform(prediction)[0]
        proba_classe = proba[0][prediction[0]]
        
        return status, proba_classe

    def update_data(self):
        """Mettre à jour les données du patient"""
        hr, spo2, temp = self.generate_vital_signs()
        new_time = self.data['time'][-1] + self.time_step

        self.data['time'].append(new_time)
        self.data['hr'].append(hr)
        self.data['spo2'].append(spo2)
        self.data['temp'].append(temp)

        self.data_buffer.append({'time': new_time, 'hr': hr, 'spo2': spo2, 'temp': temp})

        # Vérifier si on peut faire une prédiction
        new_prediction = False
        if len(self.data_buffer) >= self.prediction_window:
            status, pred = self.predict_patient_status(hr, spo2, temp)
            
            # Convertir le statut en catégorie de risque
            risk_status = 'critical' if status == 'Anormal' else 'normal'
            
            self.data['status'] = risk_status
            self.data['display_status'] = risk_status
            self.data['last_prediction_time'] = new_time
            self.data['predictions'].append(pred)
            self.data_buffer = []
            new_prediction = True
        else:
            pred = self.data['predictions'][-1]

        # Tronquer l'historique
        for k in ['time', 'hr', 'spo2', 'temp', 'predictions']:
            self.data[k] = self.data[k][-self.history_length:]

        return {**self.data, 'new_prediction': new_prediction}

# Initialiser le moniteur
monitor = PatientMonitor()

# Route API
@server.route('/api/patient-data')
def patient_data():
    return jsonify({
        'status': monitor.data['status'],
        'display_status': monitor.data['display_status']
    })

# Layout de l'application
app.layout = html.Div([
    html.Div([
        html.Div(id='status-banner', style={'padding': '15px', 'textAlign': 'center', 'marginBottom': '20px'})
    ]),
    dcc.Store(id='data-store'),
    dcc.Interval(id='update-interval', interval=1000),

    html.Div([
        dcc.Graph(id='hr-graph', style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='spo2-graph', style={'display': 'inline-block', 'width': '32%'}),
        dcc.Graph(id='temp-graph', style={'display': 'inline-block', 'width': '32%'}),
    ]),

    html.Div([
        html.H3("📊 Valeurs actuelles", style={'textAlign': 'center'}),
        html.Div(id='current-values', style={
            'display': 'grid', 'gridTemplateColumns': 'repeat(2, 1fr)', 'gap': '5px', 'padding': '5px'
        })
    ])
])

# Callback pour mettre à jour le store de données
@app.callback(Output('data-store', 'data'), Input('update-interval', 'n_intervals'))
def update_data_store(n):
    return monitor.update_data()

# Fonction pour créer des graphiques à barres
def make_bar_graph(time_data, value_data, title, color, y_range=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[datetime.fromtimestamp(t).strftime('%H:%M:%S') for t in time_data],
        y=value_data,
        marker_color=color,
        width=0.8
    ))
    fig.update_layout(title=title, yaxis_title=title.split('(')[0].strip(),
                      yaxis_range=y_range, margin=dict(l=20, r=20, t=40, b=20),
                      height=300, xaxis_tickangle=-45)
    return fig

# Fonction pour créer des cartes de valeur
def value_card(label, value, status):
    color_map = {
        'success': ('#d4edda', '#155724'),
        'danger': ('#f8d7da', '#721c24'),
        'warning': ('#fff3cd', '#856404'),
        'collecting': ('#e2e3e5', '#383d41')
    }
    bg, text = color_map[status]
    return html.Div([
        html.Div(label, style={'fontWeight': 'bold'}),
        html.Div(value, style={'fontSize': '18px'})
    ], style={
        'backgroundColor': bg,
        'color': text,
        'padding': '5px',
        'borderRadius': '2px',
        'textAlign': 'center'
    })

# Callback principal pour mettre à jour le tableau de bord
@app.callback(
    [Output('hr-graph', 'figure'),
     Output('spo2-graph', 'figure'),
     Output('temp-graph', 'figure'),
     Output('status-banner', 'children'),
     Output('status-banner', 'style'),
     Output('current-values', 'children')],
    [Input('data-store', 'data')]
)
def update_dashboard(data):
    if not data:
        return go.Figure(), go.Figure(), go.Figure(), "Chargement...", {}, []

    times = data['time']
    display_count = min(15, len(times))
    start_idx = -display_count

    # Créer les graphiques
    hr_fig = make_bar_graph(times[start_idx:], data['hr'][start_idx:], "Fréquence Cardiaque (bpm)", '#3498db', [30, 180])
    spo2_fig = make_bar_graph(times[start_idx:], data['spo2'][start_idx:], "Saturation Oxygène (%)", '#3498db', [85, 100])
    temp_fig = make_bar_graph(times[start_idx:], data['temp'][start_idx:], "Température (°C)", '#3498db', [35, 39])

    # Configuration commune des graphiques
    common_layout = {
        'xaxis': {'type': 'category', 'autorange': False, 'range': [0, display_count-1],
                  'tickmode': 'array', 'tickvals': list(range(display_count)),
                  'ticktext': [datetime.fromtimestamp(t).strftime('%H:%M:%S') for t in times[start_idx:]],
                  'fixedrange': True},
        'yaxis': {'fixedrange': True}
    }

    hr_fig.update_layout(**common_layout)
    spo2_fig.update_layout(**common_layout)
    temp_fig.update_layout(**common_layout)

    # Gestion du statut
    if len(data['time']) < 30:
        status_text = "⏳ Collecte des données... (30s)"
        status_style = {'backgroundColor': '#6c757d', 'color': 'white'}
    else:
        status_map = {
            'critical': ("❗ CRITIQUE - Intervention requise", '#dc3545'),
            'warning': ("⚠️ Alerte - Surveillance accrue", '#fd7e14'),
            'normal': ("✅ Normal - Paramètres stables", '#28a745')
        }
        display_status = data.get('display_status', 'normal')
        status_text, bg_color = status_map.get(display_status, ("❓ Inconnu", '#6c757d'))
        status_style = {'backgroundColor': bg_color, 'color': 'white'}

    # Valeurs les plus récentes
    latest_hr = data['hr'][-1]
    latest_spo2 = data['spo2'][-1]
    latest_temp = data['temp'][-1]
    latest_risk = data['predictions'][-1] if data['predictions'] else 0.0

    # Statuts des différentes mesures
    hr_status = 'danger' if latest_hr < 55 or latest_hr > 100 else 'success'
    spo2_status = 'danger' if latest_spo2 < 95 else 'success'
    temp_status = 'danger' if latest_temp < 36.5 or latest_temp > 38 else 'success'
    risk_status = 'danger' if latest_risk > 0.8 else 'warning' if latest_risk > 0.4 else 'success'

    # Cartes de valeurs
    cards = [
        value_card("❤️ FC", f"{latest_hr:.0f} bpm", hr_status),
        value_card("🩸 SpO2", f"{latest_spo2:.0f} %", spo2_status),
        value_card("🌡️ Temp", f"{latest_temp:.1f} °C", temp_status),
        value_card("⚠️ Risque", f"{latest_risk:.2f}", risk_status)
    ]

    return hr_fig, spo2_fig, temp_fig, status_text, status_style, cards

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8050, debug=True)