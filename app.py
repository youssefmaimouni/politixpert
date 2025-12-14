from flask import Flask, render_template, request, jsonify
from rag_engine import PolitixpertRAG
import os

app = Flask(__name__)

# Chemin vers votre fichier CSV
CSV_PATH = "politixpert_data_cleaned.csv"

# Variable globale pour stocker le moteur
rag_engine = None

def initialize_engine():
    """Fonction pour charger le modèle (sans décorateur Flask obsolète)"""
    global rag_engine
    if rag_engine is None:
        if os.path.exists(CSV_PATH):
            rag_engine = PolitixpertRAG(CSV_PATH)
        else:
            print(f"❌ Erreur: Le fichier {CSV_PATH} est introuvable !")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/ask', methods=['POST'])
def ask():
    global rag_engine
    
    # Sécurité : Si le moteur n'est pas chargé, on le charge maintenant
    if rag_engine is None:
        initialize_engine()

    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "La question est vide"}), 400

    print(f"📩 Nouvelle question reçue: {question}")
    
    # Appel au moteur IA
    answers = rag_engine.generate_answer(question)
    
    return jsonify({"results": answers})

if __name__ == '__main__':
    # C'est ici qu'on force le chargement au démarrage
    print("🚀 Démarrage du serveur...")
    initialize_engine()
    
    # Lancement du serveur Web
    app.run(host='0.0.0.0', port=5000, debug=False)