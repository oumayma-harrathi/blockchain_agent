# app.py
import streamlit as st
from blockchain_agent import BlockchainAgent
from config import Config

# Charger la config
config = Config.load()

st.set_page_config(page_title="🎨 Blockchain Art Agent", layout="wide")
st.title("🎨 Blockchain Art & NFT Advisor")

# Initialiser l'agent dans la session
if "agent" not in st.session_state:
    with st.spinner("Chargement de l'agent... (LLM, RAG, NER)"):
        st.session_state.agent = BlockchainAgent(model_name=config.MODEL_NAME)

agent = st.session_state.agent

# Entrée utilisateur
prompt = st.text_input("🔍 Ta question (ex: galerie NFT, royalties, multi-devise...) :")

if prompt:
    st.write("### 💬 Réponse en temps réel :")
    response_placeholder = st.empty()
    full_response = ""

    # Utilise le streaming
    for chunk in agent.get_response_stream(prompt):
        full_response += chunk
        response_placeholder.markdown(full_response + "⏳")  # Effet "en cours"
    
    # Enlever le loader
    response_placeholder.markdown(full_response)

    # Optionnel : afficher l'historique
    with st.expander("📜 Voir l'historique des échanges"):
        for entry in agent.conversation_history[-3:]:
            st.markdown(f"**Q:** {entry['query']}")
            st.markdown(f"**R:** {entry['response'][:300]}...")