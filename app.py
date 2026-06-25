"""
app.py — Chatbot con RAG (interfaz de chat en el navegador).

Para cada pregunta:
  1. Busca en tu documento los fragmentos más relevantes (ChromaDB).
  2. Se los pasa como CONTEXTO a un modelo de lenguaje (Groq).
  3. El modelo redacta la respuesta priorizando tu documento.

Uso:  streamlit run app.py
"""

import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# --- Configuración ---
MODELO = "llama-3.3-70b-versatile"   # podés probar "llama-3.1-8b-instant" (más rápido)
K_FRAGMENTOS = 4                      # cuántos fragmentos recuperar por pregunta

SYSTEM_PROMPT = """Sos un asistente que ayuda a resolver un problema concreto.
Respondé SIEMPRE en español, de forma clara, ordenada y concreta.

Reglas:
- Usá PRIORITARIAMENTE la información del CONTEXTO que te paso en cada mensaje.
- Si el contexto no alcanza, podés complementar con tu conocimiento general,
  pero aclarás cuándo lo hacés (ej: "Según mi conocimiento general...").
- Si no sabés algo y el contexto no lo cubre, decilo honestamente en vez de
  inventar.
"""


@st.cache_resource
def cargar_recursos():
    """Inicializa Groq y la colección de ChromaDB una sola vez."""
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chroma = chromadb.PersistentClient(path="./chroma_db")
    coleccion = chroma.get_collection(name="documento")
    return groq, coleccion


def recuperar(coleccion, pregunta: str, k: int) -> list[str]:
    """Devuelve los k fragmentos más parecidos a la pregunta."""
    res = coleccion.query(query_texts=[pregunta], n_results=k)
    return res["documents"][0]


# --- Interfaz ---
st.set_page_config(page_title="Asistente", page_icon="🤖")
st.title("🤖 Asistente del proyecto")
st.caption("Respuestas basadas en tu documento + conocimiento general del modelo.")

groq, coleccion = cargar_recursos()

if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Re-dibuja el historial
for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if pregunta := st.chat_input("Escribí tu pregunta..."):
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # 1) Recuperar contexto de tu documento
    fragmentos = recuperar(coleccion, pregunta, K_FRAGMENTOS)
    contexto = "\n\n---\n\n".join(fragmentos)

    # 2) Armar el mensaje para el modelo
    mensaje_usuario = (
        f"CONTEXTO (extraído del documento):\n{contexto}\n\n"
        f"PREGUNTA DEL USUARIO:\n{pregunta}"
    )

    # 3) Generar respuesta en streaming
    with st.chat_message("assistant"):
        stream = groq.chat.completions.create(
            model=MODELO,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": mensaje_usuario},
            ],
            stream=True,
        )
        respuesta = st.write_stream(
            (chunk.choices[0].delta.content or "" for chunk in stream)
        )

    st.session_state.mensajes.append({"role": "assistant", "content": respuesta})

    # (Opcional) mostrar qué fragmentos se usaron, para tu defensa/demostración
    with st.expander("Ver fragmentos del documento usados"):
        for i, frag in enumerate(fragmentos, 1):
            st.markdown(f"**Fragmento {i}:** {frag}")
