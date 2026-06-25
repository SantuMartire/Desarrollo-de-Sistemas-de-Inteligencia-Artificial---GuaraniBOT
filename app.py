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

from embeddings import calcular_embeddings

load_dotenv()

# --- Configuración ---
MODELO = "llama-3.3-70b-versatile"   # podés probar "llama-3.1-8b-instant" (más rápido)
K_FRAGMENTOS = 8                      # cuántos fragmentos recuperar por pregunta

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
    """Inicializa Groq y el cliente de ChromaDB una sola vez."""
    groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    chroma = chromadb.PersistentClient(path="./chroma_db")
    return groq, chroma


def recuperar(chroma, pregunta: str, k: int) -> list[dict]:
    """Devuelve los k fragmentos más parecidos a la pregunta, con su fuente.

    Cada elemento es un dict {"texto": ..., "fuente": ...}, donde "fuente" es
    el archivo del que salió el fragmento (documento.md, calendario, etc.).

    Buscamos la colección por nombre en cada consulta (en vez de cachear el
    objeto) porque al reindexar con ingest.py la colección se borra y se vuelve
    a crear con otro id interno. Si cacheáramos el objeto, quedaría apuntando a
    una colección que ya no existe y la app rompería hasta reiniciarla.
    """
    coleccion = chroma.get_collection(name="documento")
    # Embebemos la pregunta con el MISMO modelo con que indexamos (ver
    # embeddings.py) y buscamos por ese vector.
    vector_pregunta = calcular_embeddings([pregunta])[0]
    res = coleccion.query(query_embeddings=[vector_pregunta], n_results=k)
    documentos = res["documents"][0]
    metadatos = res["metadatas"][0]
    return [
        {"texto": doc, "fuente": meta.get("fuente", "desconocida")}
        for doc, meta in zip(documentos, metadatos)
    ]


# --- Interfaz ---
st.set_page_config(page_title="Asistente", page_icon="🤖")
st.title("🤖 Asistente del proyecto")
st.caption("Respuestas basadas en tu documento + conocimiento general del modelo.")

groq, chroma = cargar_recursos()

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

    # 1) Recuperar contexto de tus documentos (con la fuente de cada fragmento)
    fragmentos = recuperar(chroma, pregunta, K_FRAGMENTOS)
    contexto = "\n\n---\n\n".join(
        f"[Fuente: {f['fuente']}]\n{f['texto']}" for f in fragmentos
    )

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
            st.markdown(
                f"**Fragmento {i}** — _fuente: {frag['fuente']}_\n\n{frag['texto']}"
            )
