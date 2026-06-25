# Chatbot con RAG — Proyecto Final

Asistente que responde sobre un problema concreto basándose en un documento
que vos escribís. Usa **RAG** (Retrieval-Augmented Generation): busca los
fragmentos relevantes de tu texto y un modelo de lenguaje redacta la respuesta.

## Arquitectura

```
Pregunta del usuario
        │
        ▼
[ChromaDB] ── busca los fragmentos más parecidos en documento.md
        │
        ▼
   CONTEXTO + PREGUNTA
        │
        ▼
[Groq / Llama 3.3] ── redacta la respuesta
        │
        ▼
  Respuesta en la interfaz (Streamlit)
```

- **documento.md** → tu conocimiento (el texto que tenés que hacer).
- **ChromaDB** → base vectorial local que permite la búsqueda por significado.
- **Groq** → API gratuita que corre el modelo de lenguaje.
- **Streamlit** → interfaz de chat en el navegador.

## Puesta en marcha (Windows)

1. **Crear y activar un entorno virtual**
   ```powershell
   cd C:\Users\Santutu\proyecto-chatbot
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Conseguir tu clave de Groq (gratis)**
   - Entrá a https://console.groq.com/keys y creá una API key.
   - Copiá el archivo `.env.example` como `.env` y pegá tu clave ahí.

4. **Escribir tu documento**
   - Editá `documento.md` con el problema que elegiste y su solución.

5. **Indexar el documento**
   ```powershell
   python ingest.py
   ```
   (Repetí este paso cada vez que modifiques `documento.md`.)

6. **Levantar el chatbot**
   ```powershell
   streamlit run app.py
   ```
   Se abre solo en el navegador (http://localhost:8501).

## Ideas para la defensa / informe
- Explicá qué es RAG y por qué evita que el bot "invente".
- Mostrá el panel "Ver fragmentos usados" para evidenciar de dónde sale cada
  respuesta.
- Probá preguntas que SÍ están en el documento y otras que NO, para mostrar
  cómo se comporta en cada caso.
- Posible mejora a futuro: correr el modelo 100% offline con Ollama.
