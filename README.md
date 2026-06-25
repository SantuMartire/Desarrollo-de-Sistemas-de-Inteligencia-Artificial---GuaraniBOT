# GuaraníBOT — Asistente de inscripción a materias (TSCIA)

> Chatbot que responde dudas sobre **cómo anotarse a las materias** de la
> Tecnicatura Superior en Ciencia de Datos e IA (Instituto Tecnológico Beltrán):
> fechas de inscripción, plan de estudios, correlatividades y el paso a paso de
> SIU Guaraní. Usa **RAG** para responder con datos reales del instituto y no
> "inventar".

---

## ¿Qué resuelve?

Anotarse a las materias suele fallar por cosas concretas: el período todavía no
abrió, no cumplís las correlativas, o te falta un trámite. La info está repartida
entre el plan de estudios, el régimen de correlatividades y el calendario
académico. **GuaraníBOT junta todo eso y te lo responde en lenguaje natural**,
citando de qué documento salió cada dato.

---

## ¿Cómo funciona? — 

**RAG** (*Retrieval-Augmented Generation*) = **buscar primero, redactar después**.

En vez de confiar en la memoria del modelo (que puede inventar), el bot:

1. **Busca** en tus documentos los fragmentos más parecidos a la pregunta.
2. Le **pasa esos fragmentos** como contexto a un modelo de lenguaje.
3. El modelo **redacta** la respuesta basándose en ese contexto.

Así las respuestas quedan ancladas a información real y verificable.

```
        Tu pregunta
            │
            ▼
   ┌─────────────────────┐     se convierte la pregunta en un vector
   │  embeddings.py      │     (modelo multilingüe, local)
   └─────────────────────┘
            │
            ▼
   ┌─────────────────────┐     busca por significado los 8 fragmentos
   │  ChromaDB           │     más parecidos (métrica coseno)
   └─────────────────────┘
            │
            ▼
   CONTEXTO (fragmentos + su fuente)  +  PREGUNTA
            │
            ▼
   ┌─────────────────────┐     redacta la respuesta priorizando
   │  Groq / Llama 3.3   │     el contexto recibido
   └─────────────────────┘
            │
            ▼
   Respuesta en el chat (Streamlit)  +  panel "Ver fragmentos usados"
```

---

## Estructura del proyecto

| Archivo | Rol |
|---|---|
| **`documento.md`** | Base de conocimiento curada (SIU Guaraní, plan, correlativas, calendario). |
| **`Datos/`** | Fuentes oficiales en bruto (calendario y correlatividades del instituto). |
| **`embeddings.py`** | Convierte texto → vectores con un modelo **multilingüe** local. |
| **`ingest.py`** | Lee los documentos, los corta en fragmentos y los **indexa** en ChromaDB. |
| **`app.py`** | La **interfaz de chat** (Streamlit) y el orquestador del flujo RAG. |
| **`chroma_db/`** | Base vectorial persistente (se genera al correr `ingest.py`). |
| **`.env`** | Tu clave de Groq (`GROQ_API_KEY`). No se sube al repo. |

---

## Cómo funciona el código, archivo por archivo

###  `embeddings.py` — el "traductor" a vectores
Carga **una sola vez** un modelo multilingüe
(`paraphrase-multilingual-MiniLM-L12-v2`, ~220 MB, corre 100 % local) y expone
`calcular_embeddings(textos)`, que convierte cada texto en un vector de 384
números. Lo usan **ingest.py** y **app.py**: es clave que ambos usen el **mismo
modelo**, si no los vectores no serían comparables.

###  `ingest.py` — la indexación (correr antes de la app)
1. Lee los archivos de la lista `FUENTES` (`documento.md` + los 2 archivos
   limpios de `Datos/`).
2. **Chunking por encabezados:** parte cada documento respetando los títulos
   Markdown, así cada sección (ej. *"Materias de 1° año"*) queda en un fragmento
   propio y completo. Descarta fragmentos basura demasiado cortos.
3. A cada fragmento le guarda de qué archivo salió (metadato **`fuente`**).
4. Calcula los embeddings y los guarda en `chroma_db/` con métrica **coseno**.

>  Cada vez que edites `documento.md` o un archivo de `Datos/`, volvé a correr
> `python ingest.py` para que el cambio entre.

###  `app.py` — el chat y el flujo RAG
Por cada pregunta del usuario:
1. **Recupera** los `K = 8` fragmentos más parecidos (vía `embeddings.py` + ChromaDB).
2. **Arma el contexto** con esos fragmentos, etiquetando cada uno con su fuente.
3. **Llama a Groq** (`llama-3.3-70b-versatile`) con un *system prompt* que le
   ordena: responder en español, priorizar el contexto y **no inventar** si no
   está la respuesta.
4. **Muestra** la respuesta en streaming y, en un panel desplegable, **los
   fragmentos exactos que usó** (ideal para demostrar de dónde sale cada dato).

---

##  Puesta en marcha (Windows)

1. **Crear y activar el entorno virtual**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Instalar dependencias**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configurar la clave de Groq (gratis)**
   - Sacá tu API key en https://console.groq.com/keys
   - Pegala en el archivo `.env`:
     ```
     GROQ_API_KEY=tu_clave_aca
     ```

4. **Indexar los documentos**
   ```powershell
   python ingest.py
   ```
   > La **primera vez** descarga el modelo de embeddings (~220 MB); después
   > queda cacheado.

5. **Levantar el chatbot**
   ```powershell
   streamlit run app.py
   ```
   Se abre solo en el navegador (http://localhost:8501).

---

##  10 preguntas para probarlo

Preguntas que recorren los cuatro temas que cubre el bot. Ideales para la demo:

**Fechas e inscripción**
1. ¿En qué fecha me inscribo a las materias del 1er cuatrimestre de la TSCIA?
2. ¿Cuándo es el receso invernal en 2026?
3. ¿Hasta cuándo puedo presentar trámites por equivalencia de materias?
4. ¿Cuándo son los exámenes finales de febrero?

**Plan de estudios y materias**
5. ¿Cuáles son las materias de primer año?
6. ¿Qué materias se cursan en tercer año?
7. ¿Qué diferencia hay entre una materia "E" y una "G"?

**Correlatividades y requisitos**
8. ¿Qué condiciones necesito para anotarme a materias de 2° año?
9. ¿Por qué puede rebotarme el sistema cuando intento inscribirme?

**SIU Guaraní**
10. ¿Cuáles son los pasos para inscribirme por SIU Guaraní?


---

## 📦 Stack

`Python` · `Streamlit` · `ChromaDB` · `fastembed` · `Groq (Llama 3.3)` · `RAG`
