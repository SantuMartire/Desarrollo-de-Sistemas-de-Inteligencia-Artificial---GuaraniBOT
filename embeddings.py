"""
embeddings.py — Modelo de embeddings multilingüe (corre local con fastembed).

Convierte textos en vectores numéricos para que ChromaDB pueda buscar por
significado. Usamos un modelo MULTILINGÜE (entrenado también en español),
porque el modelo por defecto de Chroma (all-MiniLM-L6-v2) está pensado para
inglés y en español recupera mal (no asociaba "materias de primer año" con la
sección correspondiente).

Lo importan tanto ingest.py (para indexar el documento) como app.py (para
buscar la pregunta). Es imprescindible que ambos usen EXACTAMENTE el mismo
modelo: si no, los vectores no son comparables y la búsqueda no tiene sentido.

El modelo (~220 MB) se descarga solo la primera vez y queda cacheado en disco.
"""

from functools import lru_cache

from fastembed import TextEmbedding

# Modelo multilingüe liviano y bueno en español (384 dimensiones).
MODELO_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def _modelo() -> TextEmbedding:
    """Carga el modelo una sola vez por proceso (la primera llamada tarda)."""
    return TextEmbedding(model_name=MODELO_EMB)


def calcular_embeddings(textos: list[str]) -> list[list[float]]:
    """Devuelve el vector de cada texto, como lista de floats (formato Chroma)."""
    return [vector.tolist() for vector in _modelo().embed(textos)]
