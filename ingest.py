"""
ingest.py — Indexa tu documento en la base vectorial (ChromaDB).

Lee documento.md, lo parte en fragmentos ("chunks"), calcula sus embeddings
y los guarda en ./chroma_db. Corré este script UNA vez (y de nuevo cada vez
que cambies el documento) antes de levantar la app.

Uso:  python ingest.py
"""

import chromadb

DOCUMENTO = "documento.md"
TAM_CHUNK = 800       # caracteres por fragmento
SOLAPADO = 100        # caracteres que se repiten entre fragmentos (da contexto)


def chunkear(texto: str, max_chars: int, overlap: int) -> list[str]:
    """Parte el texto en fragmentos con un poco de solapamiento."""
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + max_chars
        chunks.append(texto[inicio:fin].strip())
        inicio = fin - overlap
    return [c for c in chunks if c]  # descarta vacíos


def main() -> None:
    with open(DOCUMENTO, "r", encoding="utf-8") as f:
        texto = f.read()

    chunks = chunkear(texto, TAM_CHUNK, SOLAPADO)

    cliente = chromadb.PersistentClient(path="./chroma_db")

    # Si ya existía, la borramos para reindexar desde cero.
    try:
        cliente.delete_collection("documento")
    except Exception:
        pass

    # Sin pasar embedding_function, Chroma usa su modelo por defecto
    # (all-MiniLM-L6-v2, liviano, se descarga solo la primera vez).
    coleccion = cliente.create_collection(name="documento")

    coleccion.add(
        documents=chunks,
        ids=[f"chunk-{i}" for i in range(len(chunks))],
    )

    print(f"OK: indexados {len(chunks)} fragmentos de '{DOCUMENTO}'.")


if __name__ == "__main__":
    main()
