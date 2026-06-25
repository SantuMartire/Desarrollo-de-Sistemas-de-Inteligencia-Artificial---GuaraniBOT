"""
ingest.py — Indexa tus documentos en la base vectorial (ChromaDB).

Lee cada archivo de FUENTES, lo parte en fragmentos ("chunks"), calcula sus
embeddings y los guarda en ./chroma_db. A cada fragmento le guarda de qué
archivo salió (metadato "fuente"), para poder mostrarlo en la app y para que
el modelo sepa de dónde viene cada dato.

Corré este script UNA vez (y de nuevo cada vez que cambies algún documento)
antes de levantar la app.

Uso:  python ingest.py
"""

import os

import chromadb

from embeddings import MODELO_EMB, calcular_embeddings

# Archivos que se indexan. Para sumar o sacar fuentes, editá esta lista.
# (Dejamos afuera Datos/plan_estudios.md: es un legajo personal y la
# conversión del PDF quedó rota; su info ya está en los otros archivos.)
FUENTES = [
    "documento.md",
    os.path.join("Datos", "Calendario Academico - Beltran.md"),
    os.path.join("Datos", "Estructura Curricular y Correlatividades - TSCIA - 2024.md"),
]

TAM_CHUNK = 1000      # tope de caracteres por fragmento
SOLAPADO = 100        # caracteres que se repiten al subdividir (da contexto)


def chunkear_plano(texto: str, max_chars: int, overlap: int) -> list[str]:
    """Parte un texto en fragmentos por tamaño, con un poco de solapamiento.

    Se usa solo como respaldo, cuando una sección es más larga que max_chars.
    """
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + max_chars
        chunks.append(texto[inicio:fin].strip())
        inicio = fin - overlap
    return [c for c in chunks if c]  # descarta vacíos


def chunkear(texto: str, max_chars: int, overlap: int) -> list[str]:
    """Parte el texto respetando los encabezados Markdown.

    Cada sección (lo que va de un encabezado `#`/`##`/`###` hasta el
    siguiente) se mantiene junta y arranca con su título. Así, por ejemplo,
    la sección "### 1° Año" queda en un único fragmento con su lista completa
    de materias, y su embedding matchea bien con preguntas tipo "materias de
    primer año". Si una sección es más larga que max_chars, se subdivide
    repitiendo el título al inicio de cada parte para no perder el contexto.
    """
    secciones: list[str] = []
    actual: list[str] = []
    for linea in texto.split("\n"):
        # Una línea de encabezado abre una sección nueva (si ya veníamos
        # juntando contenido, cerramos la sección anterior).
        if linea.lstrip().startswith("#") and actual:
            secciones.append("\n".join(actual).strip())
            actual = [linea]
        else:
            actual.append(linea)
    if actual:
        secciones.append("\n".join(actual).strip())

    chunks: list[str] = []
    for seccion in secciones:
        if not seccion:
            continue
        if len(seccion) <= max_chars:
            chunks.append(seccion)
            continue
        # Sección demasiado larga: la subdividimos, repitiendo el encabezado.
        primera, _, resto = seccion.partition("\n")
        if primera.lstrip().startswith("#"):
            encabezado, cuerpo = primera.strip(), resto
        else:
            encabezado, cuerpo = "", seccion
        for parte in chunkear_plano(cuerpo, max_chars, overlap):
            chunks.append(f"{encabezado}\n{parte}".strip() if encabezado else parte)
    # Descartamos fragmentos muy cortos: en general son ruido de la conversión
    # del PDF (encabezados sueltos como "RESOLUCIÓN N°:", "2 | Página", etc.)
    # sin información útil, y si no ensucian los resultados de la búsqueda.
    return [c for c in chunks if len(c.strip()) >= 50]


def main() -> None:
    cliente = chromadb.PersistentClient(path="./chroma_db")

    # Si ya existía, la borramos para reindexar desde cero.
    try:
        cliente.delete_collection("documento")
    except Exception:
        pass

    # Métrica "cosine": es la correcta para este tipo de embeddings de frases.
    # (Con la L2 por defecto, los rankings salían distorsionados.)
    coleccion = cliente.create_collection(
        name="documento",
        metadata={"hnsw:space": "cosine"},
    )

    documentos: list[str] = []
    metadatos: list[dict] = []
    ids: list[str] = []

    for ruta in FUENTES:
        nombre = os.path.basename(ruta)
        with open(ruta, "r", encoding="utf-8") as f:
            texto = f.read()

        # Chunkeamos cada archivo por separado: así ningún fragmento mezcla
        # contenido de dos archivos y la "fuente" que guardamos es exacta.
        chunks = chunkear(texto, TAM_CHUNK, SOLAPADO)
        for i, chunk in enumerate(chunks):
            documentos.append(chunk)
            metadatos.append({"fuente": nombre})
            ids.append(f"{nombre}-chunk-{i}")

        print(f"  - {nombre}: {len(chunks)} fragmentos")

    # Calculamos los embeddings con NUESTRO modelo multilingüe y se los pasamos
    # a Chroma explícitamente (así no usa su modelo por defecto, malo en español).
    print(f"Calculando embeddings con '{MODELO_EMB}' (la 1ra vez baja el modelo)...")
    vectores = calcular_embeddings(documentos)

    coleccion.add(
        documents=documentos,
        embeddings=vectores,
        metadatas=metadatos,
        ids=ids,
    )

    print(
        f"OK: indexados {len(documentos)} fragmentos "
        f"de {len(FUENTES)} archivos en ./chroma_db."
    )


if __name__ == "__main__":
    main()
