#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script para indexar metadata y features de canciones en Elasticsearch usando múltiples núcleos.

Archivos a indexar (ubicados en MUSIC_4_ALL_PATH):
  - id_information.csv      (id, artist, song, album_name)
  - id_lang.csv             (id, lang)
  - id_metadata.csv         (id, spotify_id, popularity, release,
                             danceability, energy, key, mode, valence, tempo, duration_ms)

Archivos a indexar (ubicados en MUSIC_4_ALL_ONION_PATH):
  - id_genres_tf-idf.tsv.bz2 (id + columnas de géneros con valores float)
  - id_lyrics_word2vec.tsv.bz2 (id + columnas de word2vec con valores float)

El índice resultante se llamará “song” y contendrá:
  • Campos básicos: artist, song, album_name, lang, spotify_id, popularity, release, danceability,
    energy, key, mode, valence, tempo, duration_ms
  • Dos vectores densos (dense_vector):
      - genres_embedding    (tf-idf de géneros)
      - lyrics_embedding    (word2vec de lyrics)

El script usa `elasticsearch.helpers.parallel_bulk` para paralelizar el envío de documentos
a Elasticsearch en tantos hilos como núcleos de CPU detecte.
"""

import os
import json
import multiprocessing
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk, BulkIndexError
from app.core.config import settings

# ---------------------- Configuración general ----------------------

# Ajustar estas rutas según tu sistema
MUSIC_4_ALL_PATH = settings.MUSIC_4_ALL_PATH
MUSIC_4_ALL_ONION_PATH = settings.MUSIC_4_ALL_ONION_PATH

# URL de Elasticsearch (modificar si hace falta autenticación, HTTPS, etc.)
INDEX_NAME = "song"

# ---------------------- Funciones auxiliares ----------------------

def crear_indice_song(es_client, genre_dim, lyrics_dim, genre_feature_names, lyrics_feature_names):
    """
    Crea el índice "song" con el mapeo apropiado.
    Incluye campos básicos + dense_vector para géneros y lyrics.
    
    Parámetros:
      - es_client: instancia de Elasticsearch
      - genre_dim: número de dimensiones del vector de géneros
      - lyrics_dim: número de dimensiones del vector de lyrics
      - genre_feature_names: lista de nombres de columnas individuales de géneros (si se desea guardar)
      - lyrics_feature_names: lista de nombres de columnas individuales de lyrics (si se desea guardar)
    """
    if es_client.indices.exists(index=INDEX_NAME):
        print(f"El índice '{INDEX_NAME}' ya existe. Se eliminará y se volverá a crear.")
        es_client.indices.delete(index=INDEX_NAME)

    # Construir el mapeo
    mapping = {
        "mappings": {
            "properties": {
                "id":            {"type": "keyword"},
                "artist":        {"type": "text"},
                "song":          {"type": "text"},
                "album_name":    {"type": "text"},
                "lang":          {"type": "keyword"},
                "spotify_id":    {"type": "keyword"},
                "popularity":    {"type": "float"},
                "release":       {"type": "integer"},
                "danceability":  {"type": "float"},
                "energy":        {"type": "float"},
                "key":           {"type": "integer"},
                "mode":          {"type": "integer"},
                "valence":       {"type": "float"},
                "tempo":         {"type": "float"},
                "duration_ms":   {"type": "integer"},
                # Vector denso de géneros
                "genres_embedding": {
                    "type": "dense_vector",
                    "dims": genre_dim
                },
                # Vector denso de lyrics
                "lyrics_embedding": {
                    "type": "dense_vector",
                    "dims": lyrics_dim
                }
            }
        }
    }

    # (Opcional) Si quieres guardar también cada valor individual de género y lyric:
    # Descomentar las siguientes líneas
    # for fname in genre_feature_names:
    #     mapping["mappings"]["properties"][fname] = {"type": "float"}
    # for fname in lyrics_feature_names:
    #     mapping["mappings"]["properties"][fname] = {"type": "float"}

    # Luego creas el índice normalmente
    es_client.indices.create(index=INDEX_NAME, body=mapping)
    print(f"✅ Índice '{INDEX_NAME}' creado correctamente.")



def cargar_dataframe_basico(path_info, path_lang, path_meta):
    """
    Lee los CSV básicos y los une en un solo DataFrame:
      - id_information.csv  → id, artist, song, album_name
      - id_lang.csv         → id, lang
      - id_metadata.csv     → id, spotify_id, popularity, release, danceability,
                              energy, key, mode, valence, tempo, duration_ms

    Retorna:
      DataFrame con columnas [id, artist, song, album_name, lang,
                              spotify_id, popularity, release, danceability, energy,
                              key, mode, valence, tempo, duration_ms]
    """
    # Leer id_information.csv
    df_info = pd.read_csv(path_info, sep="\t", dtype=str)
    # Asegurarse que la columna 'id' exista
    if 'id' not in df_info.columns:
        raise KeyError(f"El archivo {path_info} no contiene columna 'id'.")

    # Leer id_lang.csv
    df_lang = pd.read_csv(path_lang, sep="\t", dtype=str)
    if 'id' not in df_lang.columns or 'lang' not in df_lang.columns:
        raise KeyError(f"El archivo {path_lang} debe contener columnas 'id' y 'lang'.")

    # Leer id_metadata.csv
    df_meta = pd.read_csv(path_meta, sep="\t", dtype={
        "id": str,
        "spotify_id": str,
        "popularity": float,
        "release": int,
        "danceability": float,
        "energy": float,
        "key": float,
        "mode": float,
        "valence": float,
        "tempo": float,
        "duration_ms": float
    })
    if 'id' not in df_meta.columns:
        raise KeyError(f"El archivo {path_meta} no contiene columna 'id'.")

    # Merge: info + lang
    df = pd.merge(df_info, df_lang, on="id", how="left")
    # Merge con metadata
    df = pd.merge(df, df_meta, on="id", how="left")

    # Rellenar NaNs si es necesario (por ejemplo popularidad faltante → 0)
    # df.fillna({"popularity": 0.0, "release": 0}, inplace=True)
    # Para diferencias puntuales, se puede usar fillna().

    return df


def cargar_dataframe_vecs(path_tsv_bz2):
    """
    Lee un archivo TSV comprimido (bz2) en pandas DataFrame.
    Se asume que la primera columna es 'id' y las demás son floats.

    Retorna:
      - df (pandas.DataFrame) con la primera columna 'id'
      - feature_names: lista de nombres de columnas (excepto 'id')
    """
    df = pd.read_csv(path_tsv_bz2, sep="\t", compression="bz2", dtype=str)
    if 'id' not in df.columns:
        raise KeyError(f"El archivo {path_tsv_bz2} no contiene columna 'id'.")
    # Convertir todas las columnas excepto 'id' a float
    feature_cols = [c for c in df.columns if c != "id"]
    for col in feature_cols:
        df[col] = df[col].astype(float)

    return df, feature_cols

import numpy as np

def sanitize_vector(vec):
    return [float(v) if (not pd.isna(v) and np.isfinite(v)) else 0.0 for v in vec]


def generar_acciones_para_indexar(df_combined, genre_cols, lyrics_cols):
    """
    Generador de acciones (dicts) para parallel_bulk.

    Cada documento tendrá:
      {
        "_index": INDEX_NAME,
        "_id": id,
        "_source": {
            "id": id,
            "artist": ...,
            "song": ...,
            "album_name": ...,
            "lang": ...,
            "spotify_id": ...,
            "popularity": ...,
            "release": ...,
            "danceability": ...,
            "energy": ...,
            "key": ...,
            "mode": ...,
            "valence": ...,
            "tempo": ...,
            "duration_ms": ...,
            "genres_embedding": [float, float, ...],
            "lyrics_embedding": [float, float, ...]
            # (Opcional) cada campo individual de genre_cols y lyrics_cols
        }
      }
    """
    
    for _, row in df_combined.iterrows():
        yield {
            "_index": "song",
            "_id": row["id"],
            "_source": {
                "id": row["id"],
                "artist": row["artist"],
                "song": row["song"],
                "album_name": row["album_name"],
                "lang": row["lang"],
                "metadata": {
                    "spotify_id": row.get("spotify_id"),
                    "popularity": row.get("popularity"),
                    "release": row.get("release"),
                    "danceability": row.get("danceability"),
                    "energy": row.get("energy"),
                    "key": row.get("key"),
                    "mode": row.get("mode"),
                    "valence": row.get("valence"),
                    "tempo": row.get("tempo"),
                    "duration_ms": row.get("duration_ms")
                },
                "genres_vector": sanitize_vector(
                    row[genre_cols].astype(float).fillna(0.0).tolist()
                ),
                "lyrics_vector": sanitize_vector(
                    row[lyrics_cols].astype(float).fillna(0.0).tolist()
                )
            }
        }

        # (Opcional) Si quieres guardar cada columna de géneros y lyrics:
        # for col in genre_cols:
        #     source[col] = row[col]
        # for col in lyrics_cols:
        #     source[col] = row[col]



# ---------------------- Bloque principal ----------------------

if __name__ == "__main__":

    # 1. Conectar a Elasticsearch
    es = Elasticsearch(settings.ELASTIC_URL)

    # 2. Cargar DataFrames básicos
    path_info = os.path.join(MUSIC_4_ALL_PATH, "id_information.csv")
    path_lang = os.path.join(MUSIC_4_ALL_PATH, "id_lang.csv")
    path_meta = os.path.join(MUSIC_4_ALL_PATH, "id_metadata.csv")
    print("Cargando datos básicos (id_information, id_lang, id_metadata)...")
    df_basic = cargar_dataframe_basico(path_info, path_lang, path_meta)
    print(f"  → Filas cargadas (básico): {len(df_basic)}")

    # 3. Cargar DataFrames de vectores (géneros y lyrics)
    path_genres = os.path.join(MUSIC_4_ALL_ONION_PATH, "id_genres_tf-idf.tsv.bz2")
    path_lyrics = os.path.join(MUSIC_4_ALL_ONION_PATH, "id_lyrics_word2vec.tsv.bz2")
    print("Cargando características de géneros (TF-IDF)...")
    df_genres, genre_cols = cargar_dataframe_vecs(path_genres)
    print(f"  → Géneros: {len(df_genres)} filas, {len(genre_cols)} dimensiones")
    print("Cargando características de lyrics (Word2Vec)...")
    df_lyrics, lyrics_cols = cargar_dataframe_vecs(path_lyrics)
    print(f"  → Lyrics: {len(df_lyrics)} filas, {len(lyrics_cols)} dimensiones")

    # 4. Unir todos los DataFrames en uno solo (basico + géneros + lyrics)
    print("Uniendo DataFrames básicos + géneros + lyrics ...")
    df_temp = pd.merge(df_basic, df_genres, how="left", on="id")
    df_all  = pd.merge(df_temp, df_lyrics, how="left", on="id")
    print(f"  → Total de filas combinadas: {len(df_all)}")

    # 5. Crear índice en ES con mapeo adecuado
    print("Creando índice en Elasticsearch (si no existe)...")
    crear_indice_song(
        es_client=es,
        genre_dim=len(genre_cols),
        lyrics_dim=len(lyrics_cols),
        genre_feature_names=genre_cols,
        lyrics_feature_names=lyrics_cols
    )

    # 6. Preparar y ejecutar parallel_bulk para indexar
    print("Iniciando indexación en paralelo con parallel_bulk...")
    num_workers = multiprocessing.cpu_count()/2
    print(f"  → Usando {num_workers} hilos (núcleos de CPU).")

    # Generador de acciones
    actions_generator = generar_acciones_para_indexar(df_all, genre_cols, lyrics_cols)

    # parallel_bulk retorna tuplas (success, info)
    success_count = 0
    error_count = 0


    try:
        for ok, info in parallel_bulk(
            client=es,
            actions=actions_generator,
            thread_count=int(num_workers),
            chunk_size=1000
        ):
            if ok:
                success_count += 1
            else:
                error_count += 1
                print("❌ Falla en documento:", info)
    except BulkIndexError as e:
        print(f"\n🔥 {len(e.errors)} errores en la indexación.")
        for err in e.errors[:5]:  # Solo muestra algunos
            print(json.dumps(err, indent=2))
        

    print(f"Indexación finalizada: {success_count} documentos indexados correctamente, {error_count} errores.")
