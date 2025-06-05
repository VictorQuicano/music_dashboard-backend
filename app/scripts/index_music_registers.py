import os
import csv
from datetime import datetime
from multiprocessing import Pool, Manager, cpu_count
from elasticsearch import Elasticsearch
from app.core.config import settings
from app.core.logger import logger

es = Elasticsearch(settings.ELASTIC_URL)

def process_file(args):
    file_path, filename = args
    indexed = skipped = failed = 0

    logger.info(f"📄 [PROCESS] Procesando archivo: {filename}")

    try:
        with open(file_path, newline='') as tsvfile:
            reader = csv.reader(tsvfile, delimiter='\t')
            for i, row in enumerate(reader, 1):
                if len(row) != 3:
                    logger.warning(f"⚠️  Fila malformada en {filename}, línea {i}: {row}")
                    skipped += 1
                    continue

                user_id, track_id, timestamp_str = row
                try:
                    timestamp = datetime.strptime(timestamp_str.strip(), "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    logger.warning(f"⚠️  Timestamp inválido en {filename}, línea {i}: '{timestamp_str}'")
                    skipped += 1
                    continue

                doc = {
                    "user_id": user_id.strip(),
                    "track_id": track_id.strip(),
                    "timestamp": timestamp.isoformat(),
                    "year": timestamp.year,
                    "month": timestamp.month
                }

                try:
                    es.index(index="music", document=doc)
                    indexed += 1
                except Exception as e:
                    logger.error(f"❌ Error indexando en {filename}, línea {i}: {e}")
                    failed += 1
    except Exception as e:
        logger.error(f"❌ Error general al procesar {filename}: {e}")

    logger.info(f"✅ [PROCESS] Completado {filename} | ✅: {indexed} ⚠️: {skipped} ❌: {failed}")
    return filename, {"indexed": indexed, "skipped": skipped, "failed": failed}


def index_music_registers():
    folder = settings.MUSIC_REGISTERS
    logger.info(f"📂 Iniciando indexación con 10 procesos desde: {folder}")

    if not os.path.exists(folder):
        logger.error(f"❌ La carpeta {folder} no existe.")
        return

    files = [
        (os.path.join(folder, filename), filename)
        for filename in os.listdir(folder)
        if filename.endswith(".tsv")
    ]

    with Pool(processes=10) as pool:  # Usa 10 procesos
        results = pool.map(process_file, files)

    stats = dict(results)
    total_indexed = sum(s["indexed"] for s in stats.values())
    total_skipped = sum(s["skipped"] for s in stats.values())
    total_failed = sum(s["failed"] for s in stats.values())

    logger.info("📊 Indexación terminada.")
    logger.info(f"📦 Archivos procesados: {len(stats)}")
    logger.info(f"✅ Total indexados: {total_indexed}")
    logger.info(f"⚠️  Total omitidos: {total_skipped}")
    logger.info(f"❌ Total fallidos: {total_failed}")

if __name__ == "__main__":
    index_music_registers()
