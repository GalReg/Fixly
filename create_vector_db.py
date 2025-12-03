# create_vector_db.py (v4 - надежный, с чекпоинтами и ручным контролем скорости)

import os
import logging
import time
import math
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import YandexGPTEmbeddings
from langchain.schema import Document
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- НАСТРОЙКИ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

DATA_PATH = "ifixit_data"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
TMP_DB_PATH = "vectorstore/tmp"
FINAL_DB_PATH = "vectorstore/db_faiss" 

# Пауза между запросами к API для соблюдения лимита (10 запросов/сек)
API_REQUEST_DELAY = 0.11 

# --- Получение ключей из .env ---
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    raise ValueError("YANDEX_API_KEY и YANDEX_FOLDER_ID должны быть установлены в .env файле")

# --- ФУНКЦИЯ ЗАГРУЗКИ (без изменений) ---
def load_documents_from_folder(folder_path: str) -> list[Document]:
    # ... (код функции остался тот же, скрыт для краткости) ...
    documents = []
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.txt')]
    def process_file(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            title = "Без заголовка"
            if "Заголовок:" in content:
                title_line = next((line for line in content.split('\n') if "Заголовок:" in line), None)
                if title_line:
                    title = title_line.split("Заголовок:", 1)[1].strip()
            return Document(page_content=content, metadata={"source": os.path.basename(file_path), "title": title})
        except Exception: return None
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {executor.submit(process_file, path): path for path in file_paths}
        progress_bar = tqdm(as_completed(future_to_file), total=len(file_paths), desc="Чтение и парсинг файлов")
        for future in progress_bar:
            result = future.result()
            if result: documents.append(result)
    return documents

# --- ОСНОВНАЯ ФУНКЦИЯ ИНДЕКСАЦИИ (ИЗМЕНЕНА) ---
def main():
    if not os.path.exists(DATA_PATH):
        logging.error(f"Папка с данными '{DATA_PATH}' не найдена.")
        return

    documents = load_documents_from_folder(DATA_PATH)
    if not documents:
        logging.warning("Документы для индексации не найдены.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(documents)
    total_chunks = len(chunks)
    logging.info(f"Документы разбиты на {total_chunks:,} чанков.")

    # Размер одного чекпоинта (например, 1000 чанков). 
    # 5% (5136 чанков) - слишком большой батч для хранения в памяти.
    # Уменьшим его до более разумного размера.
    checkpoint_size = 1000
    num_checkpoints = math.ceil(total_chunks / checkpoint_size)
    logging.info(f"Размер одного чекпоинта: {checkpoint_size:,} чанков. Всего чекпоинтов: {num_checkpoints}.")

    embeddings_model = YandexGPTEmbeddings(api_key=YANDEX_API_KEY, folder_id=YANDEX_FOLDER_ID)

    os.makedirs(TMP_DB_PATH, exist_ok=True)

    # Проходим по всем чанкам, порциями (чекпоинтами)
    for i in range(0, total_chunks, checkpoint_size):
        batch_num = (i // checkpoint_size) + 1
        batch_path = os.path.join(TMP_DB_PATH, f"batch_{batch_num}")

        if os.path.exists(batch_path):
            logging.info(f"Чекпоинт {batch_num}/{num_checkpoints} уже существует. Пропускаем.")
            continue

        logging.info(f"Начинаем обработку чекпоинта {batch_num}/{num_checkpoints}...")
        batch_chunks = chunks[i:i + checkpoint_size]
        
        texts_to_embed = [chunk.page_content for chunk in batch_chunks]
        metadatas = [chunk.metadata for chunk in batch_chunks]
        
        all_embeddings = []
        progress_bar = tqdm(texts_to_embed, total=len(texts_to_embed), desc=f"Чекпоинт {batch_num}")
        
        # --- НАДЕЖНЫЙ ПОСЛЕДОВАТЕЛЬНЫЙ ЦИКЛ С ПАУЗОЙ ---
        for text in progress_bar:
            try:
                embedding = embeddings_model.embed_query(text)
                all_embeddings.append(embedding)
                time.sleep(API_REQUEST_DELAY) 
            except Exception as e:
                logging.error(f"Ошибка при получении эмбеддинга: {e}. Пропускаем чанк.")
                # Добавляем "пустой" вектор
                all_embeddings.append([0.0] * 256) # 256 - размерность эмбеддингов YandexGPT

        # Убедимся, что мы получили эмбеддинги
        if not all_embeddings or len(all_embeddings) != len(texts_to_embed):
             logging.error(f"В батче {batch_num} не удалось создать эмбеддинги. Пропускаем сохранение.")
             continue
        
        text_embedding_pairs = list(zip(texts_to_embed, all_embeddings))
        
        # Создаем и сохраняем временный FAISS индекс
        db = FAISS.from_embeddings(text_embedding_pairs, embeddings_model, metadatas=metadatas)
        db.save_local(batch_path)
        logging.info(f"Чекпоинт {batch_num} успешно сохранен в '{batch_path}'.")

    logging.info("Все чекпоинты созданы!")
    logging.info(f"Теперь запустите скрипт 'merge_vector_db.py', чтобы объединить их в финальную базу в '{FINAL_DB_PATH}'.")

if __name__ == "__main__":
    main()
