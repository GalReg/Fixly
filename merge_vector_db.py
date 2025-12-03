# merge_vector_db.py

import os
import logging
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import YandexGPTEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm

# --- НАСТРОЙКИ ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# Пути к папкам
TMP_DB_PATH = "vectorstore/tmp"
FINAL_DB_PATH = "vectorstore/db_faiss"

# --- Получение ключей для загрузки эмбеддера ---
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
if not YANDEX_API_KEY or not YANDEX_FOLDER_ID:
    raise ValueError("YANDEX_API_KEY и YANDEX_FOLDER_ID должны быть установлены в .env файле")

def merge_faiss_indexes():
    """
    Объединяет все временные FAISS-индексы из папки tmp в один финальный.
    """
    if not os.path.exists(TMP_DB_PATH) or not os.listdir(TMP_DB_PATH):
        logging.error(f"Папка с временными индексами '{TMP_DB_PATH}' пуста или не существует.")
        logging.error("Сначала запустите 'create_vector_db.py' для создания чекпоинтов.")
        return

    # 1. Инициализируем модель эмбеддингов, чтобы загрузить индексы
    logging.info("Инициализация модели эмбеддингов для загрузки индексов...")
    embeddings_model = YandexGPTEmbeddings(api_key=YANDEX_API_KEY, folder_id=YANDEX_FOLDER_ID)

    # 2. Находим все папки с батчами
    batch_folders = sorted(
        [os.path.join(TMP_DB_PATH, d) for d in os.listdir(TMP_DB_PATH) if d.startswith("batch_")],
        key=lambda x: int(x.split('_')[-1]) # Сортируем по номеру батча
    )

    if not batch_folders:
        logging.error("Не найдено папок с батчами в '{TMP_DB_PATH}'.")
        return
        
    # 3. Загружаем самый первый индекс как основу для объединения
    logging.info(f"Загрузка основного индекса из '{batch_folders[0]}'...")
    try:
        # allow_dangerous_deserialization=True - необходимо для загрузки локальных индексов FAISS
        main_db = FAISS.load_local(batch_folders[0], embeddings_model, allow_dangerous_deserialization=True)
    except Exception as e:
        logging.critical(f"Не удалось загрузить основной батч! Ошибка: {e}")
        return

    # 4. В цикле загружаем остальные индексы и "мержим" их в основной
    logging.info("Начинаем объединение всех временных индексов...")
    for folder in tqdm(batch_folders[1:], desc="Объединение индексов"):
        try:
            db_to_merge = FAISS.load_local(folder, embeddings_model, allow_dangerous_deserialization=True)
            main_db.merge_from(db_to_merge)
        except Exception as e:
            logging.warning(f"Не удалось загрузить или объединить индекс из '{folder}'. Ошибка: {e}. Пропускаем.")
    
    # 5. Сохраняем объединенный, финальный индекс
    logging.info("Все индексы объединены. Сохраняем финальную базу данных...")
    os.makedirs(os.path.dirname(FINAL_DB_PATH), exist_ok=True)
    main_db.save_local(FINAL_DB_PATH)

    logging.info(f"Финальная векторная база данных успешно сохранена в '{FINAL_DB_PATH}'.")
    logging.info("Теперь вы можете использовать ее в своем боте. Папку 'vectorstore/tmp' можно удалить для экономии места.")

if __name__ == "__main__":
    merge_faiss_indexes()

