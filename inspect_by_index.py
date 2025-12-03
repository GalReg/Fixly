import random
from libzim.reader import Archive

# --- НАСТРОЙКИ ---
ZIM_FILE_PATH = "/Users/rrgalieva/Documents/fixly_bot/ifixit_ru_all_2025-06.zim"

# Укажите, НАЧИНАЯ С КАКОГО ИНДЕКСА мы будем брать случайные записи.
# 100000 - идеальное значение, чтобы посмотреть, что идет после контента.
START_INDEX = 800000 

# Сколько случайных записей мы хотим посмотреть.
NUMBER_OF_SAMPLES = 500 

# --- СКРИПТ ДЛЯ АНАЛИЗА СЛУЧАЙНЫХ ЗАПИСЕЙ ---

def inspect_random(archive_path, start_index, num_samples):
    print(f"Открываю архив: {archive_path}")
    try:
        archive = Archive(archive_path)
    except Exception as e:
        print(f"Ошибка открытия архива: {e}")
        return

    total_entries = archive.entry_count
    print(f"Всего записей в архиве: {total_entries}")

    # Проверяем, что наш начальный индекс корректен
    if start_index >= total_entries:
        print(f"Ошибка: Начальный индекс ({start_index}) больше или равен общему числу записей ({total_entries}).")
        return

    # Определяем диапазон для взятия случайных индексов
    sampling_range = range(start_index, start_index + 50000)
    
    # Убедимся, что мы не пытаемся взять больше сэмплов, чем есть записей в диапазоне
    if len(sampling_range) < num_samples:
        print(f"Внимание: В диапазоне с {start_index} по {start_index + 50000} всего {len(sampling_range)} записей.")
        print("Будут показаны все записи из этого диапазона.")
        random_indices = list(sampling_range)
    else:
        print(f"\nГенерирую {num_samples} случайных индексов в диапазоне от {start_index} до {start_index + 50000}...")
        # Берем num_samples случайных, уникальных индексов из нашего диапазона
        random_indices = random.sample(sampling_range, num_samples)

    print("\n--- Показываю случайные записи ---")
    
    # Проходимся по нашим случайным индексам и выводим информацию
    for index in sorted(random_indices): # Сортируем для более логичного просмотра
        entry = archive._get_entry_by_id(index)
        
        # Используем форматирование для красивого вывода в одну строку
        path_str = entry.path if entry.path else "N/A"
        title_str = entry.title if entry.title else "N/A"
        
        print(f"Index: {index:<10} | Path: {path_str:<70} | Title: {title_str}")
        
    print("\nАнализ завершен.")


if __name__ == "__main__":
    inspect_random(ZIM_FILE_PATH, START_INDEX, NUMBER_OF_SAMPLES)

