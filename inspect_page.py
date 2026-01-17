from libzim.reader import Archive
from bs4 import BeautifulSoup

# --- НАСТРОЙКИ ПОИСКОВИКА-ИНСПЕКТОРА ---
ZIM_FILE_PATH = "/Users/rrgalieva/Documents/fixly_bot/ifixit_ru_all_2025-06.zim"

# Введите сюда ключевое слово или фразу для поиска в путях (path).
# Лучше использовать английскую часть URL, она более стабильна.
# Давайте попробуем этот, он достаточно уникален.
TARGET_KEYWORD = "Refrigerator"  # Refrigerator+Compressor+Running+But+Not+Cooling

# --- КОНЕЦ НАСТРОЕК ---


def save_inspection_files(archive, target_path):
    """Эта функция берет ТОЧНЫЙ путь и сохраняет HTML/текст файлы."""
    print(f"\n--- Инспектирую выбранный путь: '{target_path}' ---")

    entry = archive.get_entry_by_path(target_path)
    item = entry.get_item()
    html_content = str(item.content, "utf-8")

    # 1. Сохраняем сырой HTML
    output_html_file = "inspector_output.html"
    with open(output_html_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[УСПЕХ] Сырой HTML сохранен в: '{output_html_file}'")

    soup = BeautifulSoup(html_content, "html.parser")

    # 2. Пытаемся извлечь текст "умно"
    main_content = None
    # Ищем контент по разным возможным ID
    if soup.find("div", id="wikitext"):
        main_content = soup.find("div", id="wikitext")
        print("Найден основной блок контента: <div id='wikitext'>")
    elif soup.find("div", id="guide-contents"):
        main_content = soup.find("div", id="guide-contents")
        print("Найден основной блок контента: <div id='guide-contents'>")
    # Добавьте сюда другие возможные id, если найдете их при инспекции

    if main_content:
        text_from_div = main_content.get_text(separator="\n", strip=True)
        output_txt_smart_file = "inspector_output_smart.txt"
        with open(output_txt_smart_file, "w", encoding="utf-8") as f:
            f.write(text_from_div)
        print(f"[УСПЕХ] 'Умный' текст сохранен в: '{output_txt_smart_file}'")
    else:
        print(
            "[ВНИМАНИЕ] Не удалось найти стандартный блок контента (wikitext, guide-contents)."
        )
        # Сохраним 'тупой' текст как запасной вариант
        text_from_get_text = soup.get_text(separator="\n", strip=True)
        output_txt_get_text_file = "inspector_output_get_text.txt"
        with open(output_txt_get_text_file, "w", encoding="utf-8") as f:
            f.write(text_from_get_text)
        print(
            f"[ИНФО] Результат 'тупого' get_text() сохранен в: '{output_txt_get_text_file}' для анализа."
        )


def find_and_inspect(archive_path, keyword):
    """Основная функция: ищет по ключевому слову и предлагает выбор."""
    print(f"Открываю архив: {archive_path}")
    try:
        archive = Archive(archive_path)
    except Exception as e:
        print(f"Ошибка открытия архива: {e}")
        return

    print(f"\nИщу все записи, содержащие в пути '{keyword}'...")
    print("(Это может занять несколько минут)")

    found_entries = []
    total_entries = archive.entry_count

    for i in range(total_entries):
        if i > 0 and i % 100000 == 0:
            print(f"...проверено {i} из {total_entries} записей...")

        entry = archive._get_entry_by_id(i)
        path = entry.path

        if path and keyword in path:
            found_entries.append({"path": path, "title": entry.title})
            print(f"Найдено совпадение: {path}")

    print(f"\nПоиск завершен. Всего найдено совпадений: {len(found_entries)}")

    if not found_entries:
        return

    # Выводим пронумерованный список и просим пользователя сделать выбор
    print("\n--- Найденные совпадения ---")
    for i, entry_data in enumerate(found_entries):
        print(
            f"  [{i + 1}] Path: {entry_data['path']:<70} | Title: {entry_data['title']}"
        )

    while True:
        try:
            choice = input(
                "\nВведите номер записи для инспекции (или 'q' для выхода): "
            )
            if choice.lower() == "q":
                return
            choice_index = int(choice) - 1
            if 0 <= choice_index < len(found_entries):
                chosen_path = found_entries[choice_index]["path"]
                save_inspection_files(archive, chosen_path)
                break
            else:
                print("Ошибка: неверный номер. Попробуйте еще раз.")
        except ValueError:
            print("Ошибка: введите число.")


if __name__ == "__main__":
    find_and_inspect(ZIM_FILE_PATH, TARGET_KEYWORD)
