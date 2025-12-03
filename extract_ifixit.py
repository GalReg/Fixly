import os
import re
from libzim.reader import Archive
from bs4 import BeautifulSoup, NavigableString, Tag

# --- НАСТРОЙКИ ---
ZIM_FILE_PATH = "/Users/rrgalieva/Documents/fixly_bot/ifixit_ru_all_2025-06.zim" 
OUTPUT_DIR = "/Users/rrgalieva/Documents/fixly_bot/ifixit_data" # Новая папка для чистых данных

# --- ФИНАЛЬНЫЙ СКРИПТ ---

def sanitize_filename(name):
    """Очищает имя файла от недопустимых символов."""
    name = re.sub(r'[\s/]+', '_', name) # Заменяем пробелы и слэши на подчеркивания
    return re.sub(r'[\\*?:"<>|]', "", name)

def extract_smart_text(soup: BeautifulSoup) -> str:
    """
    "Умная" функция извлечения текста, основанная на нашем исследовании.
    Она ищет контент в разных блоках и собирает его воедино.
    """
    # Этот контейнер, кажется, есть на всех страницах с контентом
    main_container = soup.find('div', class_='wrapper articleContainer')
    if not main_container:
        # Если даже его нет, то это, скорее всего, не контентная страница
        return None

    text_parts = []

    # 1. Извлекаем заголовок
    title_tag = main_container.find('h1', class_='guide-intro-title')
    if title_tag:
        text_parts.append(f"Заголовок: {title_tag.get_text(strip=True)}")

    # 2. Извлекаем введение
    intro_div = main_container.find('div', class_='introduction-container')
    if intro_div:
        intro_text = intro_div.get_text(separator='\n', strip=True)
        if intro_text:
            text_parts.append(f"\n--- ВВЕДЕНИЕ ---\n{intro_text}")

    # 3. Извлекаем инструменты и запчасти (очень полезно для RAG)
    tools_div = main_container.find('div', class_='item-list-tools')
    if tools_div:
        tools_text = tools_div.get_text(separator='\n', strip=True)
        if tools_text:
            text_parts.append(f"\n--- ИНСТРУМЕНТЫ ---\n{tools_text}")
            
    parts_div = main_container.find('div', class_='item-list-parts')
    if parts_div:
        parts_text = parts_div.get_text(separator='\n', strip=True)
        if parts_text:
            text_parts.append(f"\n--- ЗАПЧАСТИ ---\n{parts_text}")

    # 4. Извлекаем шаги руководства (самая важная часть!)
    steps_container = main_container.find('ol', class_='steps-container')
    if steps_container:
        text_parts.append("\n--- ШАГИ ---")
        steps = steps_container.find_all('li', class_='step-wrapper')
        for step in steps:
            step_title_tag = step.find('strong', class_='stepValue')
            step_content_div = step.find('div', class_='step-lines-container')
            
            step_text = ""
            if step_title_tag:
                step_text += f"\n{step_title_tag.get_text(strip=True)}:"
            if step_content_div:
                # Убираем лишние пустые строки для чистоты
                step_content_text = step_content_div.get_text(separator='\n', strip=True)
                step_text += f"\n{step_content_text}"
            
            if step_text:
                text_parts.append(step_text)
    
    # 5. Запасной вариант для Wiki-страниц (где структура другая)
    if not steps_container:
        wiki_div = main_container.find('div', id='wikitext')
        if wiki_div:
            text_parts.append("\n--- ИНФОРМАЦИЯ ИЗ WIKI ---")
            text_parts.append(wiki_div.get_text(separator='\n', strip=True))

    if not text_parts:
        return None

    return "\n".join(text_parts)


def main():
    print(f"Открываю ZIM-файл: {ZIM_FILE_PATH}")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Создана директория для вывода: {OUTPUT_DIR}")

    try:
        archive = Archive(ZIM_FILE_PATH)
    except Exception as e:
        print(f"Ошибка при открытии ZIM-файла: {e}")
        return

    total_entries = archive.entry_count
    print(f"Найдено {total_entries} записей. Начинаю извлечение...")

    processed_count = 0
    for i in range(total_entries):
        if i > 0 and i % 10000 == 0:
            print(f"Обработано {i} из {total_entries} записей. Найдено и сохранено: {processed_count}")
        
        entry = archive._get_entry_by_id(i)
        
        if entry.is_redirect:
            continue
        
        path = entry.path

        # НАШ НОВЫЙ, РАСШИРЕННЫЙ ФИЛЬТР
        if not path or not (path.startswith("Guide/") or path.startswith("Wiki/") or path.startswith("Device/")):
            continue
        
        try:
            item = entry.get_item()
            # Используем правильное декодирование для memoryview
            html_content = str(item.content, 'utf-8')
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Используем нашу новую умную функцию
            smart_text = extract_smart_text(soup)

            if not smart_text:
                # Если умная функция ничего не вернула, пропускаем этот файл
                continue

            # --- Сохранение результата ---
            safe_title = sanitize_filename(entry.title)
            # Добавим префикс, чтобы было понятно, откуда файл
            prefix = path.split('/')[0]
            output_filename = os.path.join(OUTPUT_DIR, f"{prefix}_{safe_title}_{i}.txt")
            
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(f"Source Path: {path}\n")
                f.write(f"Original Title: {entry.title}\n")
                f.write("-" * 20 + "\n\n")
                f.write(smart_text)
                
            processed_count += 1

        except Exception as e:
            # print(f"Не удалось обработать '{path}': {e}")
            continue

    print("-" * 20)
    print("Извлечение завершено!")
    print(f"Всего сохранено файлов: {processed_count}")

if __name__ == "__main__":
    main()
