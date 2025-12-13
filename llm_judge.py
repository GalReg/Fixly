#!/usr/bin/env python3
"""
Скрипт для оценки качества диалогов с помощью LLM-as-a-Judge.
Обрабатывает generated_dialogues.csv, оценивает каждый диалог по 11 критериям,
выводит промежуточные метрики и сохраняет финальный отчет.
"""

import os
import logging
import asyncio
import csv
import json
import statistics
from typing import List, Dict, Any
from tqdm.asyncio import tqdm as async_tqdm
from dotenv import load_dotenv
from langchain_community.llms import YandexGPT

# --- НАСТРОЙКИ ---
load_dotenv()

YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
INPUT_CSV = "generated_dialogues.csv"
OUTPUT_CSV = "evaluated_dialogues.csv"
REPORT_FILE = "evaluation_report.txt"
PROMPT_JUDGE_FILE = "eval_data/prompt_judge.txt"
NUM_PARALLEL = 5  # Количество параллельных запросов

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- ЗАГРУЗКА ПРОМПТА ---
def load_judge_prompt() -> str:
    """Загружает промпт для LLM-as-a-Judge"""
    with open(PROMPT_JUDGE_FILE, 'r', encoding='utf-8') as f:
        return f.read()

JUDGE_PROMPT_TEMPLATE = load_judge_prompt()

# --- КРИТЕРИИ ОЦЕНКИ ---
CRITERIA = [
    "diagnostic_accuracy",
    "use_of_dialogue_information",
    "logical_consistency",
    "estimate_completeness",
    "parts_and_works_adequacy",
    "safety",
    "clarity",
    "actionability",
    "uncertainty_handling",
    "business_logic_alignment",
    "contextual_relevancy"
]

# --- ИНИЦИАЛИЗАЦИЯ LLM ---
llm = YandexGPT(
    api_key=YANDEX_API_KEY,
    folder_id=YANDEX_FOLDER_ID,
    model_name="yandexgpt",
    temperature=0.3,  # Низкая температура для более консистентной оценки
)

# --- ФУНКЦИИ ДЛЯ ОЦЕНКИ ---

async def evaluate_dialogue(row: Dict[str, str], row_index: int) -> Dict[str, Any]:
    """Оценивает один диалог с помощью LLM-as-a-Judge"""
    
    # Формируем запрос для судьи
    dialogue = row['all_dialogue']
    model_answer = row['final_answer']
    context_chunks = row.get('chunks', '')
    
    # Формируем промпт
    evaluation_prompt = f"""
{JUDGE_PROMPT_TEMPLATE}

---

### ВХОДНЫЕ ДАННЫЕ ДЛЯ ОЦЕНКИ

**DIALOG:**
{dialogue}

**MODEL_ANSWER:**
{model_answer}

**CONTEXT_CHUNKS:**
{context_chunks if context_chunks else "Контекст не предоставлен."}

---

Оцени ответ по всем 11 критериям и верни результат СТРОГО в формате JSON.
"""
    
    try:
        # Отправляем запрос к LLM
        response = await asyncio.to_thread(llm.invoke, evaluation_prompt)
        
        # Парсим JSON ответ
        # Ищем JSON блок в ответе
        response_text = response.strip()
        
        # Пытаемся найти JSON между ```json и ``` или просто JSON
        if '```json' in response_text:
            start_idx = response_text.find('```json') + 7
            end_idx = response_text.find('```', start_idx)
            json_text = response_text[start_idx:end_idx].strip()
        elif '```' in response_text:
            start_idx = response_text.find('```') + 3
            end_idx = response_text.find('```', start_idx)
            json_text = response_text[start_idx:end_idx].strip()
        else:
            json_text = response_text
        
        evaluation = json.loads(json_text)
        
        # Добавляем метаданные
        result = {
            'row_index': row_index,
            'start_dialogue': row['start_dialogue'],
            'evaluation': evaluation
        }
        
        return result
        
    except json.JSONDecodeError as e:
        logging.error(f"Ошибка парсинга JSON для строки {row_index}: {e}")
        logging.error(f"Ответ LLM: {response[:500]}...")
        return {
            'row_index': row_index,
            'start_dialogue': row['start_dialogue'],
            'evaluation': None,
            'error': f"JSON parse error: {str(e)}"
        }
    except Exception as e:
        logging.error(f"Ошибка при оценке строки {row_index}: {e}")
        return {
            'row_index': row_index,
            'start_dialogue': row['start_dialogue'],
            'evaluation': None,
            'error': str(e)
        }

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Вычисляет средние и медианные значения для каждого критерия"""
    metrics = {}
    
    for criterion in CRITERIA:
        scores = []
        for result in results:
            if result.get('evaluation') and criterion in result['evaluation']:
                score = result['evaluation'][criterion].get('score')
                if score is not None:
                    scores.append(score)
        
        if scores:
            metrics[criterion] = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'min': min(scores),
                'max': max(scores),
                'count': len(scores)
            }
        else:
            metrics[criterion] = {
                'mean': 0,
                'median': 0,
                'min': 0,
                'max': 0,
                'count': 0
            }
    
    return metrics

def print_metrics(metrics: Dict[str, Dict[str, float]], prefix: str = ""):
    """Выводит метрики на экран"""
    print(f"\n{prefix}{'='*60}")
    print(f"{prefix}МЕТРИКИ ПО КРИТЕРИЯМ:")
    print(f"{prefix}{'='*60}")
    
    for criterion, values in metrics.items():
        print(f"{prefix}{criterion:35s} | Mean: {values['mean']:5.2f} | Median: {values['median']:5.2f} | Count: {values['count']:4d}")
    
    # Общая средняя оценка
    all_means = [v['mean'] for v in metrics.values() if v['count'] > 0]
    if all_means:
        overall_mean = statistics.mean(all_means)
        print(f"{prefix}{'='*60}")
        print(f"{prefix}{'ОБЩАЯ СРЕДНЯЯ ОЦЕНКА':35s} | {overall_mean:5.2f}")
        print(f"{prefix}{'='*60}\n")

async def process_dialogues():
    """Основная функция обработки диалогов"""
    
    # Загружаем CSV
    logging.info(f"Загрузка диалогов из {INPUT_CSV}...")
    dialogues = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            dialogues.append((idx, row))
    
    logging.info(f"Загружено {len(dialogues)} диалогов для оценки")
    
    # Создаем семафор для ограничения параллелизма
    semaphore = asyncio.Semaphore(NUM_PARALLEL)
    
    # Список для хранения результатов
    results = []
    
    async def process_with_semaphore(idx: int, row: Dict[str, str]):
        async with semaphore:
            result = await evaluate_dialogue(row, idx)
            results.append(result)
            
            # Выводим промежуточные метрики каждые 10 диалогов
            if len(results) % 10 == 0:
                metrics = calculate_metrics(results)
                print_metrics(metrics, prefix="[ПРОМЕЖУТОЧНО] ")
            
            return result
    
    # Создаем задачи для всех диалогов
    tasks = [process_with_semaphore(idx, row) for idx, row in dialogues]
    
    # Запускаем обработку с прогресс-баром
    logging.info("Начинаем оценку диалогов...")
    for task in async_tqdm.as_completed(tasks, total=len(tasks), desc="Оценка диалогов"):
        await task
    
    logging.info(f"Оценка завершена. Обработано {len(results)} диалогов")
    
    # Вычисляем финальные метрики
    final_metrics = calculate_metrics(results)
    print_metrics(final_metrics, prefix="[ФИНАЛЬНЫЕ МЕТРИКИ] ")
    
    # Сохраняем результаты в новый CSV
    logging.info(f"Сохранение результатов в {OUTPUT_CSV}...")
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['row_index', 'start_dialogue'] + CRITERIA + ['error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for result in results:
            row_data = {
                'row_index': result['row_index'],
                'start_dialogue': result['start_dialogue'],
                'error': result.get('error', '')
            }
            
            if result.get('evaluation'):
                for criterion in CRITERIA:
                    if criterion in result['evaluation']:
                        score = result['evaluation'][criterion].get('score', 0)
                        row_data[criterion] = score
                    else:
                        row_data[criterion] = 0
            else:
                for criterion in CRITERIA:
                    row_data[criterion] = 0
            
            writer.writerow(row_data)
    
    logging.info(f"Результаты сохранены в {OUTPUT_CSV}")
    
    # Сохраняем отчет
    logging.info(f"Создание отчета в {REPORT_FILE}...")
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("ОТЧЕТ ПО ОЦЕНКЕ КАЧЕСТВА ДИАЛОГОВ\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Всего оценено диалогов: {len(results)}\n")
        f.write(f"Успешно обработано: {sum(1 for r in results if r.get('evaluation'))}\n")
        f.write(f"Ошибок: {sum(1 for r in results if r.get('error'))}\n\n")
        
        f.write("="*80 + "\n")
        f.write("МЕТРИКИ ПО КРИТЕРИЯМ\n")
        f.write("="*80 + "\n\n")
        
        for criterion, values in final_metrics.items():
            f.write(f"{criterion:35s}\n")
            f.write(f"  Среднее:  {values['mean']:6.2f}\n")
            f.write(f"  Медиана:  {values['median']:6.2f}\n")
            f.write(f"  Минимум:  {values['min']:6.2f}\n")
            f.write(f"  Максимум: {values['max']:6.2f}\n")
            f.write(f"  Оценок:   {values['count']:6d}\n\n")
        
        all_means = [v['mean'] for v in final_metrics.values() if v['count'] > 0]
        if all_means:
            overall_mean = statistics.mean(all_means)
            f.write("="*80 + "\n")
            f.write(f"ОБЩАЯ СРЕДНЯЯ ОЦЕНКА: {overall_mean:.2f} / 10\n")
            f.write("="*80 + "\n")
    
    logging.info(f"Отчет сохранен в {REPORT_FILE}")
    logging.info("Готово!")

if __name__ == "__main__":
    asyncio.run(process_dialogues())

