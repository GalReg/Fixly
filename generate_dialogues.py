#!/usr/bin/env python3
"""
Скрипт для генерации диалогов между ботом и пользователем.
Имитирует работу бота из bot_w_langchain_rag.py, но без Telegram API.
Использует YandexGPT для обоих сторон диалога.
"""

import os
import logging
import asyncio
import csv
import random
from typing import List, Any, Optional
from operator import itemgetter
from dotenv import load_dotenv

# --- Импорты LangChain ---
from langchain_community.llms import YandexGPT
from langchain_community.embeddings import YandexGPTEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# --- НАСТРОЙКИ ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
DB_FAISS_PATH = "vectorstore/db_faiss"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('langchain.chains').setLevel(logging.WARNING)

# --- ПРОМПТЫ ДЛЯ НЕЙРОСЕТИ ---

# 1. НОВЫЙ ПРОМПТ ДЛЯ "ЖИВОГО" ДИАЛОГА
SYSTEM_PROMPT_DIALOGUE = """
Ты — AI-ассистент "Fixly", опытный и вдумчивый диагност бытовой техники. Твоя цель — не просто задавать вопросы, а вести пользователя к решению, попутно объясняя свою логику.

### Твои принципы общения:
1.  **Будь как детектив.** Каждый твой вопрос должен быть шагом в расследовании. Задавай 1-2 умных, связанных вопроса в одной реплике, чтобы сузить круг "подозреваемых" неисправностей.
2.  **Думай вслух.** После ответа пользователя кратко поделись своей гипотезой. Например: "Понял. Если вода есть, но в бачок не идет, возможно, проблема в заливном клапане. Давайте это проверим...". Это вызывает доверие.
3.  **Не повторяйся.** Внимательно следи за историей диалога, чтобы не задавать вопросы, на которые уже есть ответ.
4.  **Соблюдай лимит.** У тебя есть не более 4-х реплик для сбора информации. После получения ответа на твою 4-ю реплику, ты должен перейти к смете.
5.  **Сигнал к завершению.** Когда информации достаточно (или лимит в 4 реплики исчерпан), закончи свой финальный диагностический вывод СТРОГОЙ фразой-сигналом: "Отлично, я готовлю смету...". Не добавляй ничего после нее.

### Пример идеального диалога:
Пользователь: Холодильник не морозит, а только слабо холодит.
Ассистент: Здравствуйте. Похоже на проблему с системой No Frost или утечку фреона. Давайте уточним: компрессор работает постоянно или отключается? И есть ли на задней стенке внутри камеры наледь?
Пользователь: Компрессор работает почти без остановки. Наледи не видно.
Ассистент: Спасибо. Постоянная работа компрессора и отсутствие холода часто указывают на утечку хладагента. Чтобы это проверить, потрогайте конденсатор (решетку сзади холодильника). Он должен быть горячим. Он горячий или еле теплый?
Пользователь: Еле теплый.
Ассистент: Это подтверждает гипотезу об утечке. При нормальной работе он был бы горячим. В таком случае, скорее всего, потребуется поиск места утечки и перезаправка системы. Отлично, я готовлю смету...
"""

# 2. ПРОМПТ ДЛЯ СМЕТЫ С ПРАВИЛЬНОЙ РОЛЬЮ RAG
SYSTEM_PROMPT_ESTIMATE = """
## ТВОЯ ЗАДАЧА
Твоя задача — выступить в роли эксперта-диагноста и, основываясь на всей доступной информации, составить финальное заключение в формате JSON.

## АЛГОРИТМ РАБОТЫ
1.  **Основа — твои знания и диалог.** В первую очередь используй свои обширные знания о поломках и логику, выстроенную в ходе переписки с клиентом. Это твой главный источник истины.
2.  **Инструкции — для сверки.** Приложенные СПРАВОЧНЫЕ ИНСТРУКЦИИ используй как "второе мнение" или технический справочник. Сверяйся с ними, чтобы подтвердить свои догадки, уточнить правильные названия запчастей и узлов. Не копируй из них информацию слепо.
3.  **Сформируй JSON.** Синтезируй все данные (свои знания + диалог + сверка по инструкциям) и выдай ответ СТРОГО в формате JSON, без лишних слов.

## СТРОГАЯ СХЕМА JSON ДЛЯ ОТВЕТА
{{
  "estimated_cause": "Здесь твой финальный, развернутый диагноз. Например: 'Постоянная работа компрессора при отсутствии охлаждения и теплый конденсатор с высокой вероятностью указывают на утечку хладагента в системе'.",
  "required_parts": [
    {{
      "name": "Название запчасти или услуги. Например: 'Устранение утечки и заправка хладагентом R600a'.",
      "quantity": 1
    }}
  ]
}}

### СПРАВОЧНЫЕ ИНСТРУКЦИИ ИЗ БАЗЫ ЗНАНИЙ (для сверки):
{context}

### ИСТОРИЯ ПЕРЕПИСКИ С КЛИЕНТОМ:
{history}
"""

# Промпт для симуляции пользователя (YandexGPT как пользователь)
USER_SIMULATION_PROMPT = """
Ты - пользователь сервиса по ремонту бытовой техники Fixly. Ты задал вопрос: "{initial_question}"

История диалога до этого момента:
{dialogue_history}

Бот только что спросил: "{bot_question}"

Ты должен ответить как реальный пользователь, который не разбирается в ремонте и дает правдоподобный ответ. Не будь слишком технически подкованным, но дай полезную информацию. Если у тебя нет конкретной информации, скажи что-то вроде "я не знаю" или "не уверен".

Ответь кратко, как обычный пользователь.
"""

# --- PYDANTIC МОДЕЛИ (без изменений) ---
class Part(BaseModel):
    name: str = Field(description="Название запчасти, материала или услуги")
    quantity: int = Field(description="Количество")

class Estimate(BaseModel):
    estimated_cause: str = Field(description="Развернутое описание наиболее вероятной причины поломки")
    required_parts: List[Part] = Field(description="Список необходимых запчастей, материалов или услуг")

# --- ИНИЦИАЛИЗАЦИЯ LANGCHAIN ---

llm = YandexGPT(
    api_key=YANDEX_API_KEY, folder_id=YANDEX_FOLDER_ID,
    model_name="yandexgpt", temperature=0.5,
)

try:
    embeddings_model = YandexGPTEmbeddings(api_key=YANDEX_API_KEY, folder_id=YANDEX_FOLDER_ID)
    db = FAISS.load_local(DB_FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    logging.info("Векторная база данных и ретривер успешно загружены.")
except Exception as e:
    logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить RAG компоненты: {e}")
    retriever = None

# --- Цепочки LangChain ---

def format_docs(docs: List[Document]) -> str:
    if not docs:
        logging.warning("RAG: В базе знаний не найдено релевантных документов.")
        return "Инструкции не найдены."
    log_output = [f"'{doc.metadata.get('source', 'N/A').split('/')[-1]}'" for doc in docs]
    logging.info(f"RAG: Найдено {len(docs)} релевантных документов: {', '.join(log_output)}")
    return "\n\n".join([f"### Документ: {doc.metadata.get('source', 'Без заголовка').split('/')[-1]}\n{doc.page_content}" for doc in docs])

user_memory = {}

parser = JsonOutputParser(pydantic_object=Estimate)
estimate_prompt_template = PromptTemplate(template=SYSTEM_PROMPT_ESTIMATE, input_variables=["history", "context"])

def log_llm_output(x: Any) -> Any:
    logging.info(f"LLM RAW JSON OUTPUT:\n{x}")
    return x

rag_chain = (
    {"context": itemgetter("question") | retriever | format_docs, "history": itemgetter("history")}
    | estimate_prompt_template
    | llm
    | RunnableLambda(log_llm_output)
    | parser
)

async def simulate_user_response(initial_question: str, dialogue_history: str, bot_question: str) -> str:
    """Симуляция ответа пользователя с помощью YandexGPT"""
    user_llm = YandexGPT(
        api_key=YANDEX_API_KEY, folder_id=YANDEX_FOLDER_ID,
        model_name="yandexgpt", temperature=0.7,  # Более высокая температура для разнообразия
    )

    prompt = USER_SIMULATION_PROMPT.format(
        initial_question=initial_question,
        dialogue_history=dialogue_history,
        bot_question=bot_question
    )

    response = await asyncio.to_thread(user_llm.invoke, prompt)
    return response.strip()

async def generate_single_dialogue(initial_question: str) -> dict:
    """Генерация одного полного диалога"""
    dialogue_history = []
    full_dialogue = []
    user_id = f"simulated_{random.randint(1000, 9999)}"

    # Очищаем память для пользователя
    if user_id in user_memory:
        del user_memory[user_id]

    # Создаем цепочку для диалога
    user_memory[user_id] = []
    conversation_chain = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_DIALOGUE),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    current_input = initial_question
    max_turns = 5  # Максимум 5 ходов диалога

    for turn in range(max_turns):
        # Добавляем пользовательский ввод в историю
        dialogue_history.append(HumanMessage(content=current_input))
        full_dialogue.append(f"Пользователь: {current_input}")

        # Получаем ответ бота
        conversation_chain_with_memory = conversation_chain.partial(history=dialogue_history)
        bot_response = await asyncio.to_thread(llm.invoke, conversation_chain_with_memory.format(input=current_input))

        bot_response = bot_response.strip()
        full_dialogue.append(f"Бот: {bot_response}")

        # Проверяем на триггер сметы
        if "готовлю смету" in bot_response.lower():
            # Формируем смету через RAG
            try:
                history_messages = dialogue_history
                def get_msg_info(msg):
                    if hasattr(msg, 'type') and hasattr(msg, 'content'):
                        return f"{msg.type}: {msg.content}"
                    elif isinstance(msg, dict):
                        return f"{msg.get('type', 'unknown')}: {msg.get('content', '')}"
                    else:
                        return f"unknown: {str(msg)}"
                history_str = "\n".join([get_msg_info(msg) for msg in history_messages])
                def get_msg_content(msg):
                    if hasattr(msg, 'content'):
                        return msg.content
                    elif isinstance(msg, dict):
                        return msg.get('content', '')
                    else:
                        return str(msg)
                user_query = " ".join([get_msg_content(msg) for msg in history_messages if (hasattr(msg, 'type') and msg.type == 'human') or (isinstance(msg, dict) and msg.get('type') == 'human')])

                # Получаем релевантные документы отдельно
                docs = await asyncio.to_thread(retriever.invoke, user_query)
                relevant_chunks = format_docs(docs)

                estimate_dict = await asyncio.to_thread(rag_chain.invoke, {"question": user_query, "history": history_str})

                # Форматируем финальный ответ
                formatted_reply = "*Предварительный список работ готов!*\n\n"
                formatted_reply += f"*Вероятная причина:* {estimate_dict.get('estimated_cause', 'Не удалось определить')}\n\n"
                parts = estimate_dict.get('required_parts', [])
                if parts:
                    formatted_reply += "*Необходимые запчасти:*\n"
                    for part in parts:
                        formatted_reply += f"- {part.get('name', 'N/A')} ({part.get('quantity', 0)} шт.)\n"
                else:
                    formatted_reply += "*Необходимые запчасти:* Запчасти, скорее всего, не потребуются.\n"
                formatted_reply += "\n_Это предварительная оценка. Точный список работ определит мастер._"

                full_dialogue.append(f"Бот (финальный): {formatted_reply}")

                return {
                    "start_dialogue": initial_question,
                    "all_dialogue": "\n".join(full_dialogue),
                    "final_answer": formatted_reply,
                    "chunks": relevant_chunks
                }

            except Exception as e:
                logging.error(f"Ошибка при генерации сметы: {e}")
                return {
                    "start_dialogue": initial_question,
                    "all_dialogue": "\n".join(full_dialogue),
                    "final_answer": "Ошибка при формировании сметы",
                    "chunks": ""
                }

        # Если нет триггера, симулируем ответ пользователя
        from langchain_core.messages import AIMessage
        dialogue_history.append(AIMessage(content=bot_response))
        current_input = await simulate_user_response(initial_question, "\n".join(full_dialogue), bot_response)

    # Если диалог не завершился сметой за max_turns, завершаем его
    return {
        "start_dialogue": initial_question,
        "all_dialogue": "\n".join(full_dialogue),
        "final_answer": "Диалог завершен без формирования сметы",
        "chunks": ""
    }

async def generate_dialogues_parallel(questions: List[str], num_parallel: int = 5) -> None:
    """Генерация диалогов параллельно"""
    csv_file = "generated_dialogues.csv"

    # Создаем/очищаем CSV файл
    with open(csv_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["start_dialogue", "all_dialogue", "final_answer", "chunks"])

    semaphore = asyncio.Semaphore(num_parallel)

    async def process_question(question: str):
        async with semaphore:
            dialogue = await generate_single_dialogue(question)
            # Добавляем в CSV
            with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    dialogue["start_dialogue"],
                    dialogue["all_dialogue"],
                    dialogue["final_answer"],
                    dialogue.get("chunks", "")
                ])
            logging.info(f"Диалог завершен для вопроса: {question[:50]}...")

    # Обрабатываем все вопросы параллельно
    tasks = [process_question(question) for question in questions]
    await asyncio.gather(*tasks)

def load_questions(filename: str) -> List[str]:
    """Загружаем вопросы из файла"""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

async def main():
    """Основная функция"""
    logging.info("Запуск генерации диалогов...")

    # Загружаем все вопросы
    questions = load_questions("eval_data/dialogue_start.txt")
    logging.info(f"Загружено {len(questions)} вопросов")

    # Генерируем диалоги для всех вопросов параллельно (по 10 одновременно)
    await generate_dialogues_parallel(questions, num_parallel=10)

    logging.info("Генерация диалогов завершена!")

if __name__ == "__main__":
    asyncio.run(main())
