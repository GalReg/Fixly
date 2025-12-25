# -*- coding: utf-8 -*-
# bot_w_langchain_rag.py

import logging
import os
import asyncio
from typing import List, Any
from operator import itemgetter

from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from dotenv import load_dotenv

from langchain_community.llms import YandexGPT
try:
    from langchain_community.embeddings.yandex import YandexGPTEmbeddings
except ImportError:
    from langchain_community.embeddings import YandexGPTEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

# --- НАСТРОЙКИ ---
load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
YANDEX_FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")
DB_FAISS_PATH = "vectorstore/db_faiss"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    model_name="yandexgpt", temperature=0.5, # Температура повышена для "живости"
)

try:
    # Initialize embeddings based on what's available
    if YandexGPTEmbeddings.__name__ == "OpenAIEmbeddings":
        # Fallback to OpenAI embeddings
        embeddings_model = YandexGPTEmbeddings()
    else:
        # Use YandexGPT embeddings
        embeddings_model = YandexGPTEmbeddings(api_key=YANDEX_API_KEY, folder_id=YANDEX_FOLDER_ID)
    
    db = FAISS.load_local(DB_FAISS_PATH, embeddings_model, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={'k': 3})
    logging.info("Векторная база данных и ретривер успешно загружены.")
except Exception as e:
    logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить RAG компоненты: {e}")
    retriever = None

# --- Цепочки LangChain ---

def format_docs(docs: List[Document]) -> str:
    # ... (код без изменений)
    if not docs:
        logging.warning("RAG: В базе знаний не найдено релевантных документов.")
        return "Инструкции не найдены."
    log_output = [f"'{doc.metadata.get('source', 'N/A').split('/')[-1]}'" for doc in docs]
    logging.info(f"RAG: Найдено {len(docs)} релевантных документов: {', '.join(log_output)}")
    return "\n\n".join([f"### Документ: {doc.metadata.get('source', 'Без заголовка').split('/')[-1]}\n{doc.page_content}" for doc in docs])

user_memory = {}
def get_conversation_chain(user_id: int) -> ConversationChain:
    # ... (код без изменений)
    if user_id not in user_memory:
        user_memory[user_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
    return ConversationChain(
        llm=llm, prompt=ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT_DIALOGUE),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ]),
        memory=user_memory[user_id]
    )

parser = JsonOutputParser(pydantic_object=Estimate)
estimate_prompt_template = PromptTemplate(template=SYSTEM_PROMPT_ESTIMATE, input_variables=["history", "context"])

def log_llm_output(x: Any) -> Any:
    # ... (код без изменений)
    logging.info(f"LLM RAW JSON OUTPUT:\n{x}")
    return x

rag_chain = (
    # ... (код без изменений)
    {"context": itemgetter("question") | retriever | format_docs, "history": itemgetter("history")}
    | estimate_prompt_template
    | llm
    | RunnableLambda(log_llm_output)
    | parser
)

# --- ЛОГИКА ТЕЛЕГРАМ БОТА ---
bot = Bot(token=TELEGRAM_BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    user_id = message.from_user.id
    if user_id in user_memory: del user_memory[user_id]
    await message.reply("Здравствуйте! Я AI-ассистент Fixly. Опишите вашу проблему.")
    if not retriever: await message.answer("⚠️ *Внимание:* База знаний (RAG) не загружена.", parse_mode="Markdown")

@dp.message_handler(content_types=['text', 'photo'])
async def handle_message(message: types.Message):
    user_id = message.from_user.id
    user_text = message.text or message.caption or ""
    if not user_text: return await message.reply("Пожалуйста, добавьте описание к вашему фото.")
    if message.photo: user_text = f"[Фото]. Описание: {user_text}"

    # Сообщение "Анализирую..." УБРАНО по вашему желанию
    
    dialogue_chain = get_conversation_chain(user_id)
    gpt_response: str = await asyncio.to_thread(dialogue_chain.predict, input=user_text)

    gpt_response = gpt_response.strip()

    # НОВАЯ, ЧИСТАЯ ЛОГИКА ПЕРЕХОДА К СМЕТЕ
    if "готовлю смету" in gpt_response.lower():
        # 1. Мы не отправляем ответ модели, если в нем есть триггер.
        #    Вместо этого отправляем свои сообщения.
        await message.answer("Ищу подходящие инструкции и формирую список работ...")
        
        try:
            # 2. Дальнейшая логика RAG и формирования сметы
            history_messages = dialogue_chain.memory.chat_memory.messages
            history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in history_messages])
            user_query = " ".join([msg.content for msg in history_messages if isinstance(msg, HumanMessage)])
            logging.info(f"RAG: Сформирован чистый запрос для поиска: '{user_query}'")
            
            estimate_dict = await asyncio.to_thread(rag_chain.invoke, {"question": user_query, "history": history_str})

            # Форматирование ответа (без изменений)
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

            await message.answer(formatted_reply, parse_mode='Markdown')
            
            # 3. Очищаем историю ПОСЛЕ успешной генерации
            if user_id in user_memory: del user_memory[user_id]

        except Exception as e:
            logging.error(f"Ошибка при генерации сметы с RAG: {e}", exc_info=True)
            await message.answer("Произошла ошибка при формировании списка работ. Похоже, вернулись неструктурированные данные. Попробуйте еще раз с другими формулировками.")
    else:
        # Если триггера нет, просто отправляем ответ бота в диалоге
        await message.answer(gpt_response)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

