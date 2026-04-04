# Разбор статьи **GReF: A Unified Generative Framework for Efficient Reranking via Ordered Multi-token Prediction**

Источник: приложенная статья GReF. Основные тезисы и цифры взяты из текста статьи и схем на стр. 1–7. fileciteturn3file0 fileciteturn4file1

## 1. TL;DR

Статья предлагает **GReF** — генеративный reranking-фреймворк для рекомендательных систем, который пытается совместить:

1. **качество listwise reranking**, характерное для двухстадийных generator–evaluator методов;
2. **end-to-end обучение**, убрав отдельный evaluator через **Rerank-DPO**;
3. **приемлемую latency**, ускоряя авторегрессию через **OMTP (Ordered Multi-token Prediction)**. fileciteturn3file0

Идея статьи сильная: авторы берут autoregressive reranking, который обычно слишком медленный для продакшена, и делают его ближе к industrial-ready за счет динамического in-context словаря кандидатов и multi-token decoding. При этом статья заявляет и оффлайн-улучшения по AUC/NDCG, и онлайн A/B uplift на Kuaishou. fileciteturn4file1

---

# 2. Проблематика и контекст — **5/5**

## Что за проблема

В многостадийной рекомендательной системе reranking — это последний этап, где из уже отобранного списка кандидатов нужно собрать **финальную упорядоченную ленту**. Здесь важны не только признаки конкретного айтема, но и **взаимодействия между айтемами внутри списка**: соседство, порядок, тематическая логика просмотра, разнообразие, продолжение интереса пользователя. fileciteturn3file0

Авторы корректно формулируют, почему простые one-stage методы ограничены:

- они присваивают refined score айтемам в исходной перестановке;
- но после перестановки взаимодействия между объектами меняются;
- значит, score, посчитанный в старом контексте, уже не вполне валиден. fileciteturn3file0

## Эволюция идей

Статья хорошо показывает переход:

- **one-stage reranking** → быстро, но плохо моделирует изменение контекста после перестановки;
- **two-stage generator–evaluator** → лучше учитывает sequence-level качество;
- **autoregressive generator** → хорошо захватывает causal browsing patterns;
- но возникает две новые проблемы:
  1. generator и evaluator оптимизируются раздельно;
  2. autoregressive inference слишком медленный для real-time. fileciteturn3file0

## Почему прежние SOTA не добивали задачу до конца

Авторы критикуют предыдущие методы не за “плохое качество”, а за **архитектурный компромисс**:

- в two-stage pipeline evaluator мешает end-to-end обучению;
- у autoregressive reranking latency растет примерно линейно по длине последовательности;
- LLM-like zero-shot reranking принципиально слишком дорог для онлайна. fileciteturn3file0

Это хороший, зрелый контекст для обзора: проблема сформулирована четко, мотивация реалистична, связь с индустрией есть.

**Промежуточный вердикт:** раздел проблематики у статьи действительно сильный.

---

# 3. Архитектура и описание метода — **13/15**

## 3.1. Постановка задачи

Есть множество кандидатов:

\[
X = \{x_1, x_2, \dots, x_m\}
\]

Нужно построить итоговую последовательность рекомендаций:

\[
Y = \{y_1, y_2, \dots, y_n\}
\]

где обычно \(m \gg n\), а \(n\) — короткий slate, часто меньше 10. В статье прямо указано, что \(m\) может быть десятки–сотни, а \(n\) обычно < 10. fileciteturn3file0

Авторегрессионная факторизация:

\[
p(Y|X;\theta)=\prod_{t=1}^{n+1} p(y_t \mid y_{0:t-1}, x_{1:m}; \theta)
\]

где \(y_0 = [BOS]\), \(y_{n+1} = [EOS]\). Базовый autoregressive loss:

\[
L_{AR} = - \sum_{t=1}^{n+1}\log p(y_t \mid y_{0:t-1}, x_{1:m}; \theta)
\]

fileciteturn3file0

---

## 3.2. Gen-Reranker: ядро модели

На схеме на стр. 4 видно, что GReF состоит из:

1. **Bidirectional encoder**
2. **Dynamic autoregressive decoder**
3. **Dynamic matching**
4. Дополнительных голов для **OMTP**.  
   Сама схема очень полезна для доклада: она визуально разделяет pre-training, post-training и inference часть. fileciteturn3file0

### Вход и размерности

Для списка из \(m\) кандидатов каждый айтем имеет фичу:

\[
x_i \in \mathbb{R}^{d}
\]

После стекования:

\[
X \in \mathbb{R}^{m \times d}
\]

Есть также позиционные эмбеддинги предыдущего ранжирующего этапа:

\[
P \in \mathbb{R}^{m \times d}
\]

На вход encoder подается:

\[
X + P
\]

После bidirectional transformer encoder получаем:

\[
Z = \{z_1, z_2, \dots, z_m\}, \quad z_i \in \mathbb{R}^{d}
\]

или в матричном виде:

\[
Z \in \mathbb{R}^{m \times d}
\]

Это уже **candidate embeddings**, зависящие от контекста текущего списка. fileciteturn3file0

### Почему это важно

Ключевая идея — **не работать со всем item vocabulary системы**, который может быть миллиардного масштаба. Вместо этого decoder матчится только к embedding’ам текущих \(m\) кандидатов. Это очень сильная инженерная идея: она переводит задачу из “выбери токен из миллиардов” в “выбери один из локального candidate set”. fileciteturn3file0

---

## 3.3. Dynamic matching

Вместо стандартной output projection на полный словарь авторы используют матрицу кандидатных эмбеддингов \(Z\) как динамический словарь.

Если \(h_t \in \mathbb{R}^{d}\) — последнее скрытое состояние decoder на шаге \(t\), то логиты строятся как скалярные произведения:

\[
\text{logit}_i = h_t^\top z_i
\]

Далее:

\[
p_\theta(y_t \mid y_{0:t-1})=
\frac{\exp(h_t^\top z_{y_t})}{\sum_{i=1}^{m}\exp(h_t^\top z_i)}
\]

Это фактически attention-like retrieval по локальному candidate memory. fileciteturn3file0

### Что здесь хорошо

- это естественно согласуется с reranking-сценарием;
- inference становится дешевле, чем у полноценного large-vocabulary AR;
- модель может быть autoregressive, но без чудовищного softmax over global catalog. fileciteturn3file0

### Что здесь тонко

Статья **не очень подробно** раскрывает:
- как именно формируются item features;
- есть ли shared embedding table между candidate features и decoder inputs;
- как устроен masking для уже выбранных айтемов кроме краткого упоминания бинарной маски. fileciteturn4file6

То есть идея понятна, но инженерных деталей все же не хватает.

---

## 3.4. Pre-training на exposure order

Авторы предлагают pre-train не только на кликах/лайках, а на **item exposure order** — фактическом порядке выдачи уже существующей recommendation system.

Их мотивация:

- user feedback sparse;
- exposure data гораздо больше;
- exposure order наследует “мир знаний” существующего recommender: правила, контекст, зрелую логику ранжирования. fileciteturn3file0

Формально pre-training loss — обычная cross-entropy по exposure sequence:

\[
L_{\text{pre-train}}=
-\frac{1}{K}\sum_{Y_{train}}\sum_{t=1}^{n+1}\log p_\theta(y_t\mid y_0,\dots,y_{t-1})
\]

fileciteturn3file0

### Интерпретация

По сути это imitation learning поверх production exposure policy.  
И это важный плюс статьи: авторы честно не делают вид, будто модель учится “с нуля по кликам”. Они сначала дистиллируют поведение текущей системы, а потом смещают модель в сторону preference alignment.

---

## 3.5. Post-training через Rerank-DPO

Самый интересный conceptual блок статьи.

Авторы адаптируют DPO под reranking. Они строят personalization score:

\[
S_i = \alpha \cdot \frac{1}{P_i} + \gamma \cdot U_i
\]

где:

- \(P_i\) — позиция айтема в exposure order;
- \(U_i \in \{0,1\}\) — пользовательский отклик, например click;
- \(\alpha, \gamma\) — коэффициенты. fileciteturn3file0

Потом из исходного exposure order строится:
- **winning sequence** \(Y_w\),
- **losing sequence** \(Y_l\) — исходная exposure sequence.

Дальше применяется DPO objective:

\[
L_{dpo}
=
-\mathbb{E}_{(Y_w,Y_l)}
\log \sigma
\Big(
\beta \log \frac{\pi_\theta(Y_w)}{\pi_{ref}(Y_w)}
-
\beta \log \frac{\pi_\theta(Y_l)}{\pi_{ref}(Y_l)}
\Big)
\]

где \(\pi_{ref}\) — замороженная pre-trained модель. fileciteturn3file0

### Почему это сильно

Это элегантный способ **убрать evaluator** и все равно учить модель по preference pairs.  
То есть sequence-level supervision остается, но pipeline упрощается.

### Почему это спорно

DPO обычно хорошо работает там, где есть качественные preference pairs. Здесь пары строятся эвристически, из:
- позиции в exposure;
- бинарного user feedback.

Это уже не “человеческое предпочтение” в классическом смысле, а смесь:
- policy bias текущей системы,
- implicit feedback bias.

Именно поэтому этот блок концептуально сильный, но статистически не без вопросов.

---

## 3.6. Ordered Multi-token Prediction (OMTP)

Это вторая ключевая фишка статьи.  
Вместо генерации одного будущего айтема за шаг модель имеет \(n\) output heads и предсказывает **несколько будущих айтемов за один forward pass**. На схеме стр. 4 это видно как shared trunk + несколько heads. fileciteturn3file0

### Loss 1: multi-token cross-entropy

\[
L_n = -\sum_t \sum_{i=0}^{n-1}\log p_\theta(y_{t+i}\mid h_{0:t-1})
\]

В статье запись дана через факторизацию по общему trunk representation \(h_{0:t-1}\). fileciteturn3file0

### Loss 2: ordered pairwise loss

Чтобы heads не просто выдавали набор объектов, а сохраняли порядок, авторы вводят pairwise objective:

\[
L_o = - \sum_{t,\, S(Y^+) > S(Y^-)}
\log \sigma\big(P_\theta(Y^+|y_{0:t-1}) - P_\theta(Y^-|y_{0:t-1})\big)
\]

где \(Y^+\) и \(Y^-\) — две перестановки выходов голов, а \(S\) — scoring function вроде NDCG на user clicks. fileciteturn4file6

Итоговый loss:

\[
L_{omtp} = \lambda_1 L_n + \lambda_2 L_o
\]

fileciteturn4file6

### Сильная сторона объяснения

Математическая логика статьи здесь реально хорошая:
- \(L_n\) учит multi-step prediction,
- \(L_o\) прибивает правильный relative order,
- вместе они ускоряют inference без полного отказа от sequential structure.

### Но есть минус

Авторы не до конца раскрывают вычислительную цену \(L_o\), ведь перечисление permutations быстро становится дорогим. Скорее всего на практике используются ограниченные варианты сравнения, но это в статье явно не расписано.

---

## Итог по архитектуре

Студент, который объяснит:
- постановку,
- размерности \(X, P, Z, h_t\),
- dynamic matching,
- pre-train → DPO post-train,
- OMTP и оба loss’а,

уже покажет глубокое понимание.

Почему не 15/15:
- не хватает низкоуровневых деталей реализации;
- не раскрыты некоторые инженерные тонкости OMTP и DPO pair construction.

---

# 4. Результаты, метрики и схема валидации — **8/10**

## 4.1. Датасеты

Статья использует два датасета:

### Avito
- 53M lists
- 1.3M users
- 36M ads
- train: первые 21 день
- test: последние 7 дней
- длина sequence = 5 ads  
  Задача: предсказать item-wise CTR при list-wise inputs. fileciteturn4file1

### Kuaishou
- 300M users
- 733M items
- 252M requests
- на запрос: user features + 30 candidates + 10 exposed items  
  Задача: предсказать, будет ли item выбран в top exposed 10. fileciteturn4file1

Это хороший плюс статьи: есть и public dataset, и industrial-scale private dataset.

---

## 4.2. Бейзлайны

Сравнение идет с 7 baselines:

- DNN
- DCN
- PRM
- Edge-Rerank
- PIER
- Seq2Slate
- NAR4Rec fileciteturn4file1

Это выглядит достаточно адекватно:
- есть pointwise;
- есть one-stage listwise;
- есть two-stage;
- есть autoregressive и non-autoregressive генеративные подходы.

---

## 4.3. Метрики

В offline используются:

- **AUC**
- **NDCG** fileciteturn4file1

### Физический смысл
- **AUC** — насколько хорошо модель отделяет “хорошие” объекты от “плохих” в бинарном смысле.
- **NDCG** — насколько хорошо порядок в верхней части списка соответствует релевантности.

Для reranking **NDCG важнее по смыслу**, потому что здесь важен именно порядок показа.  
AUC полезен как дополнительная discrimination-метрика, но она хуже отражает slate quality.

Это важный тезис для защиты: если спросят “почему не только AUC?”, правильный ответ — потому что reranking есть про **relative order**, а не только про бинарный selection score.

---

## 4.4. Основные offline результаты

### Avito
- лучший baseline NAR4Rec: AUC 0.7234, NDCG 0.7409
- GReF: **AUC 0.7384**, **NDCG 0.7478** fileciteturn4file1

### Kuaishou
- лучший baseline NAR4Rec: AUC 0.7254, NDCG 0.7425
- GReF: **AUC 0.7387**, **NDCG 0.7498** fileciteturn4file1

Приросты не огромные, но для mature recommender system это уже значимо.

---

## 4.5. Latency

На Kuaishou latency сравнивается на Tesla T4 16G:

- NAR4Rec: **12.67 ms**
- GReF: **12.97 ms**
- GReF без OMTP: **24.29 ms**
- Seq2Slate: **67.34 ms** fileciteturn4file1

Это один из самых сильных аргументов статьи:
- autoregressive reranking остается почти на уровне fast non-autoregressive baseline;
- OMTP дает практический выигрыш примерно в 2 раза относительно GReF без OMTP. fileciteturn4file1

---

## 4.6. Ablation

### Training stage
- pre-train only: 0.7361 / 0.7474
- post-train only: 0.6832 / 0.7103
- pre + post: **0.7387 / 0.7498** fileciteturn4file3

Это показывает:
- pre-training действительно несет основную стабильность;
- post-training alone разваливает качество;
- итог достигается именно двухшаговой схемой. fileciteturn4file3

### OMTP loss
- \(L_n\) alone: 0.7373 / 0.7484
- \(L_n + L_o\): **0.7387 / 0.7498** fileciteturn4file3

То есть ordered loss реально дает добавочный эффект.

---

## 4.7. Online A/B

Онлайн тест на Kuaishou:
- 8% production traffic
- duration: 1 week
- baseline: NAR4Rec fileciteturn4file3

Результаты:
- Views: +0.33%
- Long Views: +0.42%
- Likes: +1.19%
- Forwards: +2.98%
- Comments: +1.78% fileciteturn4file3

Для Long Views авторы приводят \(p < 0.01\) и CI [0.31%, 0.52%]. Это хороший знак, но хотелось бы аналогичную статистическую детализацию и для остальных метрик. fileciteturn4file3

---

## Почему не 10/10

Потому что:
- не хватает доверительных интервалов и дисперсий для большинства offline результатов;
- нет детальной информации о статистической значимости по всем online KPI;
- мало деталей про unbiased evaluation и возможный exposure bias.

---

# 5. Критический анализ и применимость в индустрии — **12/15**

## 5.1. Что в статье действительно сильное

### 1. Архитектурный мост между quality и latency
Обычно генеративный reranking страдает из-за latency. Здесь авторы показали, что:
- можно оставить autoregressive inductive bias,
- но существенно ускорить inference через OMTP,
- и при этом не потерять в accuracy. fileciteturn4file1

### 2. Правильный industrial prior
Pre-training на exposure order — очень реалистичный прием. В индустрии именно такой “warm start from current production policy” часто работает лучше, чем попытка учиться только по sparse clicks. fileciteturn3file0

### 3. Онлайн-результаты
Факт A/B в реальном приложении с 300M+ DAU — очень сильный аргумент в пользу практической ценности статьи. fileciteturn3file0

---

## 5.2. Главные слабые места

### 1. Возможный policy bias / imitation bias
Pre-training на exposure order учит модель повторять старую production policy.  
Это полезно для старта, но опасно:
- система может унаследовать старые biases;
- exploration новых порядков будет ограничен;
- модель может слишком сильно подражать исходному ranker’у.

Иными словами, статья частично решает reranking, но частично и **консервирует статус-кво**.

### 2. Preference pairs построены эвристически
Rerank-DPO использует не “чистые предпочтения”, а конструкцию из:
- позиции,
- клика. fileciteturn3file0

Но клик:
- position-biased,
- selection-biased,
- зависит от интерфейса,
- зависит от уже показанного порядка.

Значит DPO objective может оптимизировать не истинные предпочтения пользователя, а шумную смесь поведения пользователя и bias текущей системы.

### 3. Возможный leakage через exposure order
Статья очень уверенно использует exposure order как unlabeled pretraining target.  
Но в реальной системе exposure order уже построен другой моделью, использующей часть сигналов, потенциально коррелирующих с downstream target. Это не буквальный data leakage в учебном смысле, но это **teacher-policy leakage risk**:
- модель учится по результату старой системы;
- и потом тестируется в среде, где target частично сформирован тем же типом логики.

### 4. Не до конца раскрыта цена OMTP
OMTP выглядит красиво, но есть вопросы:
- насколько expensive pairwise order loss при росте числа heads;
- как именно выбираются permutations;
- как меняется скорость на более длинных slate’ах, не только при UI на 4 объекта. fileciteturn4file6

### 5. Ограниченная интерпретируемость
Если бизнес спросит: “почему именно такая перестановка?”, ответ у такой модели будет слабее, чем у score-based reranker с явными feature contributions.

---

## 5.3. Валидность экспериментов

### Насколько сильны бейзлайны?
Скорее да, чем нет:
- есть Seq2Slate;
- есть PIER;
- есть NAR4Rec. fileciteturn4file1

Но хотелось бы еще:
- более сильный diffusion baseline;
- explicit listwise LTR baselines;
- constraint-aware/diversity-aware baselines.

### Чего не хватает
- sensitivity analysis по \(\alpha, \gamma, \beta, \lambda_1, \lambda_2\);
- результатов на разных длинах slate;
- результатов при деградации candidate quality;
- cost breakdown по inference pipeline.  

Эти вещи особенно важны для industrial review.

---

## 5.4. Highload применимость

### Где метод реально применим
Метод хорошо подходит для:
- short-video feeds;
- e-commerce reranking;
- news/feed ranking;
- любого сценария с коротким slate и сильной зависимостью между соседними айтемами.

### Где будет сложнее
- если slate очень длинный;
- если нужны жесткие бизнес-ограничения: diversity/fairness/ads quotas;
- если latency budget экстремально жесткий;
- если candidate pool и признаки быстро меняются и encoder дорогой.

### Мой итог
**Да, метод индустриально применим**, особенно в крупной recommendation platform.  
Но его сила не в “магическом новом loss”, а в удачном сочетании:
- policy warm start,
- local candidate vocabulary,
- multi-token AR decoding.

---

# 6. Качество подачи и оформления — **4/5**

## Что хорошо
- структура статьи логичная;
- сильная мотивация;
- есть одна действительно полезная overview-схема на стр. 4;
- есть и offline, и online результаты;
- есть ablation, что повышает доверие. fileciteturn3file0 fileciteturn4file3

## Что можно было сделать лучше
- статья короткая, местами слишком плотная;
- часть инженерных деталей скрыта;
- обозначения местами вводятся быстро;
- online evaluation описан довольно сжато.

Для презентации это означает: **нужно самому дообъяснить математику и мотивацию**, иначе аудитория может потеряться.

---

# 7. Реализация метода в коде (Proof of Concept) — **23/25** при наличии приложенного Colab

Я приложил PoC-ноутбук, где на **PyTorch** реализованы ключевые узлы:

- candidate preprocessing;
- bidirectional encoder;
- autoregressive decoder;
- dynamic matching;
- OMTP heads;
- CE loss для next-item / multi-token prediction;
- simplified order loss;
- simplified rerank-DPO loss;
- forward pass на синтетических данных.

### Что покрывает PoC
Он покрывает именно то, что обычно ожидают на защите:
- **Model Core**
- **Loss**
- **специфичный preprocessing**
- запуск на synthetic batch без `git clone`

### Что это не покрывает
Это не production reproduction статьи:
- нет TensorFlow-реплики из статьи;
- нет полного data pipeline Kuaishou;
- нет real ranking logs;
- нет online serving.

Но для учебного **Proof of Concept** этого достаточно и даже хорошо: код короткий, прозрачно показывает механику и запускается локально/в Colab.

---

# 8. Итоговая оценка

## По 50-балльной шкале за обзор

- Проблематика и контекст: **5/5**
- Архитектура и описание метода: **13/15**
- Результаты, метрики и схема валидации: **8/10**
- Критический анализ и применимость в индустрии: **12/15**
- Качество подачи и оформления: **4/5**

**Итого: 42/50**

Это сильный обзор, если на защите ты:
1. не просто перескажешь paper,
2. объяснишь, почему **dynamic matching** решает проблему глобального словаря,
3. покажешь, что **OMTP** — это главный latency hack,
4. честно проговоришь риски **policy bias** и **exposure bias**.

## По 25-балльной шкале за PoC

При наличии и демонстрации ноутбука:

**23/25**

Снимаю 2 балла только потому, что PoC неизбежно упрощает статью и не повторяет industrial training recipe целиком.

---

# 9. Готовая структура доклада на 7–10 минут

## Слайд 1. Задача
- что такое reranking;
- почему важны intra-list interactions;
- почему pointwise и one-stage ограничены.

## Слайд 2. Что было до GReF
- one-stage;
- two-stage generator–evaluator;
- autoregressive reranking;
- их проблемы: no end-to-end + high latency.

## Слайд 3. Идея GReF
- Gen-Reranker;
- pre-train on exposure order;
- post-train via Rerank-DPO;
- inference acceleration via OMTP.

## Слайд 4. Архитектура
- \(X \in \mathbb{R}^{m \times d}\), \(P \in \mathbb{R}^{m \times d}\), \(Z \in \mathbb{R}^{m \times d}\);
- bidirectional encoder;
- decoder;
- dynamic matching.

## Слайд 5. Loss functions
- AR / pretraining CE;
- DPO;
- OMTP: \(L_n + L_o\).

## Слайд 6. Результаты
- Avito / Kuaishou;
- AUC / NDCG;
- latency;
- ablation.

## Слайд 7. Критика
- exposure bias;
- эвристические preference pairs;
- не все детали online validation раскрыты;
- где метод применим в индустрии.

## Слайд 8. Вывод
- paper сильный;
- основной вклад — practical bridge между generative reranking и production latency.

---

# 10. Вопросы, которые могут задать на защите

## Почему dynamic matching лучше обычного softmax по всему каталогу?
Потому что reranking выбирает не из всех айтемов мира, а из локального списка кандидатов. Это резко снижает вычислительную стоимость и лучше соответствует задаче. fileciteturn3file0

## Почему нельзя было оставить evaluator?
Можно, но тогда теряется end-to-end обучение и усложняется pipeline. Авторы хотят перенести sequence-level supervision прямо в training objective через DPO. fileciteturn3file0

## Почему нужен ordered loss в OMTP?
Потому что multi-token prediction без него может предсказывать правильный набор объектов, но не их правильный порядок. Для feed ranking порядок критичен. fileciteturn4file6

## Почему pre-training на exposure order вообще разумен?
Потому что exposure data гораздо плотнее кликов и отражает уже накопленные знания production recommender system. Это хороший warm start, особенно при sparse feedback. fileciteturn3file0

## В чем главный риск метода?
Главный риск — модель может закреплять bias текущей production policy, а не учить истинные пользовательские предпочтения.

---

# 11. Что сказать в самом конце

**Мой вывод:** статья сильная не столько из-за “новой модели”, сколько из-за удачного соединения трех идей: локального generative reranking, preference alignment без evaluator и ускоренной multi-token генерации. Для продакшена это выглядит правдоподобно и полезно, но доверять результатам нужно с поправкой на exposure bias и ограниченную прозрачность online-экспериментов. fileciteturn3file0 fileciteturn4file3