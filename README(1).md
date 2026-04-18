# ИАР: Классификация документов клиентов банка

Итоговая аттестационная работа по курсу «Разработка нейросетей на Python с нуля до проекта (вторая ступень)».

**Тема:** Автоматическая классификация типов документов клиентов банка с использованием свёрточных нейронных сетей и переноса обучения (Transfer Learning).

**Авторы:**
- Зангиев Мовсар Абдул-Саламович
- Зангиева Медина Абдул-Саламовна
- Зангиев Муслим Абдул-Саламович

**Руководитель:** Дмитрий Кузьмин

## Структура проекта

```
iar_project/
├── 01_eda.ipynb            # Разведочный анализ + скачивание датасета
├── 02_models.ipynb         # Обучение 3 моделей + сравнение + демо
├── check_env.py            # Проверка GPU и версий TensorFlow/CUDA
├── requirements.txt        # Зависимости
├── README.md               # Этот файл
├── report.pdf              # Итоговая аттестационная работа (PDF)
├── presentation.pptx       # Презентация для защиты
├── midv500_data/           # Датасет (скачивается автоматически, ~5-7 ГБ)
└── outputs/                # Графики, метрики, сохранённые модели
```

## Окружение

**Протестировано на:**
- Fedora 43, ядро 6.19+
- NVIDIA RTX 3060 (12 ГБ VRAM) + проприетарный драйвер из RPM Fusion nonfree
- Python 3.10 в conda-среде
- TensorFlow 2.15 (CUDA 12.2 + cuDNN 8.9 идут внутри pip-пакета `tensorflow[and-cuda]`)

## Установка (Fedora + conda)

### 1. Создаём отдельное conda-окружение

```bash
# создаём окружение на Python 3.10 (TF 2.15 не работает на 3.12+)
conda create -n iar python=3.10 -y
conda activate iar
```

### 2. Ставим TensorFlow с поддержкой GPU

TF 2.15+ официально поддерживает установку CUDA runtime через pip —
это удобнее, чем системный CUDA из RPM Fusion, потому что версии совпадают
с TF без ручной настройки путей.

```bash
pip install "tensorflow[and-cuda]==2.15.1"
```

Пакет `tensorflow[and-cuda]` сам подтянет нужные `nvidia-cudnn-cu12`,
`nvidia-cublas-cu12` и другие CUDA-зависимости как wheels (бинарные пакеты).

### 3. Ставим остальные зависимости

```bash
pip install -r requirements.txt
```

### 4. Проверяем, что GPU виден TensorFlow

```bash
python check_env.py
```

Ожидаемый вывод:

```
Python:     3.10.x
TensorFlow: 2.15.x
CUDA доступна: True
GPU-устройства: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
Имя GPU: NVIDIA GeForce RTX 3060
VRAM:   12288 МБ
```

Если GPU не виден — самые частые причины:

1. **NVIDIA драйвер не установлен** — проверить `nvidia-smi`. Если команды нет:
   ```bash
   sudo dnf install akmod-nvidia xorg-x11-drv-nvidia-cuda
   sudo reboot
   ```

2. **cuDNN не подхватился** — проверить `LD_LIBRARY_PATH`:
   ```bash
   echo $LD_LIBRARY_PATH
   # должен содержать пути типа .../nvidia/cudnn/lib
   ```

3. **Версия Python несовместима** — должен быть 3.9, 3.10 или 3.11.

## Запуск

### Шаг 1: Разведочный анализ данных (EDA)

```bash
conda activate iar
jupyter notebook 01_eda.ipynb
```

Выполнить ячейки по порядку. Первый запуск скачивает датасет MIDV-500
(~5-7 ГБ, 20-60 минут в зависимости от скорости интернета).

**Примечание для пользователей из РФ:** FTP-сервер Smart Engines может быть
недоступен из некоторых провайдеров. Альтернативы:
- включить V2RayN / AmneziaWG перед запуском;
- скачать вручную с Kaggle-зеркала и распаковать в `midv500_data/midv500/`:
  https://www.kaggle.com/datasets/kontheeboonmeeprakob/midv500

**Результат этапа:** `outputs/metadata.csv` со всеми путями к изображениям и
метками классов + графики баланса классов, условий съёмки, разрешений.

### Шаг 2: Обучение моделей

```bash
jupyter notebook 02_models.ipynb
```

Обучаются три модели:

1. **CNN с нуля** — базовая архитектура (5 свёрточных слоёв), базовая линия
   (baseline).
2. **MobileNetV2 + Transfer Learning** — лёгкая модель (~3.5 млн параметров),
   сценарий «модель для мобильного приложения банка».
3. **ResNet50 + Transfer Learning** — тяжёлая модель (~25 млн параметров),
   сценарий «модель для серверной обработки».

**Время обучения на RTX 3060:** ~30-60 минут на каждую модель.

**Результат этапа:**

- Сохранённые модели в `outputs/models/` (формат Keras `.keras`).
- Таблица сравнения моделей в `outputs/comparison.csv`.
- Графики кривых обучения (loss, accuracy по эпохам).
- Матрицы ошибок (confusion matrix) для каждой модели.
- Демонстрация работы на «картинке из интернета» (требование руководителя).

## Воспроизводимость

Для полной воспроизводимости результатов зафиксирован `random_state = 42` во
всех местах, где используется случайность:

- разделение выборки на train/valid/test (`sklearn.train_test_split`);
- инициализация весов моделей;
- перемешивание данных в DataLoader'ах.

## Используемые технологии

- **Python 3.10** — язык программирования.
- **TensorFlow / Keras 2.15** — фреймворк для глубокого обучения (deep learning).
- **scikit-learn** — стратифицированное разделение выборки и метрики.
- **Pandas / NumPy** — обработка табличных метаданных.
- **Matplotlib / Seaborn** — визуализация.
- **Pillow (PIL)** — чтение изображений.
- **midv500** (fcakyon) — Python-пакет для автоматической загрузки датасета.

## Источники

1. Arlazarov V.V., Bulatov K., Chernov T., Arlazarov V.L. MIDV-500: A Dataset
   for Identity Documents Analysis and Recognition on Mobile Devices in Video
   Stream. Computer Optics, 2019, Vol. 43(5), P. 818-824.
2. Howard A. et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks.
   CVPR 2018.
3. He K., Zhang X., Ren S., Sun J. Deep Residual Learning for Image Recognition.
   CVPR 2016.
4. Шолле Ф. Глубокое обучение на Python. — СПб.: Питер, 2018. — 400 с.
5. Géron A. Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow.
   O'Reilly Media, 2019.
