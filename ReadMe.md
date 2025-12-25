# Проект Gesturesphone
Добро пожаловать в репозиторий проекта Gesturesphone!

Проект направлен на улучшение коммуникации для людей с нарушениями слуха.

## Описание проекта

Gesturesphone — это нейросетевая система, способная распознавать движения рук и интерпретировать их в команды в реальном времени.  

Основные возможности:  
- Распознавание жестов с высокой точностью.  
- Работа в реальном времени на автономных устройствах.  
- Поддержка визуального отображения результатов.  
- Интеграция с внешними системами через API.

## Технологии

- **Python 3.10**  
- **OpenCV** 
- **Mediapipe** 
- **Web-интерфейс**
- **Raspberry Pi / ПК** 

## Установка

1. Клонируйте репозиторий:

```bash
git clone https://github.com/ogled/AutoDriveBot.git
cd Gesturesphone
````

2. Создайте виртуальное окружение и установите зависимости:

```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

3. Запустите сборку frontend:

```bash
cd Frontend
npm install
npm run build
cd ..
```

4. Запуск

```bash
cd Backend
python -m uvicorn server:app --port 8080
```