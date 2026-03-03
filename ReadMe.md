<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="Assets/Logo_dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="Assets/Logo_light.svg">
    <img src="Assets/Logo_light.svg" alt="Logo">
  </picture>
</p>

--------------------------------------------------------------------------------

Этот репозиторий содержит:
- backend/runtime для распознавания языка жестов в реальном времени;
- обучение модели на PyTorch;
- ONNX runtime для более компактного desktop-дистрибутива.

## Быстрый запуск runtime (Windows/CPU, ONNX)

```bash
git clone https://github.com/ogled/AutoDriveBot.git
cd Gesturesphone
python -m venv venv
venv\Scripts\activate
pip install -r requirements.runtime.txt
cd Frontend
npm install
npm run build
cd ..
cd Backend
python server.py
```

`Backend/backend_app/core/picam.py` использует ONNX backend по умолчанию.  
Для dev fallback можно включить pytorch backend:

```bash
set GESTURE_RUNTIME_BACKEND=torch
```

## Разделенные зависимости

- `requirements.runtime.txt` — runtime (без train-only библиотек).
- `requirements.train.txt` — полный стек обучения и экспорта.
- `requirements.txt` — совместимый полный набор (`-r requirements.train.txt`).

## Обучение собственной модели

1. Установить датасет [Slovo](https://github.com/hukenovs/slovo) версии `Slovo` или `360p`.
2. Подготовить датасет:

```bash
python Train/Datasets/createDataSet.py --input "C:\SlovoDS"
```
> [!TIP]
> Вы можете отредактировать файл `AllowedGestures.csv`, добавив жесты, которые требуются.

3. Обучить:

```bash
pip install -r requirements.train.txt
python Train/startTrain.py --out model.pth
```

4. Экспортировать в ONNX:

```bash
python Train/export_to_onnx.py --checkpoint model.pth
```

## Сборка Windows (one-folder)

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_runtime.ps1
```

Скрипт:
- проверяет наличие `Train/model.onnx` и `Train/model.runtime.json`;
- собирает desktop runtime в `dist\GesturesphoneRuntime`.

## GigaChat

Для доступа к **GigaChat** создайте файл `config.json` по шаблону `Backend/Config/config.example.json` и вставьте в поле `TOKEN` свой API-ключ.