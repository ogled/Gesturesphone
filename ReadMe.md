<p align="center">
  <source media="(prefers-color-scheme: dark)" srcset="Assets/Logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="Assets/Logo_light.svg">
  <img src="Assets/Logo_light.svg" alt="Logo">
</p>

--------------------------------------------------------------------------------

Этот репозиторий содержит код модели классификации жестов рук на PyTorch, которая с точностью около **68%** распознаёт жесты русского жестового языка, а также код для запуска и некоторый дополнительный функционал.

## Установка

Для базовой установки выполните следующие команды:

```bash
git clone https://github.com/ogled/AutoDriveBot.git
cd Gesturesphone
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
cd Frontend
npm install
npm run build
cd ..
cd Backend
python server.py
````

Для доступа к **GigaChat** создайте файл `config.json` по шаблону `config.example.json` в директории `Backend/Config` и вставьте в поле `TOKEN` свой API-ключ.

## Обучение собственной модели

Для обучения собственной модели требуется:

1. Установить датасет [Slovo](https://github.com/hukenovs/slovo) версии `Slovo` или `360p`.
2. Запустить файл `Train/Datasets/createDataSet.py`, передав аргумент `--input` с путём к папке, в которой хранится датасет Slovo. Пример команды:

````bash
python Train/Datasets/createDataSet.py --input "C:\SlovoDS"
````

> [!TIP]
> Вы можете отредактировать файл `AllowedGestures.csv`, добавив жесты, которые требуются.

3. Запустить файл `Train/startTrain.py` и дождаться окончания обучения:

````bash
python Train/startTrain.py --out model.pth
````
4. Протестировать полученную модель, выполнив стандартный запуск программы.