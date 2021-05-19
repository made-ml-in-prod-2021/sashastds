# Online Inference for classification model: LightGBM, FastAPI and Docker

## Требования

- Python 3.7
- docker >= 2.0.0.3

## Как запустить

### Build from source

```bash
cd online_inference
docker build -t raisedtoward/online_inference:v1 .
```

### Pull from DockerHub

```bash
docker pull raisedtoward/online_inference:v1
```

## Как использовать

### Запуск контейнера на инференс

```bash
docker run --rm -p 8000:8000 raisedtoward/online_inference:v1
```

### Из браузера
```
http://localhost:8000/docs
```
и далее следовать примерам

### Запросы из терминала
```bash
python make_requests.py
```

## Какие вообще есть эндпойнты 
Полезные:
- /docs
- /predict

Не очень полезные:
- /status
- /

## Что насчёт оптимизации образа

- в `requirements.txt` только релеватные пакеты
- в докер образ из кода идёт только то, что необходимо для инференса и src не дублирует первый проект для обучения, лишних данных и тестов нет
- python:3.7-slim вместо python:3.7 как базовый слой

	
## Запуск тестов

```bash
pip install -q pytest
pytest app_tests.py -v 
```