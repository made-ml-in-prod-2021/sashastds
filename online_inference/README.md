# Online inference for classification model using FastAPI and Docker

## Требования

- Python 3.7
- Docker Desktop >= 2.0.0.3

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

### Запросы из браузера
```
http://localhost:8000/docs
```
и далее следовать примерам

### Тестовые запросы из терминала
```bash
python make_requests.py
```
\+ можно заменить в скрипте PATH_TO_DATA к своему .csv

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
pip install pytest
pytest -v tests_app.py
```

## Checklist

0) ветку назовите homework2, положите код в папку online_inference

\+

1) Оберните inference вашей модели в rest сервис(вы можете использовать как FastAPI, так и flask, другие желательно не использовать, дабы не плодить излишнего разнообразия для проверяющих), должен быть endpoint /predict (3 балла)

\+

2) Напишите тест для /predict  (3 балла) (https://fastapi.tiangolo.com/tutorial/testing/, https://flask.palletsprojects.com/en/1.1.x/testing/)

\+

3) Напишите скрипт, который будет делать запросы к вашему сервису -- 2 балла

\+

4) Сделайте валидацию входных данных (например, порядок колонок не совпадает с трейном, типы не те и пр, в рамках вашей фантазии)  (вы можете сохранить вместе с моделью доп информацию, о структуре входных данных, если это нужно) -- 3 доп балла
https://fastapi.tiangolo.com/tutorial/handling-errors/ -- возращайте 400, в случае, если валидация не пройдена

\+

5) Напишите dockerfile, соберите на его основе образ и запустите локально контейнер(docker build, docker run), внутри контейнера должен запускать сервис, написанный в предущем пункте, закоммитьте его, напишите в readme корректную команду сборки (4 балл)

\+

6) Оптимизируйте размер docker image (3 доп балла) (опишите в readme.md что вы предприняли для сокращения размера и каких результатов удалось добиться)  -- https://docs.docker.com/develop/develop-images/dockerfile_best-practices/

\+

7) опубликуйте образ в https://hub.docker.com/, используя docker push (вам потребуется зарегистрироваться) (2 балла)

\+

8) напишите в readme корректные команды docker pull/run, которые должны привести к тому, что локально поднимется на inference ваша модель (1 балл)
Убедитесь, что вы можете протыкать его скриптом из пункта 3

\+

9) проведите самооценку -- 1 доп балл

Вроде как сделаны все пункты, поэтому bottom up:

1 + 1 + 2 + 3 + 4 + 3 + 2 + 3 + 3 = 22

\+ 

10) создайте пулл-реквест и поставьте label -- hw2

\+
