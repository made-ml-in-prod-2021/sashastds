Model Training and Inference:
-----------

Training Examples:
-----------
Config0
`python app.py train=train_config_v0`

Config1
`python app.py train=train_config_v1`

Inference Examples:
-----------
`python app.py inference=inference_config_v0`

Tests Run:
-----------
`pytest -v tests.py`

Примеры конфигов лежат в директории configs, в ней поддиректории для типов запуска, используется Hydra.

Чек-лист с разбалловкой разобран ниже:


- (-2) Назовите ветку homework1 (1 балл)
\+
- (-1) положите код в папку ml_project
\+
- (0) В описании к пулл реквесту описаны основные "архитектурные" и тактические решения, которые сделаны в вашей работе. В общем, описание что именно вы сделали и для чего, чтобы вашим ревьюерам было легче понять ваш код. (2 балла)
*Реализовано автораспознавание фичей по типам, в т.ч категориальные из интовых, кастомные трансформеры для категориальных фичей на основе OneHot и LabelEncoder (0..N-1) (конкретно для данного датасета бессмысленно, поскольку все кат. фичи и так уже в таком виде, но в целом полностью рабочий вариант), два типа моделей - RandomForest и LightGBM, подбор оптимального порога отсечения максимизацией f1-score для возможности бинарного прогноза (это уже управляется конфигом на этапe inference).*

- (1) Выполнение EDA, закоммитьте ноутбук в папку с ноутбуками (2 баллов). Вы так же можете построить в ноутбуке прототип(если это вписывается в ваш стиль работы). Можете использовать не ноутбук, а скрипт, который сгенерит отчет, закоммитьте и скрипт и отчет (за это +1 балл)
*EDA в директории notebooks, отчет в форме jsonа с метриками, графиками permutation feature importance и confusion matrix генерируется при обучении модели на тестовом сете, на инференсе также возможно, если указано в конфиге и в датафрейме есть истинный таргет.*
[rendered notebook](https://nbviewer.jupyter.org/github/made-ml-in-prod-2021/sashastds/blob/homework1/ml_project/notebooks/EDA.ipynb)

- (2) Проект имеет модульную структуру(не все в одном файле =) ) (2 баллов)
\+

- (3) использованы логгеры (2 балла)
\+

- (4) написаны тесты на отдельные модули и на прогон всего пайплайна(3 баллов)
\+

- (5) Для тестов генерируются синтетические данные, приближенные к реальным (3 баллов)
\+

- (6) Обучение модели конфигурируется с помощью конфигов в json или yaml, закоммитьте как минимум 2 корректные конфигурации, с помощью которых можно обучить модель (разные модели, стратегии split, preprocessing) (3 балла)
\+

- (7) Используются датаклассы для сущностей из конфига, а не голые dict (3 балла) 
\+

- (8) Используйте кастомный трансформер(написанный своими руками) и протестируйте его(3 балла)
\+
- (9) Обучите модель, запишите в readme как это предлагается (3 балла)
\+, см. выше  в примерах

- (10) Напишите функцию predict, которая примет на вход артефакт/ы от обучения, тестовую выборку(без меток) и запишет предикт, напишите в readme как это сделать (3 балла)  
\+, см. выше  в примерах

- (11) Используется hydra  (https://hydra.cc/docs/intro/) (3 балла - доп баллы)
\+

- (12) Настроен CI(прогон тестов, линтера) на основе github actions (3 балла - доп баллы (будем проходить дальше в курсе, но если есть желание поразбираться - welcome)
*Пока что нет, долго возился с гидрой, запушу после софт дедлайна.*

- (13) Проведите самооценку, опишите, в какое колво баллов по вашему мнению стоит оценить вашу работу и почему (1 балл доп баллы) 
38 - 3(CI)