FROM python:3.7-slim

RUN apt-get update -y
RUN apt-get install libgomp1

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY src /src
COPY app.py /app.py
COPY models /models

WORKDIR .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]