FROM python:3.9-slim-bullseye

WORKDIR /

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
EXPOSE $PORT

CMD ["python", "server.py"]     