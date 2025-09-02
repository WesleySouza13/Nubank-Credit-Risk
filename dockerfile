# syntax=docker/dockerfile:1

FROM python:3.11.9

WORKDIR /modelo-aplicationScore

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt 

COPY . . 

EXPOSE 10000

CMD ["python", "-m", "uvicorn", "app.main:app", "--reload", "--port", "10000"]