FROM python:3.11.9

WORKDIR /modelo-aplicationScore

COPY requirements.txt . 

RUN pip install --no-cache-dir -r requirements.txt 

COPY . . 

EXPOSE 10000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "10000"]
