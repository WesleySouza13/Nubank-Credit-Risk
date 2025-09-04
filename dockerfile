FROM python:3.11.9-slim

WORKDIR /app

# copiar apenas requirements primeiro para cache de instalação
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# copiar o restante do código
COPY . .

EXPOSE 10000

# rodar FastAPI com 4 workers
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:10000"]
