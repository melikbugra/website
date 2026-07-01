FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

COPY bot.py pricedb.py momentum.py .

CMD ["python", "-u", "bot.py"]
