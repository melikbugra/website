FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

COPY bot.py .

# Pre-create log file so the volume mount works without Docker making it a directory
RUN touch bot.log

RUN useradd -r -u 1001 botuser && chown -R botuser /app
USER botuser

CMD ["python", "-u", "bot.py"]
