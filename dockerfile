FROM python:3.12-slim-bookworm

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 3000
CMD ["python", "app.py"]
