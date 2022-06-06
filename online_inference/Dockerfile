FROM python:3.9-alpine
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt --no-cache-dir
CMD uvicorn main:app --reload --host 0.0.0.0 --port 8000 && python src/webapp/make_requests.py
