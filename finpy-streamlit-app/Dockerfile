FROM python:3.10.1-slim

WORKDIR /app

# COPY requirements-finpy-streamlit-app.txt requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "Hello.py"]