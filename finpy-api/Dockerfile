FROM python:3.10.1-slim

WORKDIR /app

RUN apt-get update

# 'ffmpeg', 'libsm6', and 'libxext6' are required for OpenCV.
# 'default-jre' i.e Java is used by H2O.
RUN apt-get install \
    # 'default-jre'\
    'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

# COPY requirements-finpy-api.txt requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

# CMD ["python", "main.py"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" , "--reload"]