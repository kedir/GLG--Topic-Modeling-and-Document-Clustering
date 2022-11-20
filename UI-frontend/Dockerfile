# app/Dockerfile

FROM python:3.8

EXPOSE 8501

WORKDIR .

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -U cython
RUN pip install --no-cache-dir -U numpy
RUN pip install --no-cache-dir -U pystan

COPY app/requirements.txt .

RUN pip install --no-cache-dir -U -r  requirements.txt

CMD ["streamlit", "run", "main.py", "--host=0.0.0.0", "--port=8501"]