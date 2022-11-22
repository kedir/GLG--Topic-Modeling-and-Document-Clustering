# app/Dockerfile

FROM python:3.7

EXPOSE 8501

WORKDIR /src/app

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

COPY app/ .

ENTRYPOINT ["streamlit", "run"]

CMD ["main.py", "--browser.gatherUsageStats=false", "--global.disableWatchdogWarning=true"]