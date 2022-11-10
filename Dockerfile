# 
FROM python:3.7

# Setting Home Directory for containers
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

# 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80", "--debug"]
