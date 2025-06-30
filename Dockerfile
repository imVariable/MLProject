FROM python:3.11-slim-buster

WORKDIR /app 

COPY . /app

RUN apt install -y && apt install awscli -y 

RUN pip install -r requirements.txt 

CMD ["python", "app.py"]
