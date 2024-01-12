FROM python:3.11

RUN mkdir /dockerized_model

WORKDIR /dockerized_model

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /dockerized_model/app
#COPY ./data /dockerized_model/data
COPY ./configs.py /dockerized_model/configs.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]