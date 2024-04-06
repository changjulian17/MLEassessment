FROM python:3.11

WORKDIR /SectionB

COPY ./src ./src

RUN pip install -r "./src/requirements.txt"

COPY ./data ./data

COPY ./model ./model

CMD ["python", "./src/train_model.py"]