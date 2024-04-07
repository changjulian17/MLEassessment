FROM python:3.11

WORKDIR /MLEassessment

COPY ./src ./SectionB/src

RUN pip install -r "./SectionB/src/requirements.txt"

COPY ./data ./SectionB/data

ENV PYTHONPATH=/MLEassessment:$PYTHONPATH

CMD ["python", "SectionB/src/utils/main.py"]

