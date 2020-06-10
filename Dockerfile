FROM python:3.7-slim

ADD /src/. /

RUN pip3 install numpy

RUN mkdir data

ENTRYPOINT ["python", "./main_job.py"]