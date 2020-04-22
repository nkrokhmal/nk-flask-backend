from python:3.6

WORKDIR /app
COPY /app /app
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt
RUN mkdir -p /opt/
RUN chmod -R 777 /opt/
RUN mkdir -p /opt/download
RUN chmod -R 777 /opt/download

EXPOSE 8818

cmd ["python", "run.py"]

