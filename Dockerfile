from python:3.6

WORKDIR /app
COPY /app /app
COPY requirements.txt /app
RUN pip install -r /app/requirements.txt

EXPOSE 6666

cmd ["python", "run.py"]

