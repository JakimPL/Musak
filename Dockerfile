FROM python:3.10
ENV APP_DIR=/home/app/

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR
COPY . $APP_DIR

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBUG 0

RUN apt-get update
RUN apt-get install -y ffmpeg lilypond fluidsynth
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8000
#ENTRYPOINT ["sh", "-c", "apt-get update; pt-get install -y ffmpeg lilypond fluidsynth; pip install --upgrade pip; pip install -r requirements.txt; python manage.py runserver 0.0.0.0:8000"]

CMD python manage.py runserver 0.0.0.0:$PORT
