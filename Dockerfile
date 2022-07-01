FROM python:3.10
ENV APP_DIR=/home/app/

RUN mkdir -p $APP_DIR
WORKDIR $APP_DIR

COPY requirements.txt requirements.txt
COPY manage.py manage.py
COPY db.sqlite3 db.sqlite3

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update
RUN apt-get install -y ffmpeg lilypond fluidsynth
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN chmod 755 "./entrypoint.sh"

COPY ./config/ ./config/
COPY ./intervals/ ./intervals/
COPY ./inversions/ ./inversions/
COPY ./modules/ ./modules/
COPY ./musak/ ./musak/
COPY ./rhythm/ ./rhythm/
COPY ./shared/ ./shared/
COPY ./start/ ./start/
COPY ./static/ ./static/
COPY ./templates/ ./templates/

CMD ["sh", "-c", "python manage.py runserver 0.0.0.0:$MUSAK_PORT"]
