FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=true

RUN apt-get update && apt-get install -y ffmpeg curl

RUN pip install decord google-cloud-datastore decord numpy

# Install youtube-dl
RUN curl -L https://yt-dl.org/downloads/latest/youtube-dl -o /usr/local/bin/youtube-dl &&\
    chmod a+rx /usr/local/bin/youtube-dl

# Update to the path to your trained model
COPY trained/20210430-0031/1 /model

COPY label_video.py .

CMD ["python", "./label_video.py"]
