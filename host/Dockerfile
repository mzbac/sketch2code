FROM ubuntu:17.10

ENV LANG C.UTF-8
RUN apt update && apt install -y python python3.5 python-pip virtualenv
RUN apt install -y vim
RUN pip install setuptools pip --upgrade --force-reinstall
RUN apt install -y python3-pip
RUN pip3 install torch
RUN pip3 install torchvision
RUN pip3 install opencv-python
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev
COPY . /app
WORKDIR /app

RUN mkdir /app/uploads
RUN pip3 install Flask

EXPOSE 5000

ENTRYPOINT ["python3.6", "server.py"]