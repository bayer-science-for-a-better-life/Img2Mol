FROM tensorflow/tensorflow:1.10.1-gpu-py3
EXPOSE 1111
ENV PATH="/root/miniconda3/bin:$PATH"
ARG PATH="/root/miniconda3/bin:$PATH"
SHELL ["/bin/bash", "--login", "-c"]


RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*
# Install miniconda
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh
RUN conda init bash

# Install CDDD
COPY cddd_environment.yml ./
COPY default_model /root/miniconda3/envs/cddd/lib/python3.6/site-packages/cddd/data/default_model/
RUN conda env create -f cddd_environment.yml

COPY run_cddd_server.py ./
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cddd", "python", "run_cddd_server.py"]
