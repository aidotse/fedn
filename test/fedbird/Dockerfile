FROM nvcr.io/nvidia/tensorflow:20.09-tf1-py3
RUN pip install numpy==1.18.5 h5py==2.10.0 keras==2.1.4 pandas sklearn pillow matplotlib mean_average_precision

RUN mkdir -p /app && \
    mkdir -p /app/client && \
    mkdir -p /app/certs

RUN mkdir -p /app/fedn

COPY fedn /app/fedn
COPY fedn-network.yaml /app/

RUN pip install -e /app/fedn

WORKDIR /app

