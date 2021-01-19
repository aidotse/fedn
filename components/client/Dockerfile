FROM nvcr.io/nvidia/tensorflow:20.09-tf1-py3
RUN pip install numpy==1.18.5 h5py==2.10.0 keras==2.1.4 pandas tensorflow  sklearn pillow matplotlib
RUN mkdir -p /app/client
RUN mkdir /app/certs
COPY fedn /app/fedn
RUN pip install -e /app/fedn
WORKDIR /app
