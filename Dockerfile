FROM continuumio/miniconda3:4.8.2

WORKDIR /app

RUN conda config --add channels conda-forge \
    && conda config --set channel_priority strict \
    && conda install -y cartopy flask gunicorn xarray matplotlib pyproj mercantile numpy scipy xesmf==0.3.0 esmpy==7.1.0 rasterio shapely h5py \
    && rm -rf /opt/conda/pkgs/*


#ARG PIP_EXTRA_INDEX_URL
#COPY requirements.yaml .
#RUN conda config --add channels conda-forge \
#    && conda config --set channel_priority strict \
#    && conda env update -f requirements.yaml \
#    && rm -rf /opt/conda/pkgs/*
RUN echo '#!/bin/bash\n\
    set -e\n\
    source /root/.bashrc\n\
    exec python "$@"' > /usr/local/bin/condapython \
    && chmod +x /usr/local/bin/condapython

COPY app .
COPY gunicorn.conf.py .

EXPOSE 8000
CMD ["gunicorn", "--config", "gunicorn.conf.py", "app.app:app"]
