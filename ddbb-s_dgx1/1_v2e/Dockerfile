# Usar imagen base con soporte CUDA y Ubuntu 20.04
FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Evitar preguntas interactivas durante la instalación
ENV DEBIAN_FRONTEND=noninteractive

# instalat tmux y vim
# instalar requirements en v2e

# Actualizar el sistema e instalar dependencias esenciales
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    bzip2 \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    libgl1-mesa-glx \
    libjpeg62 \
    ffmpeg \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Instalar Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Configurar Conda en el PATH
ENV PATH=/opt/conda/bin:$PATH

# Crear carpetas necesarias para el proyecto
WORKDIR /app
RUN mkdir -p /app/input /app/output /app/tmp

# Copiar los contenidos actuales al contenedor
ADD . /app

# Crear el entorno Conda y evitar 'conda activate'
RUN /opt/conda/bin/conda create -n v2e python=3.10 -y

# Instalar dependencias usando el entorno creado
COPY requirements.txt /app/requirements.txt

# Configurar el entorno por defecto en bash
RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate v2e" >> ~/.bashrc

#RUN echo "pip install -r /app/requirements.txt"

# Comando por defecto
CMD ["/bin/bash"]
