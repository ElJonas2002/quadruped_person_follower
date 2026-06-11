# Base con CUDA 11.8 + cuDNN 8 sobre Ubuntu 20.04 (Focal).
# Es el stack alineado con Python 3.8, que es el que ROS Noetic requiere.
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video
ENV LANG=en_US.UTF-8

# ─── Configuración base e instalación de ROS Noetic ───
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales curl gnupg2 lsb-release ca-certificates && \
    locale-gen en_US.UTF-8 && \
    # Llave y repositorio de ROS (método moderno con signed-by)
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros/ubuntu focal main" \
        > /etc/apt/sources.list.d/ros-latest.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        ros-noetic-ros-base \
        ros-noetic-cv-bridge \
        python3-cv-bridge \
        python3-pip \
        python3-rosdep \
        build-essential \
        cmake && \
    rosdep init && rosdep update --rosdistro=noetic && \
    rm -rf /var/lib/apt/lists/*

# ─── Pin de typing-extensions a una versión compatible con Python 3.8 ───
# Esto evita que pip resuelva una versión que requiera Python >= 3.9.
RUN pip3 install --no-cache-dir "typing-extensions<4.13"

# ─── PyTorch con soporte CUDA 11.8 (lo necesita YOLOv8 para usar la GPU) ───
# Se instala ANTES de ultralytics para que pip no jale por defecto la versión CPU.
RUN pip3 install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu118

# ─── Resto de dependencias de Python ───
# - tensorflow 2.13.* es la última versión que soporta Python 3.8.
#   Detecta automáticamente el CUDA 11.8 + cuDNN 8 de la imagen base.
# - onnxruntime-gpu 1.16.* es la última con wheels para Python 3.8 (1.17+ ya los dropeó).
#   Está compilada contra CUDA 11.8 + cuDNN 8, justo lo que tiene la imagen base.
# - ultralytics < 8.3 para mantener compatibilidad con Python 3.8.
RUN pip3 install --no-cache-dir \
        "numpy>=1.20,<2.0" \
        mediapipe==0.10.9 \
        "tensorflow==2.13.*" \
        "ultralytics<8.3" \
        "onnxruntime-gpu==1.16.*" \
        onnx \
        "lap>=0.5.12"

# ─── Crear workspace catkin con el paquete-espejo de puppy_control ───
RUN mkdir -p /catkin_ws/src
COPY ros_msgs/puppy_control /catkin_ws/src/puppy_control
COPY ros_msgs/sensor /catkin_ws/src/sensor

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
                  cd /catkin_ws && \
                  catkin_make"

# ─── Tus scripts ───
COPY scripts/puppypi_system/puppy_ibvs_controller.py /puppy_ibvs_controller.py
COPY scripts/puppypi_system/puppy_pov.py /puppy_pov.py
COPY scripts/puppy_frame_processing.py /puppy_frame_processing.py
COPY scripts/puppy_command.py /puppy_command.py

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]