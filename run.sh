#!/bin/bash

# Detectar IP de la laptop automáticamente
LAPTOP_IP=$(hostname -I | awk '{print $1}')

# IP del PuppyPi — se pasa como argumento o se detecta via ping
PUPPYPI_IP=${1:-""}

# Si no se escribe nada: muestra la sintaxis esperada
if [ -z "$PUPPYPI_IP" ]; then
    echo "Uso: ./run.sh <IP_PUPPYPI>"
    echo "Ejemplo: ./run.sh 192.168.100.166"
    exit 1
fi

#IPs del sistema
echo "Laptop IP:  $LAPTOP_IP"
echo "PuppyPi IP: $PUPPYPI_IP"

# Permitir acceso al display
xhost +local:docker

# Ejecutar contenedor
docker run -it \
    --gpus all \
    --network host \
    -e ROS_MASTER_URI=http://$PUPPYPI_IP:11311 \
    -e ROS_IP=$LAPTOP_IP \
    -e PUPPYPI_IP=$PUPPYPI_IP \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $(pwd)/models:/models \
    -v $(pwd)/scripts/camera_calibration/camera_params.npz:/models/camera_params.npz \
    puppypi_follower