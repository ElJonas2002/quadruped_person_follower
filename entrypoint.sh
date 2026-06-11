#!/bin/bash
echo "${PUPPYPI_IP} raspberrypi" >> /etc/hosts
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash

python3 /puppy_ibvs_controller.py &
python3 /puppy_pov.py