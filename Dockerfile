# Use the official ROS 2 Dashing image as the base
FROM ros:dashing

# Set the working directory inside the container
WORKDIR /ros2_ws

# Set bash as the default shell and enable executing chained commands
SHELL ["/bin/bash", "-c"]

# Install essential ROS 2 development tools
# python3-colcon-common-extensions is needed for building packages
RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Automatically source ROS 2 setup script for interactive shells
RUN echo "source /opt/ros/dashing/setup.bash" >> ~/.bashrc

# Optionally, copy local source code during build (comment out if using volumes primarily)
# COPY ./ros2_ws/src /ros2_ws/src

# Default command to run when the container starts (provides an interactive shell)
CMD ["bash"] 