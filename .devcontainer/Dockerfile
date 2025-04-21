FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install language and timezone
RUN apt-get update \
    && apt-get install -y locales \
    && locale-gen en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && ln -fs /usr/share/zoneinfo/UTC /etc/localtime \
    && apt-get update \
    && apt-get install -y tzdata \
    && dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        curl wget gnupg2 lsb-release sudo software-properties-common \
        python3 python3-dev python3-pip python3-tk python3-distutils python3-apt && \
    apt-get install -y --no-install-recommends \
        ssh git jq ffmpeg dkms build-essential cmake ninja-build x264 libx264-dev \
        libxcb-xinerama0 libqt5gui5 libx11-xcb-dev libglu1-mesa-dev libxrender-dev \
        libxi-dev libxkbcommon-dev libxkbcommon-x11-dev && \
    rm -rf /var/lib/apt/lists/*

# Install ROS
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt-get update && \
    apt-get install -y --allow-downgrades --allow-remove-essential --allow-change-held-packages --no-install-recommends \
        libpcap-dev libopenblas-dev gstreamer1.0-tools libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libgstreamer-plugins-good1.0-dev python3-rosinstall python3-rosinstall-generator python3-wstool \
        ros-noetic-ros-base ros-noetic-cv-bridge ros-noetic-rgbd-launch ros-noetic-xacro ros-noetic-angles \
        ros-noetic-tf2-geometry-msgs ros-noetic-camera-info-manager ros-noetic-joint-state-publisher \
        ros-noetic-robot-state-publisher ros-noetic-rqt-image-view ros-noetic-moveit python3-catkin-tools \
        ros-noetic-joint-state-publisher-gui ros-noetic-moveit-visual-tools ros-noetic-ros-numpy \
        python3-empy python3-rosdep ipython3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install kinect
RUN curl -sSL -O https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb && \ 
    dpkg -i packages-microsoft-prod.deb && \
    rm -f packages-microsoft-prod.deb && \
    apt-get update && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends k4a-tools libk4a1.4-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV ROS_DISTRO=noetic
ENV LD_LIBRARY_PATH=/opt/ros/noetic/lib
ENV PATH=/opt/ros/noetic/bin:$PATH
ENV ROS_PYTHON_VERSION=3
RUN rosdep init && rosdep update

ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN python3 -m pip install pip --upgrade && \
    pip3 install rospkg catkin_pkg numpy && \
    pip3 install packaging setuptools==58.2.0 && \
    pip3 install torch==2.1.2 torchvision==0.16.2 -f https://download.pytorch.org/whl/cu118/torch_stable.html && \
    pip3 install git+https://github.com/facebookresearch/detectron2.git && \
    pip3 install --upgrade opencv-python transformers einops scikit-video timm scipy && \
    pip3 install gradio==4.11.0

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libpcl-dev libusb-1.0-0-dev libfmt-dev udev clang-format-12 ros-noetic-pcl-conversions && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib:$LD_LIBRARY_PATH

WORKDIR /
