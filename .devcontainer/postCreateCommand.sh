set -e

git config --global safe.directory '*'
git config --global core.editor "code --wait"
git config --global pager.branch false

echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" >> ~/.bashrc
echo "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.zsh_history" >> ~/.zshrc
echo "setopt histignorealldups" >> ~/.zshrc
sudo chown -R $USER /commandhistory

echo "export ROS_IP=127.0.0.1" >> ~/.bashrc
echo "export ROS_IP=127.0.0.1" >> ~/.zshrc
echo "export ROS_HOSTNAME=127.0.0.1" >> ~/.bashrc
echo "export ROS_HOSTNAME=127.0.0.1" >> ~/.zshrc
echo "export ROS_MASTER_URI=http://127.0.0.1:11311" >> ~/.bashrc
echo "export ROS_MASTER_URI=http://127.0.0.1:11311" >> ~/.zshrc

echo "export ROS_PYTHON_VERSION=3" >> ~/.bashrc
echo "export ROS_PYTHON_VERSION=3" >> ~/.zshrc

echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> ~/.bashrc
echo "export ROSLAUNCH_SSH_UNKNOWN=1" >> ~/.zshrc

echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
echo "source /opt/ros/noetic/setup.zsh" >> ~/.zshrc
echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
echo "source $(pwd)/devel/setup.zsh" >> ~/.zshrc

git submodule update --init --recursive
sudo usermod -a -G plugdev $USER
sudo usermod -aG dialout $USER
sudo mkdir -p /etc/udev/rules.d
sudo cp src/Azure_Kinect_ROS_Driver/scripts/99-k4a.rules /etc/udev/rules.d/
sudo chmod a+rw /dev/bus/usb/*/*

source /opt/ros/noetic/setup.bash
# rosdep install -r -y --from-paths src --ignore-src
catkin clean --yes
catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DPYTHON_INCLUDE_DIR=/usr/include/python3.8 -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.8.so
catkin build

echo "DONE!"
