
![Lunar Lander Workshop](images/LunarLanderBanner.png)

# Lunar Lander Workshop

This repo contains an education workshop developed to allow students to build an understanding of reinforced learning. In the workshop the students learn aspects of machine learning required to land a lunar lander which is part of the OpenAI Gym environment. Step by step they can build an AI and see it working. The workshop takes around 2 hours and is intended to be run on Chromebooks. 

In addition there is a particular focus on how to measure the 'fitness' of the resulting AI. 



## Installation

### Jupyter Notebook - Local Machine Ubuntu Linux 18.04 (Not Chromebook)

Install Python3, Pip3, iPython3 and Python3 Tkinter

`sudo apt install python3-pip python3-dev ipython3 python3-tk`

Now add to your `.bashrc` file

`export PATH=$PATH:~/.local/bin/`

Install Jupyter

`pip3 install jupyter`

Install OpenAI Gym Pre-req

`sudo apt-get install -y zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb`


Install OpenAI Gym itself, this is the minimal install

`pip3 install gym`

Now some specific environments (only box2d is required)

`pip3 install 'gym[box2d]'`

`pip3 install 'gym[atari]'`

`pip3 install 'gym[classic_control]'`

Now the notebook requirements

`pip3 install numpy torch matplotlib JSAnimation tensorflow`

Enable progress bar extensions

`pip3 install ipywidgets`
`jupyter nbextension enable --py widgetsnbextension`


### JupyterHub - Remote Server Ubuntu Linux 18.04 (Chromebook/WebBrowser)
Follow these steps to install on a remotely hostly machine 







