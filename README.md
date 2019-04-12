
![Lunar Lander Workshop](images/LunarLanderBanner.png)

# Lunar Lander Workshop

This repo contains an education workshop developed to allow students to build an understanding of reinforced learning. In the workshop the students learn aspects of machine learning required to land a lunar lander which is part of the OpenAI Gym environment. Step by step they can build an AI and see it working. The workshop takes around 2 hours and is intended to be run on Chromebooks. 

In addition there is a particular focus on how to measure the 'fitness' of the resulting AI. 



## Installation

### Jupyter Notebook - Local Machine Ubuntu Linux 18.04 (Not Chromebook)

Install Python3, Pip3, iPython3 and Python3 Tkinter

`sudo apt install python3-pip python3-dev ipython3 python3-tk`

Install OpenAI Gym Pre-req

`sudo apt-get install -y zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb`


Now add to your `.bashrc` file

`export PATH=$PATH:~/.local/bin/`

Install Jupyter

`pip3 install jupyter`

Install OpenAI Gym itself, and some default environments and the notebook requirements

`pip3 install gym 'gym[box2d]' 'gym[atari]' 'gym[classic_control]' numpy torch matplotlib JSAnimation tensorflow ipywidgets`


Enable progress bar extensions

`jupyter nbextension enable --py widgetsnbextension`


### JupyterHub - Remote Server Ubuntu Linux 18.04 (Chromebook/WebBrowser)
Follow these steps to install on a remotely hostly machine 

Install Python3, Pip3, iPython3, Python3 Tkinter, 

`sudo apt install python3-pip python3-dev ipython3 python3-tk`

Now install NPM, NodeJS
`curl -sL https://deb.nodesource.com/setup_8.x | sudo -E bash -`
`sudo apt-get install -y nodejs nodejs-legacy`

Install OpenAI Gym Pre-req

`sudo apt-get install -y zlib1g-dev libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev libsdl2-dev libosmesa6-dev patchelf ffmpeg xvfb`

Install JupyterHub

`python3 -m pip install jupyterhub`

Install HTTP Proxy

`sudo npm install -g configurable-http-proxy`

Install Notebook

`python3 -m pip install notebook`

Install the OpenAI Gym stuff and our notebook dependancies

`pip3 install gym 'gym[box2d]' 'gym[atari]' 'gym[classic_control]' numpy torch matplotlib JSAnimation tensorflow ipywidgets`
