# Helix Jump DQN Bot

This project is a Deep Q-Learning (DQN) bot for automating gameplay, designed for Helix Jump-like games using screen capture and AI-based control.

## Features
- Real-time screen capture using `mss`
- Action simulation with `pyautogui`
- Deep Q-Network (DQN) with TensorFlow
- Experience Replay & Target Network sync


## Requirements
- Python 3.9 or 3.10 (recommended for TensorFlow compatibility)

## Get python
- `https://www.python.org/downloads/?ref=nerdschalk.com`
- this downlaods python 3.12 and pip
  
## Instructions 
- mac: `cmd + space` and type `terminal` for terminal
- win: `win + r` for `cmd prompt`
- download the repo by opening the main folder and click the code button at the top-right and click downlaod into zip; also unzip the .zip into a folder
- run `cd ~/Downloads/dqn-helixjump-main` # for Mac
- run `cd %user%/Downlaods/dqn-helix-jump` # for Windows; %user% = the user or pc ex C:/Users/`ADAMthesmarty`>
- run `python -m venv helix-dqn`
- run `helix-dqn\Scripts\activate`  # Windows activation
- run `source helix-dqn/bin/activate` # Linux and Mac
- run `pip install -r requirements.txt`
- run `python helix_jump_dqn.py` # Windows
- run `python3 helix_jump_dqn.py` # Linux and Mac
- IMPORTANT: when running helix_jump_dqn.py, you must open full screen on https://www.crazygames.com/game/helix-jump at the corner of the game windows it has a fullscreen button, click it.
- for better results try to get a bright colored tower
- if u wanna force quit:
- when done just close the cmd prompt and clikc terminate action if it asks
- if u wanna close the app but continue working:
- click q to exit/end python code
- when finished with project run: deactivate ; deactivates the venv
- delete the venv: rm -r helix-dqn
## Important
- making a virtual enviroment makes sure u ahevb the right package and lets u simple remove the env to remove the packages
- so it u wanna make a project it is better to amek a venv so that package data is permemant but dletes when u delete the venv
