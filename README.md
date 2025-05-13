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
``bash
- `https://www.python.org/downloads/?ref=nerdschalk.com`
- go to the link above and select the OS u are on and do the instructions
- mac: `cmd + space` and type `terminal` for terminal
- win: `win + r` for `cmd prompt`
- download the repo by opening the main folder and click the code button at the top-right and click downlaod into zip
- run `cd ~/Downloads/dqn-helixjump-main` # for Mac
- run `cd %user%/Downlaods/dqn-helix-jump` # for Windows; %user% = the user or pc ex C:/Users/`ADAMthesmarty`>
- run `python -m venv venv`
- run `venv\Scripts\activate`  # Windows activation
- run `source myvenv/bin/activate` # Linux and Mac
- run `pip install -r requirements.txt`
- run `python 1.py` # Windows
- run `python3 1.py` # Linux and Mac
- IMPORTANT: when running helix_jump_dqn.py, you must open full screen on https://www.crazygames.com/game/helix-jump at the corner of the game windows it has a fullscreen button, click it.
- for better results try to get a bright colored tower
- when finished with project run: deactivate
