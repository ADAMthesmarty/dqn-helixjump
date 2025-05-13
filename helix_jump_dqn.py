import cv2
import numpy as np
import mss
import pyautogui
import tensorflow as tf
import random
import time
import os
import sys
import select
import termios
import tty
import collections

# ——— DQN Hyperparameters ———
IMG_HEIGHT, IMG_WIDTH = 84, 84
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.001
MEMORY_SIZE = 20000
BATCH_SIZE = 70
ACTION_SPACE = ["left", "right", "none"]
MODEL_FILENAME = "exam.h5"

# ——— Memory buffer ———
memory = collections.deque(maxlen=MEMORY_SIZE)


# ——— Build / load model ———
def create_dqn_model():
    m = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1)),
            tf.keras.layers.Conv2D(64, (8, 8), strides=4, activation="relu"),
            tf.keras.layers.Conv2D(128, (4, 4), strides=2, activation="relu"),
            tf.keras.layers.Conv2D(256, (3, 3), strides=1, activation="relu"),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(2048, activation="relu"),
            tf.keras.layers.Dense(1024, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(len(ACTION_SPACE), activation="linear"),
        ]
    )
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.MeanSquaredError(),
    )
    return m


if os.path.exists(MODEL_FILENAME):
    print("Loading saved model...")
    model = tf.keras.models.load_model(
        MODEL_FILENAME,
        custom_objects={"MeanSquaredError": tf.keras.losses.MeanSquaredError()},
    )
else:
    print("Creating new model...")
    model = create_dqn_model()

target_model = create_dqn_model()
target_model.set_weights(model.get_weights())


# ——— Utilities ———
def press_action(action):
    if action == "left":
        pyautogui.keyDown("left")
        time.sleep(0.1)
        pyautogui.keyUp("left")
    elif action == "right":
        pyautogui.keyDown("right")
        time.sleep(0.1)
        pyautogui.keyUp("right")


def preprocess(frame):
    f = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
    f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    f = f.astype("float32") / 255.0
    return np.expand_dims(f, axis=-1)


def choose_action(state, eps):
    if np.random.rand() < eps:
        return random.randrange(len(ACTION_SPACE))
    q = model.predict(state[None, ...], verbose=0)[0]
    return np.argmax(q)


def replay():
    if len(memory) < BATCH_SIZE:
        return
    batch = random.sample(memory, BATCH_SIZE)
    states, acts, rews, nexts, dones = zip(*batch)
    states = np.stack(states)
    nexts = np.stack(nexts)
    q_targets = model.predict(states, verbose=0)
    q_next = target_model.predict(nexts, verbose=0)
    for i in range(BATCH_SIZE):
        q_targets[i, acts[i]] = rews[i] + (1 - dones[i]) * GAMMA * q_next[i].max()
    model.fit(states, q_targets, epochs=1, verbose=0)


# ——— Non-blocking console input setup ———
fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)
tty.setcbreak(fd)


def quit_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        c = sys.stdin.read(1)
        return c.lower() == "q"
    return False


# ——— Main loop ———
def main():
    global EPSILON
    with mss.mss() as sct:
        last_update = time.time()
        last_save = time.time()
        state = None
        score = 0
        try:
            while True:
                img = np.array(sct.grab(sct.monitors[0]))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                proc = preprocess(frame)

                if state is None:
                    state = proc
                    continue

                a = choose_action(state, EPSILON)
                press_action(ACTION_SPACE[a])

                reward = 1
                if np.mean(frame) > 220:
                    reward, done = -100, True
                else:
                    done = False

                memory.append((state, a, reward, proc, done))
                state = proc
                score += reward

                replay()

                # update target network
                if time.time() - last_update > 5:
                    target_model.set_weights(model.get_weights())
                    last_update = time.time()
                    print(f"[Sync] Epsilon={EPSILON:.3f}")

                # save model periodically
                if time.time() - last_save > 30:
                    print("[Save] Model checkpoint")
                    model.save(MODEL_FILENAME)
                    last_save = time.time()

                # epsilon decay
                if EPSILON > EPSILON_MIN:
                    EPSILON *= EPSILON_DECAY

                # check for quit
                if quit_pressed():
                    print("Exit key pressed. Quitting now.")
                    break

                # logging
                print(f"Score={score}, Epsilon={EPSILON:.3f}")

                if done:
                    print(f"Game Over! Final Score={score}")
                    time.sleep(2)
                    state = None
                    score = 0

        finally:
            # restore terminal state
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


if __name__ == "__main__":
    print("Starting Helix Jump DQN (press 'q' to quit)...")
    main()
