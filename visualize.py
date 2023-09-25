import numpy as np
from sklearn.linear_model import LogisticRegression
import gym

if __name__ == "__main__":
    # Train model to imitate expert data
    labels = np.load("acts_data.npy")
    states = np.load("state_data.npy")
    clf = LogisticRegression(random_state=0).fit(states, labels[:,0])
    

    # Render policy behaviour when controlling the spaceship.
    env = gym.make("LunarLander-v2")
    obs = env.reset()
    prev_screen = env.render(mode='rgb_array')

    while True:
       action = clf.predict(np.asarray([obs]))
       obs, reward, done, info = env.step(action[0])
       env.render(mode='rgb_array')
       if done:
           obs = env.reset()


