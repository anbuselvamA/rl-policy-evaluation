# POLICY EVALUATION

## AIM
To develop a Python program to evaluate the given policy by maximizing its cumulative reward while dealing with Grid World with Wind.

## PROBLEM STATEMENT
The agent is placed in a 5x5 grid where it must navigate to a goal position. However, in certain columns, wind affects the agent’s movement, pushing it upward. The challenge is to find an optimal policy that maximizes the cumulative reward while considering the stochastic effects of wind.
## POLICY EVALUATION FUNCTION
![image](https://github.com/user-attachments/assets/00b628f2-a4a8-4fbb-b0c2-09c9e5dac4a1)

## PROGRAM
```
pip install git+https://github.com/mimoralea/gym-walk#egg=gym-walk
import warnings ; warnings.filterwarnings('ignore')

import gym
import numpy as np

import random
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
np.set_printoptions(suppress=True)
random.seed(123); np.random.seed(123);

def print_policy(pi, P, action_symbols=('<', 'v', '>', '^'), n_cols=4, title='Policy:'):
    print(title)
    arrs = {k:v for k,v in enumerate(action_symbols)}
    for s in range(len(P)):
        a = pi(s)
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), arrs[a].rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")


def print_state_value_function(V, P, n_cols=4, prec=3, title='State-value function:'):
    print(title)
    for s in range(len(P)):
        v = V[s]
        print("| ", end="")
        if np.all([done for action in P[s].values() for _, _, _, done in action]):
            print("".rjust(9), end=" ")
        else:
            print(str(s).zfill(2), '{}'.format(np.round(v, prec)).rjust(6), end=" ")
        if (s + 1) % n_cols == 0: print("|")
def probability_success(env, pi, goal_state, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        while not done and steps < max_steps:
            state, _, done, h = env.step(pi(state))
            steps += 1
        results.append(state == goal_state)
    return np.sum(results)/len(results)
def mean_return(env, pi, n_episodes=100, max_steps=200):
    random.seed(123); np.random.seed(123) ; env.seed(123)
    results = []
    for _ in range(n_episodes):
        state, done, steps = env.reset(), False, 0
        results.append(0.0)
        while not done and steps < max_steps:
            state, reward, done, _ = env.step(pi(state))
            results[-1] += reward
            steps += 1
    return np.mean(results)

env = gym.make('FrozenLake-v1')
P = env.env.P
init_state = env.reset()
goal_state = 15
LEFT, DOWN, RIGHT, UP = range(4)
P

init_state


state, reward, done, info = env.step(RIGHT)
print("state:{0} - reward:{1} - done:{2} - info:{3}".format(state, reward, done, info))

pi_frozenlake = lambda s: {
    0: RIGHT,
    1: DOWN,
    2: RIGHT,
    3: LEFT,
    4: DOWN,
    5: LEFT,
    6: RIGHT,
    7:LEFT,
    8: UP,
    9: DOWN,
    10:LEFT,
    11:DOWN,
    12:RIGHT,
    13:RIGHT,
    14:DOWN,
    15:LEFT #Stop
}[s]
print_policy(pi_frozenlake, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)

print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_frozenlake, goal_state=goal_state) * 100,
    mean_return(env, pi_frozenlake)))
pi_2 = lambda s: {
    0: RIGHT,  1: RIGHT,  2: DOWN,  3: LEFT,
    4: RIGHT,  5: DOWN,   6: RIGHT,  7: LEFT,
    8: DOWN,   9: RIGHT, 10: DOWN,  11: DOWN,
    12: RIGHT, 13: RIGHT, 14: DOWN, 15: LEFT  
}[s]





print("Name: Anbuselvam A")
print("Register Number: 212222240009")
print_policy(pi_2, P, action_symbols=('<', 'v', '>', '^'), n_cols=4)
print('Reaches goal {:.2f}%. Obtains an average undiscounted return of {:.4f}.'.format(
    probability_success(env, pi_2, goal_state=goal_state) * 100,
    mean_return(env, pi_2)))
success_pi2 = probability_success(env, pi_2, goal_state=goal_state) * 100
mean_return_pi2 = mean_return(env, pi_2)

print("\nYour Policy Results:")
print(f"Reaches goal: {success_pi2:.2f}%")
print(f"Average undiscounted return: {mean_return_pi2:.4f}")
success_pi1 = probability_success(env, pi_frozenlake, goal_state=goal_state) * 100
mean_return_pi1 = mean_return(env, pi_frozenlake)

print("\nComparison of Policies:")
print(f"First Policy - Success Rate: {success_pi1:.2f}%, Mean Return: {mean_return_pi1:.4f}")
print(f"Your Policy  - Success Rate: {success_pi2:.2f}%, Mean Return: {mean_return_pi2:.4f}")

if success_pi1 > success_pi2:
    print("\nThe first policy is better based on success rate.")
elif success_pi2 > success_pi1:
    print("\nYour policy is better based on success rate.")
else:
    print("\nBoth policies have the same success rate.")

if mean_return_pi1 > mean_return_pi2:
    print("The first policy is better based on mean return.")
elif mean_return_pi2 > mean_return_pi1:
    print("Your policy is better based on mean return.")
else:
    print("Both policies have the same mean return.")
def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)

    while True:
        delta = 0
        for s in range(len(P)):
            v = 0
            a = pi(s)
            for prob, next_state, reward, done in P[s][a]:
                v += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(V[s] - v))
            V[s] = v

        if delta < theta:
            break

    return V


V1 = policy_evaluation(pi_frozenlake, P,gamma=0.99)
print_state_value_function(V1, P, n_cols=4, prec=5)

V2 = policy_evaluation(pi_2, P, gamma=0.99)

print("\nState-value function for Your Policy:")
print_state_value_function(V2, P, n_cols=4, prec=5)

if np.sum(V1 >= V2) == len(V1):
    print("\nThe first policy is the better policy.")
elif np.sum(V2 >= V1) == len(V2):
    print("\nYour policy is the better policy.")
else:
    print("\nBoth policies have their merits.")
V1>=V2

if(np.sum(V1>=V2)==11):
  print("The first policy is the better policy")
elif(np.sum(V2>=V1)==11):
  print("The second policy is the better policy")
else:
  print("Both policies have their merits.")
```


## OUTPUT:
Mention the first and second policies along with its state value function and compare them
![image](https://github.com/user-attachments/assets/33657b6e-81fc-4062-84bf-c6452d7d2816)
![image](https://github.com/user-attachments/assets/6dd3596d-3019-4f92-aaa0-136d9dd629e4)
![image](https://github.com/user-attachments/assets/081807d4-9604-4cc5-a5a8-152fdecaa5ac)
![image](https://github.com/user-attachments/assets/b21762b7-2df3-4404-aaee-d70fcb34c2c7)
![image](https://github.com/user-attachments/assets/f2866915-70b2-46b3-a37a-dbb426ea2298)
![image](https://github.com/user-attachments/assets/6f3310ce-c453-4d19-8de7-0ccac3012faf)
![image](https://github.com/user-attachments/assets/1b9895d8-eb27-4661-b1cc-14880b918af0)

![image](https://github.com/user-attachments/assets/18da105e-35fc-4105-9eaa-3d5ec6e1931a)
![image](https://github.com/user-attachments/assets/e3e4a53e-f13e-4571-8203-14a9b8206ef3)













## RESULT:

Thus, the Given Policy has been Evaluated and Optimal Policy has been Computed using Python Programming and execcuted successfully.
