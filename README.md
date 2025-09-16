# VALUE ITERATION ALGORITHM

## AIM
To find an optimal policy for an agent navigating a grid-world with slippery tiles, aiming to reach a goal state while maximizing expected rewards using value iteration algorithm.

## PROBLEM STATEMENT
The problem involves using the Value Iteration algorithm to find the best strategy for an agent in the Frozen Lake environment. The agent must navigate icy terrain, avoid hazards, and reach the goal while optimizing cumulative rewards in an uncertain environment.

## POLICY ITERATION ALGORITHM

Step 1:
Set the value of each state to 0 (initial guess).<br>
Step 2:
Look at all the actions you can take from that state (like moving up, down, left, or right).<br>
Step 3:
Calculate the expected value of each action (i.e., how good that action is based on its possible results).<br>
Step 4:
Pick the action that gives the highest value and update the value of the state with that number.<br>
Step 5:
Keep updating the values for all states until the difference between the old and new values is very small.<br>
Step 6:
Once the values have stabilized, go through each state again and pick the action that leads to the highest value. This gives you the optimal action (policy) for each state.<br>
### ENVIRONMENT : 
```python
envdesc  = ['SFHH','HFFH','HHFH', 'HGFH']
env = gym.make('FrozenLake-v1',desc=envdesc)
init_state = env.reset()
goal_state = 13
P = env.env.P
```
## VALUE ITERATION FUNCTION
### Name:DHARMARAJ S
### Register Number:212222240025

```
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)   # Initialize state-value function
    pi = np.zeros(len(P), dtype=int)         # Initialize policy (best action per state)

    while True:
        delta = 0
        for s in range(len(P)):  # Loop over all states
            v = V[s]
            q_values = []
            for a in P[s]:  # Loop over possible actions
                q_sa = 0
                for prob, next_s, reward, done in P[s][a]:
                    q_sa += prob * (reward + gamma * V[next_s])
                q_values.append(q_sa)

            V[s] = max(q_values)              # Update state-value
            pi[s] = np.argmax(q_values)       # Update optimal action
            delta = max(delta, abs(v - V[s]))  # Track convergence

        if delta < theta:  # Stop when values converge sufficiently
            break

    return V, pi
```
## OUTPUT:
#### Optimal policy and state-value function :
<img width="974" height="291" alt="image" src="https://github.com/user-attachments/assets/52d59c18-5a4b-4789-84e8-16d4022f0f3e" />

#### Pobability Success :
<img width="966" height="42" alt="image" src="https://github.com/user-attachments/assets/ea2b37b3-930b-4686-9b6c-cf53543ab253" />

#### State Value Function :
<img width="797" height="188" alt="image" src="https://github.com/user-attachments/assets/58f4960e-06d2-4a4e-a848-dba19e2195d0" />


## RESULT:
Thus, a Python program is developed to find the optimal policy for the given MDP using the value iteration algorithm.
