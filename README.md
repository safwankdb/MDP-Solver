# MDP-Solver
A python codebase for solving sequential Markov Decision Problems

### Algorithms
Users can choose one of the following solving methods
- Value Iteration
- Howard's Policy Iteration
- Linear Programming (using PuLP)

### Usage

```
usage: planner.py [-h] [--mdp MDP] [--algorithm {lp,vi,hpi}]

optional arguments:
  -h, --help            show this help message and exit
  --mdp MDP
  --algorithm {lp,vi,hpi}
```


### MDP File Format
Each MDP is provided as a text file in the following format.
```
numStates S
numActions A
start st
end ed1 ed2 ... edn
transition s1 ac s2 r p
transition s1 ac s2 r p
. . .
. . .
. . .
transition s1 ac s2 r p
mdptype mdptype
discount gamma
```
The number of states _S_ and the number of actions _A_ will be integers greater than 1, and at most 100. Assume that the states are numbered _0, 1, ..., S - 1_, and the actions are numbered _0, 1, ..., A - 1_. Each line that begins with "transition" gives the reward and transition probability corresponding to a transition, where _R(s1, ac, s2) = r_ and _T(s1, ac, s2) = p_. Rewards can be positive, negative, or zero. Transitions with zero probabilities are not specified. mdptype will be one of continuing and episodic. The discount factor gamma is a real number between 0 (included) and 1 (included). _st_ is the start state, which you might need for Task 2 (ignore for Task 1). _ed1, ed2,..., edn_ are the end states (terminal states). For continuing tasks with no terminal states, this list is replaced by -1.

### Output
The output of your planner is written to standard output.
```
V*(0)   π*(0)
V*(1)   π*(1)
.
.
.
V*(S - 1)   π*(S - 1)
```
