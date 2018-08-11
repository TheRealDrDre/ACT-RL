# ACT-RL

A module to replace ACT-R's utility module with classic RL theory algorithms

## Current Implementation

U_p = U_p + α[Rt - Up]

Where Rt = R - t(Up), and t(Up) is the elapsed since the firing of Up. 

## Why a New Implementation

The standard, utility-based implementation suffers from a few problems. First, it does not estimate cumulative future rewards. Second, the quantity Rt - Up might become negative, when the R - t(Up) < Up. In other words, a production that, after high-school graduation, go-to-college or find-a-job, if go-to-college fires and gets a reward after 4 years, it will likely be “punished” for it. In many ways, this does not matter, since what matter for utility-based competition is the relative utility. However, to make sure that the relative utilities remain consistent, rewards should have a perfect value that offsets the timing. But this defeats the purpose of RL-- a RL agent should find the way to estimate the expected future rewards.

Similarly, this system cannot take into account individual differences in temporal discounting, which we know exists. Again, they could be rephrased in terms of different values for R, but this seems unrealistic since they wouild produce different behaviros for short term as well.

Q-Learning in ACT-R

In Q learning, actions a1 … an  are tied to a state s1 … sn. The value Q(s,a) of an action a updated according to the equation:

Q(st,at) = Q(st,at) + α[Rt+1 + γQ(st+1,at+1)) - Q(st,at)]

Where Q(st+1,a) is the value of an action a that applies to the state st+1 that follows st after the execution of at. The value could be the value of the next action (using a SARSA rule) or the max value of the possible actions (using an off-policy, Q-Learning rule). 

In ACT-R, productions are implicitly tied to a state by their condition. Thus a production p can be expressed as an <s,a> pair. From this, it is fairly simple to adapt SARSA with the following procedure:

After the i-th production pi  is selected, update the previous production pi-1 according to the rule:

Q(pi) = Q(pi) + α[R + γQ(pi+1) - Q(pi)]

Here it is, in Lisp pseudocode:
