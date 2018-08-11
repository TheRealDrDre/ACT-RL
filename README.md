# ACT-RL

A module to replace ACT-R's utility module with classic RL theory algorithms

## Current Implementation

The Utility _Up_ of a production _p_ is currently calculated as:

... _Up_ = _Up_ + α(_Rt_ - _Up_)

Where _Rt_ = _R_ - _t_(_Up_), and _t_(_Up_) is the elapsed since the firing of Up. 

## Why a New Implementation

The standard, utility-based implementation suffers from a few problems. First, it does not estimate cumulative future rewards. Second, the quantity _Rt_ - _Up_ might become negative, when the _R_ - _t_(_Up_) < _Up_. In other words, a production that, after high-school graduation, go-to-college or find-a-job, if go-to-college fires and gets a reward after 4 years, it will likely be "punished" for it. In many ways, this does not matter, since what matter for utility-based competition is the relative utility. However, to make sure that the relative utilities remain consistent, rewards should have a perfect value that offsets the timing. But this defeats the purpose of RL---a RL agent should find the way to estimate the expected future rewards.

Similarly, this system cannot take into account individual differences in temporal discounting, which we know exists. Again, they could be rephrased in terms of different values for R, but this seems unrealistic since they wouild produce different behaviros for short term as well.

## Classic RL Approach to Action Selection 

In classic RL, actions _a_1, _a_2 ... _aM_  an  are tied to specific states _s_1, _s_2,  _sN_. Each action has an associated *Q value* which represents the estimated rewards expected to be generated in the long-term by selecting that particular action. The value _Q_(_s_,_a_) of an action _a_ is updated according to the temporal difference equation:

... _Q_(_st_,_at_) = _Q_(_st_,_at_) + α[_Rt_ + γ_Q_(_st_+1,_at_+1)) - _Q_(_st_,_at_)]

Where _Q_(_st_+1,_a_) is the value of an action a that applies to the state st+1 that follows st after the execution of at. This value could be the value of the next action, as implied by this notation (using a SARSA rule) or the max value of the possible actions, i.e., max[Q(_st_+1, _a_1,), Q(_st_+1, _a_2,) ... Q(_st_+1, _a_N,)] when using an off-policy rule like _Q_-Learning. 

## Implementing Classic RL in ACT-R

In ACT-R, productions are implicitly tied to a state by their condition. Thus a production p can be expressed as an <s,a> pair. From this, it is fairly simple to adapt SARSA with the following procedure:

After the _i_-th production _pi_  is selected, update the previous production pi-1 according to the rule:

_Q_(_pi_) = _Q_(_pi_) + α[_R_ + γ_Q_(_pi_+1) - _Q_(_pi_)]


### Handling Non-Markov Environment

This system by itself cannot handle non-markov environments. That is not a problem per se, but we can extend it as an _n_-_Q_-Learning or _n_-SARSA version by adding a parameter _n_ that specifies the number of backup steps for the update rule. In this implementation, the update rule simply propagates back in time to production _i - n_, and consists of _n_ successive updating stees:


1. 	_Q_(pi-1) = Q(pi-1) + α[R + γQ(pi) - Q(pi-1)]
2. 	_Q_(pi-2) = Q(pi-2) + α[R + γQ(pi-1) - Q(pi-2)]
...
_n_. 	_Q_(pi-n) = Q(pi-n) + α[R + γQ(pi-n+1) - Q(pi-n)]

### Handling Continuous Time

The main problem with ACT-R is that time decays linearly, producing negative rewards when unchecked. In the expressed equation, the recursive use of γQ(pi) guarantees that reward propagates back with exponentially declining returns with the number of steps. However, ACT-R needs to handle varying intervals between rewards. The simplest trick is to scale the parameter γ with a time-encoding exponent, so that events immediately attached have a value close to γ = 1:

γ = γ^(_t_(_i_) - _t_(_i_-1)

in which _t_(_i_) is the time at which production _pi_ fires. The time scale (seconds, minutes) can be easily adjusted by a scalar value.
