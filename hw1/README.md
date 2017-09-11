# CS294-112 HW 1: Imitation Learning

Dependencies: TensorFlow, OpenAI Gym, Roboschool v1.1

The only file that you need to look at is `run_expert.py`, which is code to load up an expert policy, run a specified number of roll-outs, and save out data.

In `experts/`, the provided expert policies are:
* RoboschoolAnt-v1.py
* RoboschoolHalfCheetah-v1.py
* RoboschoolHopper-v1.py
* RoboschoolHumanoid-v1.py
* RoboschoolReacher-v1.py
* RoboschoolWalker2d-v1.py

# Things to watch
## DAgger vs Imitation
The training time (the RELATIVE table of tensorboard) of DAgger and imitation should be similar per batch. However, DAgger needs to generate training data along the way while imitation uses existing expert data, thus it appears DAgger is slower than imitation. 
For 20 rollouts and 10000 batches of 100 samples, DAgger and imitation perform similarly on Ant. Both are still far from expert (1100 vs 1600)

For 100 rollouts and 20000 batches of 100 samples, DAgger seems slightly better than imitation on Ant. Both perform close to expert. 

## Humanoid requires a larger NN than Ant