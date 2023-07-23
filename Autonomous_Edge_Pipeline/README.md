The code contained in this folder is a modified version of the 'Self-Learning Autonomous Edge Learning and Inferencing Pipeline (AEP)'. The original version can be found in this repository:

https://github.com/Edge-Learning-Machine/Autonomous-Edge-Pipeline

The original README is still in this folder, with the general information about the pipeline and the settable values, some of which are now changed.

These are the main modifications we did to use it for our project:
- No Decision Trees
- More information is now available in the 'main.c'. For instance, all the coordinates also of the k-neighbors given a test datum
- There are aggregation functions for the "coordinator", which is the board that aggregates the information from the nodes since the pipeline is now used in a network of UWB nodes
- The training dataset and test dataset are now used differently, depending on the settings used for the nodes
- Some modifications to ease the process of generating all the binaries with different settings to test the distributed approach in simulation with lots of constraints combinations (more macros are now defined)
- Everything that was printed out is now completely changed. When the code prints (in simulation mode), it prints data that is less readable since it was used to do automated performance evaluation
