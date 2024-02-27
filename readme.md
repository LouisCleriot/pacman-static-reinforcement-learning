# Readme

This project calculates optimal policies in a static pacman-type maze. There are 2 resolution algorithms:

- Value Iteration
- Q-learning

## Organization

Two text files containing information about the maze and the parameter values of the 2 algorithms are needed:

- value-iteration.txt
- Q-learning.txt

The grids consist of 4 values:

- 0: empty space
- 1: wall
- 2: goal space
- 3: ghost space

The source code is divided into 2 files:

- Environment.py: contains the classes for states and the environment as well as all the methods
- main_notebook.py: contains the code to parse text files and run the algorithms.
