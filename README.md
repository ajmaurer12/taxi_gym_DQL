# taxi_gym_DQL

This project applies deep learning to a game scenario. The game comes from the Open AI gym library for test games and tasks for machine learning. The machine learning algorithm used is Deep Q Learning, with double network update and experience replay.

https://gym.openai.com/

Description of the game:

The game has 4 locations on a 5x5 grid a passenger can be or need to go. The taxi is controlled and moves around, though there are walls. Every move in a cardinal direction loses a point, and 10 points are lost for trying to pick up a passenger where they are not, or trying to drop off a passenger without a passenger in the car or trying to drop a passenger outside of a possible goal location. No points are lost when a passenger is validly picked up or placed in a possible goal, and 20 points are won when the passenger is correctly brought to the goal.

The code used on this game is based on code posted by Arthur Juliani:

https://medium.com/@awjuliani/simple-reinforcement-learning-with-tensorflow-part-4-deep-q-networks-and-beyond-8438a3e2b8df
