import torch
import random
import numpy as np

from AI_snake_game import SnakeGameAI, Direction, Point
from collections import deque
from snake_model import Linear_QModel, QTrainer
from snake_helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
BLOCK_SIZE = 20
INITIAL_EPSILON = 80
STATE_FEATURES = 11
HIDDEN_SIZE = 256
OUTPUT_ACTIONS = 3
DISCOUNT_RATE = 0.9

class Agent:

    def __init__(self):

        self.num_games = 0
        self.epsilon = 0 #randomness
        self.gamma = DISCOUNT_RATE #discount for future rewards
        self.memory = deque(maxlen=MAX_MEMORY) #to store states, actions, rewards. 
        # ^ List of tuples in deque structure. popleft when limit exceeded
        self.model = Linear_QModel(STATE_FEATURES, HIDDEN_SIZE, OUTPUT_ACTIONS)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        # get head position
        head = game.snake[0]

        #get position of four points around head
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        #get booleans representing current direction
        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        #current game state with 11 features represented as booleans
        #Danger straight, right, left (3) dependent on current direction
        #Direction left, right, up, or down (4)
        #Food left, right, up, down (4)
        state = [
            #Danger straight
            (dir_right and game.is_collision(point_right)) or
            (dir_left and game.is_collision(point_left)) or
            (dir_up and game.is_collision(point_up)) or
            (dir_down and game.is_collision(point_down)),

            #Danger right
            (dir_up and game.is_collision(point_right)) or
            (dir_down and game.is_collision(point_left)) or
            (dir_left and game.is_collision(point_up)) or
            (dir_right and game.is_collision(point_down)),

            #Danger left
            (dir_down and game.is_collision(point_right)) or
            (dir_up and game.is_collision(point_left)) or
            (dir_right and game.is_collision(point_up)) or
            (dir_left and game.is_collision(point_down)),

            #Current direction, one true boolean
            dir_left, 
            dir_right, 
            dir_up, 
            dir_down,

            #Food location
            game.food.x < game.head.x, #food left
            game.food.x > game.head.x, #food right
            game.food.y < game.head.y, #food up
            game.food.y > game.head.y  #food down
            ]

        #convert to array, and convert booleans to 0/1
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, gameover):

        #add variables to memory in order as a tuple. Pop left once max memory reached
        #Tuple form: state(array length 11), action(array length 3), reward(int), next_state(array length 11), gameover(boolean)
        self.memory.append((state, action, reward, next_state, gameover))
    
    def train_long_memory(self):
        
        #using variables from memory
        if len(self.memory) > BATCH_SIZE:
            # list of tuples with state, action, reward, next_state, gameover
            # Size is batch size
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            # not yet enough elements in memory
            mini_sample = self.memory

        #collect all variables from mini sample of same type using zip function
        states, actions, rewards, next_states, gameovers = zip(*mini_sample)
        #Now all in tuples of same length
        self.trainer.train_step(states, actions, rewards, next_states, gameovers)

    def train_short_memory(self, state, action, reward, next_state, gameover):
        
        #training for one game step
        self.trainer.train_step(state, action, reward, next_state, gameover)

    def get_action(self, state):
        #random moves: tradeoff between exploration (randomness) and exploitation (less randomness)
        self.epsilon = INITIAL_EPSILON - self.num_games
        final_move = [0,0,0]
        #Less likelihood for random moves as number of games increases
        #After number of games > Initial Epsilon, no more random moves
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        #move based on model
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            #prediction will be in form of three values [x, y, z] corresponding to moves
            #Need to take move which corresponds to max value (one hot encode)
            prediction = self.model(state0)
            #returns index of largest value
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move

def train():
    
    scores = []
    mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, gameover, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        #train short memory (for one step)
        agent.train_short_memory(state_old, final_move, reward, state_new, gameover)

        #remember
        agent.remember(state_old, final_move, reward, state_new, gameover)

        if gameover:

            #reset game
            game.reset()

            agent.num_games += 1

            #train long memory (replay memory) trained on all previous games
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            
            print('Game:', agent.num_games, 'Score:', score, 'Record:', record)

            #plot results
            scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            mean_scores.append(mean_score)
            plot(scores, mean_scores)

if __name__ == '__main__':
    train()