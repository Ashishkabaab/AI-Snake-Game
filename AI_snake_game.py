import pygame
import random
import numpy as np
from enum import Enum
from collections import namedtuple

#RGB Colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
DARKGRAY = pygame.Color("azure4")
LIGHTGRAY = pygame.Color("azure3")

FONT_SIZE = 25
BLOCK_SIZE = 20
SMALL_BLOCK_SIZE = 12
ADJUST = 4
WIDTH = 640
HEIGHT = 480
# frame rate
SPEED = 40

pygame.init()
font = pygame.font.SysFont('Arial.ttf', FONT_SIZE)

class Direction(Enum):
    # assigning an associated integer value to each direction
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

# namedtuple format used to access x and y components of items
# in the display
Point = namedtuple('Point', 'x, y')

#agent controlled game
class SnakeGameAI:

    def __init__(self, width=WIDTH, height=HEIGHT):
        self.width = width
        self.height = height

        #initialize display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()
    
    def reset(self):
        #reset/initialize game state
        #place initial snake, initial food, initial direction
        self.direction = Direction.RIGHT

        #position of head, starting in middle of display
        self.head = Point(self.width/2, self.height/2)

        #Snake. Initially 3 coordinates
        self.snake = [self.head, 
                      Point(self.head.x - BLOCK_SIZE, self.head.y), 
                      Point(self.head.x - (2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None

        self._place_food()

        #track number of moves in a game. #Helpful for ending game if
        #snake is stalling (not dying and not eating food)
        self.frame_iteration = 0
    
    #place food at random position on screen, multiples of block size
    def _place_food(self):
        x = random.randint(0, (self.width-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x,y)

        #don't want to place food inside snake, check if inside list
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        self.frame_iteration += 1
        #Check event handler to see if game was quit. End game if yes
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        #Move snake
        self._move(action) #update head
        self.snake.insert(0, self.head) #insert new head at beginning of list

        #Check if game is over. Hit boundary or snake itself
        # Or if agent is taking too long (not eating food and not dying)
        # Time limit dependent on length of snake
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        #Place new food if eaten, or finalize move step
        #if food not eaten, remove last element of snake list
        #because we insert each time we move. If food eaten, don't
        #remove and allow snake to grow
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        #Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score
    
    def is_collision(self, point=None):
        if point is None:
            point = self.head
        #hits boundary
        if (point.x > self.width - BLOCK_SIZE) or (point.x < 0) or (point.y > self.height - BLOCK_SIZE) or (point.y < 0):
            return True
        #hits itself, check all positions other than head
        if point in self.snake[1:]:
            return True
        
        return False

    def _update_ui(self):
        #fill screen black
        self.display.fill(BLACK)

        #draw snake: color, shape, location, size
        for point in self.snake:
            pygame.draw.rect(self.display, DARKGRAY, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, LIGHTGRAY, pygame.Rect(point.x + ADJUST, point.y + ADJUST, SMALL_BLOCK_SIZE, SMALL_BLOCK_SIZE))

        #draw food: color, shape, location, size
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        #Display score in upper left of screen
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0,0])
        pygame.display.flip()
    
    # use one hot encoded action [straight, right, left]
    def _move(self, action):

        #directions in clockwise order
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

        #current direction/index
        current_index = clock_wise.index(self.direction)

        #straight action. Maintain current direction
        if np.array_equal(action, [1, 0, 0]):
            new_direction = clock_wise[current_index]

        #right action. Change to next clockwise direction
        elif np.array_equal(action, [0, 1, 0]):
            next_index = (current_index + 1) % 4
            new_direction = clock_wise[next_index]

        #left action. Change to previous clockwise direction
        elif np.array_equal(action, [0, 0, 1]):
            next_index = (current_index - 1) % 4
            new_direction = clock_wise[next_index]
        
        self.direction = new_direction

        x = self.head.x
        y = self.head.y

        #check Enum direction, update head
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        
        self.head = Point(x,y)