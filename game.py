import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import colorsys
pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')
RAINBOW_COLORS = [(148, 0, 211), (75, 0, 130), (0, 0, 255), (0, 255, 0), (255, 255, 0), (255, 127, 0), (255, 0, 0)]

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 99999999
GEN=0
class SnakeGameAI():
    
    def __init__(self, w=800, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.gen =0 
        self.reset()

        # init game state

    def reset(self):
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.gen += 1
        print(self.gen)

        self.food = None
        self._place_food()   
        self.frame_iteration=0     

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        # 1. collect user input
        self.frame_iteration=+1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
 
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        reward = 0
        # 3. check if game over
        game_over = False
        if self.is_collision() or self.frame_iteration>100*len(self.snake):
            game_over = True
            reward = -10
            return reward,game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward=+10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward,game_over, self.score
    
    def is_collision(self,pt=None):
        if pt is None:
            pt=self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)

        # Calculate the hue step based on the length of the snake
        hue_step = 360 / len(self.snake)

        for i, pt in enumerate(self.snake):
            # Calculate hue based on the index of the segment
            hue = (i * hue_step) % 360

            # Convert HSL to RGB
            rgb = colorsys.hsv_to_rgb(hue / 360, 1.0, 1.0)
            rainbow_color = (rgb[0] * 255, rgb[1] * 255, rgb[2] * 255)

            # Draw snake body with rainbow color
            pygame.draw.rect(self.display, rainbow_color, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, rainbow_color, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
       # text2=font.render("GEN"+str(GEN.copy()),True,WHITE)
        self.display.blit(text,[0, 0])
       # self.display.blit(text2,[0, 20])
        pygame.display.flip()
        
    def _move(self, action):
        clkwise=[Direction.RIGHT,Direction.DOWN,Direction.LEFT,Direction.UP]
        idx = clkwise.index(self.direction)
        if np.array_equal(action,[1,0,0]):
            new_dir=clkwise[idx]
        elif np.array_equal(action,[0,1,0]):
            next_idx=(idx+1)%4
            new_dir=clkwise[next_idx]
        else:    
            next_idx=(idx-1)%4
            new_dir=clkwise[idx]
        self.direction=new_dir    
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            

