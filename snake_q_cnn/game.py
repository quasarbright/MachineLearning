import random
from graphics import *
import keyboard
import time
import torch
from utils import *

class Vector:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
    
    def add(self, other):
        self.x += other.x
        self.y += other.y
    
    def copy(self):
        return Vector(self.x, self.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        if type(self) is not type(other):
            raise TypeError(str(type(other)))
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        if type(self) is not type(other):
            raise TypeError(str(type(other)))
        return Vector(self.x - other.x, self.y - other.y)

    def __str__(self):
        return '<{}, {}>'.format(self.x, self.y)
    
    def __repr__(self):
        return str(self)

class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        startingHead = Vector(random.randint(0,width//2), random.randint(0, height-1))
        self.tail = [startingHead] # list of vectors, from tail to head
        self.tailLength = len(self.tail)
        self.fruitPos = Vector()
        self.spawn_fruit()
        self.direction = 0 # can be 0, 1, 2, 3
        self.dead = False
        self.age = 0
        self.score = 0 # fruits eaten

    def spawn_fruit(self):
        self.fruitPos = Vector(random.randint(0, self.width-1), random.randint(0, self.height-1))
        while self.fruitPos in self.tail:
            self.fruitPos = Vector(random.randint(0, self.width-1), random.randint(0, self.height-1))

    def is_in_bounds(self, p):
        '''
        is vector p in bounds?
        '''
        return p.x >=0 and p.x < self.width and p.y >= 0 and p.y < self.height

    def is_eating_tail(self, p):
        '''
        is vector somewhere in the tail?
        '''
        return p in self.tail[0:-1]

    def return_state(self):
        '''
        return representation as a 1D list: distance from head to nearest object in all
        4 directions, and x and y distance to the fruit
        '''
        grid = torch.zeros((self.height, self.width)).to(device)
        for seg in self.tail:
            grid[seg.y, seg.x] = 100
        grid[seg.y, seg.x] = 200
        grid[self.fruitPos.y, self.fruitPos.x] = 300
        return grid


    def reward(self, oldHead, oldFruit, newHead, newFruit, ate, died):
        '''
        calculates the reward for a movement given where the head
        was and where the head moved to
        '''

        if died:
            return -10
        elif ate:
            return 10
        else:
            return 0
            oldDistance = abs(oldHead.x - oldFruit.x) + abs(oldHead.y - oldFruit.y)
            newDistance = abs(newHead.x - newFruit.x) + abs(newHead.y - newFruit.y)
            return oldDistance - newDistance

    def move_player(self, newDirection):
        '''
        direction is 0, 1, 2, or 3, or 4
        nothing, right, up, left, down
        '''
        # assert newDirection in [0,1,2,3,4]
        if not self.dead:
            # save snake's head vector
            head = self.tail[-1]

            # no change: make newDirection the previous direction
            if newDirection == 4:
                newDirection = self.direction

            # go right, up, left, or down
            newPosition = Vector(-1,-1)
            if newDirection == 0:
                newPosition = Vector(head.x + 1, head.y)
            elif newDirection == 1:
                newPosition = Vector(head.x, head.y - 1)
            elif newDirection == 2:
                newPosition = Vector(head.x - 1, head.y)
            elif newDirection == 3:
                newPosition = Vector(head.x, head.y + 1)
            # only move the snake if it's not about to go out of bounds
            if self.is_in_bounds(newPosition) and not self.is_eating_tail(newPosition):
                self.tail.append(newPosition)
            else:
                self.dead = True
            # update direction
            self.direction = newDirection

            # remove end of tail if didn't eat
            if len(self.tail) > self.tailLength and not self.dead:
                self.tail.pop(0)

            # spawn a new fruit if it ate
            ate = self.tail[-1] == self.fruitPos
            oldFruit = self.fruitPos.copy()
            # increment tail length if it ate
            if ate:
                self.tailLength += 1
                self.spawn_fruit()
                self.score += 1
            self.age += 1

            # check if you've died
            self.dead = self.dead or self.is_eating_tail(self.tail[-1]) or not self.is_in_bounds(self.tail[-1])

            # calculate reward
            reward = self.reward(head, oldFruit, self.tail[-1], self.fruitPos, ate, self.dead)

            # check state
            state = self.return_state()

            return reward, state, self.dead
    
    def copy(self):
        ans = Game(self.width, self.height)
        ans.width = self.width
        ans.height = self.height
        ans.tail = [v.copy() for v in self.tail]
        ans.tailLength = self.tailLength
        ans.fruitPos = self.fruitPos.copy()
        ans.direction = self.direction
        ans.dead = self.dead
        ans.age = self.age
        ans.score = self.score
        return ans
    
    def check_move(self, newDirection):
        g = self.copy()
        return g.move_player(newDirection)
    
    def best_move(self):
        def key(newDirection):
            reward, state, dead = self.check_move(newDirection)
            return reward
        return max(range(5), key=key)

    def status(self):
        return 'age: {}, score: {}'.format(self.age, self.score)
class VisibleGame(Game):
    '''
    a game that is rendered on screen
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_width = 500
        self.window_height = 500
        self.win = GraphWin("Snake", self.window_width, self.window_height)
        self.rect_width = self.window_width / self.width
        self.rect_height = self.window_height / self.height

    def draw_rectangle_at(self, r, c, color):
        p1 = Point(c*self.rect_width, r*self.rect_height)
        p2 = Point((c+1)*self.rect_width, (r+1)*self.rect_height)
        r = Rectangle(p1, p2)
        r.setFill(color)
        r.setWidth(1)
        r.setOutline('black')
        r.draw(self.win)

    def draw_background(self):
        b = Rectangle(Point(0,0), Point(self.window_width, self.window_height))
        b.setFill('gray')
        b.draw(self.win)
        # for r in range(self.height):
        #     for c in range(self.width):
        #         self.draw_rectangle_at(r, c, 'gray')

    def draw_tail(self):
        for position in self.tail:
            c = position.x
            r = position.y
            self.draw_rectangle_at(r, c, 'white')

    def draw_fruit(self):
        c = self.fruitPos.x
        r = self.fruitPos.y
        self.draw_rectangle_at(r, c, 'red')

    def draw(self):
        self.draw_background()
        self.draw_tail()
        self.draw_fruit()

    def close(self):
        self.win.close()

class PlayableGame(VisibleGame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        keyboard.on_press(self.key_listener)
        self.keyQueue = []

    def key_listener(self, ke):
        self.keyQueue.append(ke.name)
    
    def handle_keypress(self, name):
        if name == 'd' or name == 'right':
            self.move_player(0)
        elif name == 'w' or name == 'up':
            self.move_player(1)
        elif name ==  'a' or name == 'left':
            self.move_player(2)
        elif name == 's' or name == 'down':
            self.move_player(3)
        else:
            self.move_player(4)

    def update(self):
        if self.keyQueue == []:
            self.handle_keypress('')
        for i in range(len(self.keyQueue)):
            self.handle_keypress(self.keyQueue.pop(0))
        self.draw()

if __name__ == '__main__':
    vg = PlayableGame(10,10)
    # vg.draw()
    # vg.win.getMouse()
    # vg.move_player(0)
    # vg.close()
    for t in range(1000):
        # move = random.randint(0,4)
        # vg.move_player(move)
        # vg.draw()
        vg.update()
        time.sleep(1/5)
