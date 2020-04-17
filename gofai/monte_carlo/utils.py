import random

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __add__(self, other):
        return Vector(self.x+other.x, self.y+other.y)
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
        
    def __hash__(self):
        return hash((self.x, self.y))
    
    def copy(self):
        return Vector(self.x, self.y)
    
    def __str__(self):
        return "<{}, {}>".format(self.x, self.y)
    
    def __repr__(self):
        return str(self)

def rand_vec(w, h):
    x = random.randint(0, w-1)
    y = random.randint(0, h-1)
    return Vector(x, y)