class Game:
    '''states, actions, players, terminals with values
    - immutable
    - states, actions, and players should be hashable
    - actions may be non-deterministic
    - non-terminal states may have values
    '''
    def __init__(self):
        pass

    def get_legal_actions(self):
        '''returns [(player, action)]
        representing the legal actions in the current state
        '''
        raise NotImplementedError

    def is_legal_action(self, player, action):
        return (player, action) in self.get_legal_actions()
    
    def get_state(self):
        '''returns the current game state
        '''
        raise NotImplementedError

    def do_action(self, player, action):
        '''performs the action and returns the new state
        DOES NOT MUTATE THIS GAME OBJECT
        may raise error if action is not legal
        '''
        raise NotImplementedError

    def is_terminal(self):
        '''is the current state a terminal?
        '''
        raise NotImplementedError

    def value(self):
        '''return the value of the current state
        may throw error if non-terminal
        '''
        raise NotImplementedError

    def __str__(self):
        return str(self.get_state())

    def __repr__(self):
        return str(self)