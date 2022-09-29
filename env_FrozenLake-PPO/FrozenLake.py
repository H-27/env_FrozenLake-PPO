import numpy as np
class FrozenLake(object):
    def __init__(self, y, x, sparse = True):
        self.sparse = sparse
        self.y = y+2
        self.x = x+2
        self.start = (1,1)
        self.goal = (x,y)
        self.position = (1,1)
        self.actions = (0,1,2,3)
        self.map = self.reset_map()
        
        
    
    def reset_map(self):
        map = np.chararray((self.y,self.x))
        map[:] = '-'
        map[0] = 'x'
        map[self.y-1] = 'x'
        map[:,0] = 'x'
        map[:,-1] = 'x'
        map[self.start] = 'S'
        map[self.goal] = 'G'
        return map
        

    def reset(self):
        self.position = self.start
    
    def step(self, action):
        
        if self.sparse:
            next_position = self.next_pos(action)
            if self.map[next_position] == b'x':
                return (self.map, self.position), 0, False
            if self.map[next_position] == b'H':
                self.reset()
                return (self.map, self.position), 0, True
            if self.map[next_position] == b'G':
                self.position = next_position
                return (self.map, self.position), 1, True
            if self.map[next_position] == b'S':
                self.position = next_position
                return (self.map, self.position), 0, False
            if self.map[next_position] == b'-':
                self.position = next_position
                return (self.map, self.position), 0, False

        if not self.sparse:
            next_position = self.next_pos(action)
            if self.map[next_position] == b'x':
                return self.position, -2
            if self.map[next_position] == b'H':
                self.reset()
                return next_position, -10
            if self.map[next_position] == b'G':
                self.position = next_position
                return next_position, 100
            if self.map[next_position] == b'S':
                self.position = next_position
                return self.next_pos(action), -1
            if self.map[next_position] == b'-':
                self.position = next_position
                return self.next_pos(action), -1
    
    def next_pos(self, action):
        # 0: LEFT 1: DOWN 2: RIGHT 3: UP
        next_position = ()
        if action == 0:
            next_position = (self.position[0], self.position[1] - 1)
            return next_position
        elif action == 1:
            next_position = (self.position[0] + 1, self.position[1])
            return next_position
        elif action == 2:
            next_position = (self.position[0], self.position[1] + 1)
            return next_position
        elif action == 3:
            next_position = (self.position[0] - 1, self.position[1])
            return next_position
        else:
            print('No valid action')
            raise ValueError
    
    def one_hot(self, map):
        one_hot_map = np.zeros((self.y,self.x, 4, 1))
        for i in range(1, len(map)):
            for j in range(1, len(map)):
                if map[i][j] == b'G':
                    one_hot_map[i][j] = np.array([[0], [0], [0], [0], [1]])
                if map[i][j] == b'H':
                    one_hot_map[i][j] = np.array([[0], [0], [0], [1], [0]])
                if map[i][j] == b'x':
                    one_hot_map[i][j] = np.array([[0], [0], [1], [0], [0]])
                if map[i][j] == b'-' or b'S':
                    one_hot_map[i][j] = np.array([[0], [1], [0], [0], [0]])
                if map[i][j] == b'P':
                    one_hot_map[i][j] = np.array([[1], [0], [0], [0], [0]])
        return one_hot_map
        

    def visualise(self):
      
        self.map[self.position] = 'P'
        print(self.one_hot(self.map))