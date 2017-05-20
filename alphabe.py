import copy

def procstep(grid,color_num,action):
    x0 = action['x0']
    y0 = action['y0']
    x1 = action['x1']
    y1 = action['y1']
    grid[x1][y1] = color_num
    if abs(x0-x1)>1 or abs(y0-y1)>1:
        grid[x0][y0] = 0
    for i in range(-1,2):
        for j in range(-1,2):
            if (i==0 and j==0) or x1+i<0 or x1+i>6 or y1+j<0 or y1+j>6:
                continue
            if grid[x1+i][y1+j] == -color_num:
                grid[x1+i][y1+j] = color_num


def generateMove(grid,color_num):
    moves = []
    for i in range(7):
        for j in range(7):
            if grid[i][j] == 0:
                for c in range(-2,3):
                    for r in range(-2,3):
                        if (c==0 and r==0) or i+c<0 or i+c>6 or j+r<0 or j+r>6:
                            continue
                        if grid[i+c][j+r] == color_num:
                            mov = {}
                            mov['x0'] = i+c
                            mov['y0'] = j+r
                            mov['x1'] = i
                            mov['y1'] = j
                            moves.append(mov)
    return moves

def evaluate(grid,color_num):
    pos_num,neg_num = 0,0
    for i in range(7):
        for j in range(7):
            if grid[i][j] == color_num:
                pos_num += 1
            elif grid[i][j] == -color_num:
                neg_num += 1
            else:
                continue
    return pos_num-neg_num



def AlphaBeta(grid,depth,alpha,beta,color_num):
    if depth == 0:
        return evaluate(grid,color_num)
    moves = generateMove(grid,color_num)
    if len(moves) == 0:
        return evaluate(grid,color_num)
    for move in moves:
        new_grid = copy.deepcopy(grid)
        procstep(new_grid,color_num,move)
        value = -AlphaBeta(new_grid,depth-1,-beta,-alpha,-color_num)
        if value >= beta:
            return beta
        if value > alpha:
            alpha = value
    return alpha

def search(grid,depth,alpha,beta,color_num):
    best_move = {}
    moves = generateMove(grid,color_num)
    for move in moves:
        new_grid = copy.deepcopy(grid)
        procstep(new_grid,color_num,move)
        value = -AlphaBeta(new_grid,depth-1,-beta,-alpha,-color_num)
        if value > alpha:
            alpha = value
            best_move = move
    return best_move
