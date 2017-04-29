import numpy as np
import math

def dct(block):
    def block_cos(u, v):
        def cos_expr(xy, uv):
            return math.cos((2 * xy + 1) * uv * math.pi / 16)

        res = 0.0
        for x in range(rows):
            for y in range(cols):
                res += block[x, y] * cos_expr(x, u) * cos_expr(y, v)
        return res

    def alpha(n):
        return 1 / math.sqrt(2) if n == 0 else 1

    rows, cols = block.shape

    # transformed matrix
    trans = np.empty((rows, cols), np.float64)

    for u in range(rows):
        for v in range(cols):
            trans[u, v] = alpha(u) * alpha(v) * block_cos(u, v) / 4

    return trans

def zigzag(block):
    # constants for directions
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    # move the point in different directions
    def move(direction, point):
        return {
            UP:        lambda point: (point[0] - 1, point[1]),
            DOWN:      lambda point: (point[0] + 1, point[1]),
            RIGHT:     lambda point: (point[0], point[1] + 1),
            LEFT:      lambda point: (point[0], point[1] - 1),
            UP_RIGHT:  lambda point: move(UP, move(RIGHT, point)),
            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point)),
        }[direction](point)

    rows, cols = block.shape

    # return true if point is inside the block bounds
    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    # start in the top-left cell
    point = (0, 0)

    # True when moving up-right, False when moving down-left
    move_up = True

    arr = np.empty((rows * cols, 1), np.int16);
    for i in range(rows * cols):
        arr[i] = block[point]
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)): point = move(RIGHT, point)
                else: move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)): point = move(DOWN, point)
                else: move(RIGHT, point)
    return arr
