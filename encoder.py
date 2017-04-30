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

def quantize(block, component):
    # quantization tables are taken from the original JPEG standard
    # (https://www.w3.org/Graphics/JPEG/itu-t81.pdf) page: 143
    if component == 'lum':
        q = np.array([[16, 11, 10, 16, 24,  40,  51,  61 ],
                      [12, 12, 14, 19, 26,  58,  60,  55 ],
                      [14, 13, 16, 24, 40,  57,  69,  56 ],
                      [14, 17, 22, 29, 51,  87,  80,  62 ],
                      [18, 22, 37, 56, 68,  109, 103, 77 ],
                      [24, 35, 55, 64, 81,  104, 113, 92 ],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99 ]])
    elif component == 'chrom':
        q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                      [18, 21, 26, 66, 99, 99, 99, 99],
                      [24, 26, 56, 99, 99, 99, 99, 99],
                      [47, 66, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99]])
    else:
        raise ValueError((
            "component should be either 'lum' or 'chrom', "
            "but '{comp}' was found").format(comp = component))

    return (block / q).round().astype(np.int8)

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

    arr = np.empty((rows * cols), np.int16);
    for i in range(rows * cols):
        arr[i] = block[point]
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)): point = move(RIGHT, point)
                else: point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)): point = move(DOWN, point)
                else: point = move(RIGHT, point)
    return arr

def run_length_encode(arr):
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a (RUNLENGTH, SIZE) tuple
    symbols = []

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = int(math.log2(abs(elem))) + 1 if elem != 0 else 0
            symbols.append((run_length, size))
            run_length = 0
    return symbols
