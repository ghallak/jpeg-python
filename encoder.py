import numpy as np

def zigzag(block):
    # move the point to different directions
    up    = lambda point: (point[0] - 1, point[1])
    down  = lambda point: (point[0] + 1, point[1])
    right = lambda point: (point[0], point[1] + 1)
    left  = lambda point: (point[0], point[1] - 1)

    up_right  = lambda point: up(right(point))
    down_left = lambda point: down(left(point))

    rows, cols = block.shape

    # return true if point is inside the block bounds
    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    # start in the top-left cell
    point = (0, 0)

    # TODO: rename this variable later
    even = True

    arr = np.empty((rows * cols, 1), np.int16);
    for i in range(rows * cols):
        arr[i] = block[point]
        if even:
            if inbounds(up_right(point)):
                point = up_right(point)
            else:
                even = False
                point = right(point) if inbounds(right(point)) else down(point)
        else:
            if inbounds(down_left(point)):
                point = down_left(point)
            else:
                even = True
                point = down(point) if inbounds(down(point)) else right(point)
    return arr
