import math
import numpy as np
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree


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
            "but '{comp}' was found").format(comp=component))

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

    arr = np.empty((rows * cols), np.int16)
    for i in range(rows * cols):
        arr[i] = block[point]
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)
    return arr


def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def run_length_encode(arr):
    # determine where the sequence is ending prematurely
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i

    # each symbol is a (RUNLENGTH, SIZE) tuple
    symbols = []

    # values are binary representations of array elements using SIZE bits
    values = []

    # return the binary representation of n using SIZE bits
    def binary_str(n):
        if n == 0:
            return ''
        s = bin(abs(n))[2:]

        # change every 0 to 1 and vice verse when n is negative
        if n < 0:
            s = ''.join(map(lambda c: '0' if c == '1' else '1', s))
        return s

    run_length = 0

    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(binary_str(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(binary_str(elem))
            run_length = 0
    return symbols, values


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def write_to_file(dc, ac, blocks_count, tables, filename='image.dat'):
    f = open('data/' + filename, 'w')

    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:

        # 16 bit for 'table_size'
        f.write(uint_to_binstr(len(tables[table_name]), 16))

        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                # 4 bits for the 'category'
                # 4 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key, 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)
            else:
                # 4 bits for 'run_length'
                # 4 bits for 'size'
                # 4 bits for 'code_length'
                # 'code_length' bits for 'huffman_code'
                f.write(uint_to_binstr(key[0], 4))
                f.write(uint_to_binstr(key[1], 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)

    for b in range(blocks_count):
        for c in range(3):
            dc_code = bits_required(dc[b, c])
            symbols, values = run_length_encode(ac[b, :, c])

            if c == 0:
                dc_table = tables['dc_y']
                ac_table = tables['ac_y']
            else:
                dc_table = tables['dc_c']
                ac_table = tables['ac_c']

            f.write(dc_table[dc_code])
            for i in range(len(symbols)):
                f.write(ac_table[tuple(symbols[i])])
                if symbols[i][1] != 0:
                    f.write(values[i])
    f.close()


def main():
    image = Image.open('data/lena.tif')
    ycbcr = image.convert('YCbCr')

    npmat = np.array(ycbcr, dtype=np.int16)

    rows, cols = npmat.shape[0], npmat.shape[1]

    # block size: 8x8
    if rows % 8 == cols % 8 == 0:
        blocks_count = rows // 8 * cols // 8
    else:
        raise ValueError(("the width and height of the image "
                          "should both be mutiples of 8"))

    # dc is the top-left cell of the block, ac are all the other cells
    dc = np.empty((blocks_count, 3), dtype=np.int16)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int16)

    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            try:
                block_index += 1
            except NameError:
                block_index = 0

            for k in range(3):
                # split 8x8 block and center the data range on zero
                # [0, 255] --> [-128, 127]
                block = npmat[i:i+8, j:j+8, k] - 128

                dct_matrix = fftpack.dct(block, norm='ortho')
                quant_matrix = quantize(dct_matrix,
                                        'lum' if k == 0 else 'chrom')
                zz = zigzag(quant_matrix)

                dc[block_index, k] = zz[0]
                ac[block_index, :, k] = zz[1:]

    def flatten(l):
        return [item for sublist in l for item in sublist]

    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    H_AC_Y = HuffmanTree(
            flatten(run_length_encode(ac[i, :, 0])[0]
                    for i in range(blocks_count)))
    H_AC_C = HuffmanTree(
            flatten(run_length_encode(ac[i, :, j])[0]
                    for i in range(blocks_count) for j in [1, 2]))

    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
              'ac_y': H_AC_Y.value_to_bitstring_table(),
              'dc_c': H_DC_C.value_to_bitstring_table(),
              'ac_c': H_AC_C.value_to_bitstring_table()}

    write_to_file(dc, ac, blocks_count, tables)


if __name__ == "__main__":
    main()
