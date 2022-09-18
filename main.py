
import numpy as np


def normalize(matrix):
    if matrix[0][0] != 1:
        for row in range(matrix.shape[0]):
            if matrix[row][0] == 1:
                matrix[[0, row]] = matrix[[row, 0]]
                break
    else:
        matrix[0] /= matrix[0, 0]
    return matrix


def simplify(matrix):
    for i in range(1, matrix.shape[0]):
        for j in range(i, matrix.shape[0]):
            matrix[j] -= matrix[i - 1] * matrix[j][i - 1]
        matrix[i] /= matrix[i][i]
    return matrix


def find_roots_of(matrix):
    roots = list()
    roots.append(matrix[matrix.shape[0] - 1, matrix.shape[0]])

    for i in range(matrix.shape[0], 0, -1):
        for j in range(0, matrix.shape[0]):
            roots.append(matrix[i - 1][i] - matrix[i - 1][i - 1] * roots[j])

    #x3 = matrix[2][3] - x3 * matrix[2][2] - x2 * matrix[2][1]
    #x2 = matrix[1][3] - x3 * matrix[1][2] - x2 * matrix[1][1]
    #x1 = matrix[0][3] - x3 * matrix[0][2] - x2 * matrix[0][1]
    print(roots)


def find_x(i):
    return matrix1[i - 1][i] - matrix1[i - 1][i - 1] * find_x(i - 1)


matrix1 = np.array([[3, 2, -5, -1],
                   [2, -1, 3, 13],
                   [1, 2, -1, 9]], dtype=float)

print(matrix1, end="\n\n")


matrix1 = normalize(matrix1)
print(matrix1, end="\n\n")
matrix1 = simplify(matrix1)
print(matrix1)
find_roots_of(matrix1)



#print(roots1)

