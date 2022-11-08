import numpy as np


def generateMatrix(min, max, size):
  return np.random.randint(min, max, size=size)


def getSortedJobIndices(mat, indexes, reverse=False):
  mins = []
  for i in indexes:
    mins.append(min([mat[i][j] + mat[i][j + 1]
                     for j in range(len(mat[0]) - 1)]))
  if not reverse:
    return np.argsort(mins)
  else:
    return np.argsort(mins)[::-1]


def calculateMakeSpan(mat):
  # calculate optimal job order
  x = []
  y = []

  n = len(mat)
  m = len(mat[0])

  for i in range(n):
    if mat[i][0] <= mat[i][-1]:
      x.append(i)
    else:
      y.append(i)

  group_x = np.array(x)
  group_y = np.array(y)

  group_x = group_x[getSortedJobIndices(mat, x)]
  group_y = group_y[getSortedJobIndices(mat, y, reverse=True)]

  jobs_order = np.concatenate((group_x, group_y))

  jobs_sorted_matrix = np.array(mat[jobs_order])

  # first row
  jobs_sorted_matrix[0] = [sum(mat[jobs_order[0]][:i]) for i in range(1, m + 1)]

  # first column
  jobs_sorted_matrix[:, 0] = [
    sum(jobs_sorted_matrix[:, 0][:i]) for i in range(1, n + 1)]

  # inner submatrix
  for i in range(1, n):
    for j in range(1, m):
      if jobs_sorted_matrix[i][j - 1] >= jobs_sorted_matrix[i - 1][j]:
        jobs_sorted_matrix[i][j] += jobs_sorted_matrix[i][j - 1]
      else:
        jobs_sorted_matrix[i][j] += jobs_sorted_matrix[i - 1][j]

  return jobs_sorted_matrix[-1][-1]


mat = np.array([
  [3, 4, 6, 7],
  [4, 5, 4, 6],
  [8, 7, 2, 2],
  [5, 3, 1, 5],
  [7, 6, 8, 4]
])

print(calculateMakeSpan(mat))
