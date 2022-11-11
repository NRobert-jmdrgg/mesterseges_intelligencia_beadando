import numpy as np
from flowshop import NEH, makespan, getSchedule
import cProfile
import pstats
import io


def generateMatrix(jobs, machines):
  return np.random.randint(1, 10, size=(jobs, machines))


def swap(seq, i, j):
  seq[i], seq[j] = seq[j], seq[i]
  return seq


def insertion(seq, i, j):

  seq = np.insert(seq, i, seq[j])
  seq = np.delete(seq, j + 1)

  return seq


# def blockInsertion(seq, i, j, k):
#   temp = seq[j:k]

#   print("block: ", seq, "i: ", i, "j: ", j, "k: ", k, "temp:", temp)

#   seq = np.insert(seq, i, temp)
#   seq = np.delete(seq, range(i + j, k))

#   print("after: ", seq)
#   return seq


def getRandomIndices(n):
  return np.random.choice(np.arange(n), 3, replace=False)


def getNeighborhood(seq):
  n = len(seq)
  neighborhood = []

  for _ in range(2 * n):
    r = np.random.randint(0, 1)
    i, j = sorted(np.random.choice(n, 2))
    tmp = seq.copy()
    if r == 0:
      neighborhood.append(swap(tmp, i, j))
    elif r == 1:
      neighborhood.append(insertion(tmp, i, j))

  # print(neighborhood)
  return np.unique(neighborhood, axis=0)


def InTabuList(seq, tabuList):
  for t in tabuList:
    if np.array_equal(seq, t):
      return True
  return False


def tabuSearch(mat):
  maxIterations = 100
  tabuListSize = 7
  initialSequence = getSchedule(mat)

  tabuList = []
  best = initialSequence
  bestMakeSpan = makespan(mat, best)
  for k in range(maxIterations):
    neighborhood = getNeighborhood(best)
    for n in neighborhood:
      if not InTabuList(n, tabuList):
        nMakeSpan = makespan(mat, n)
        if nMakeSpan < bestMakeSpan:
          best = n
          tabuList.append(best)
          bestMakeSpan = nMakeSpan
          if len(tabuList) > tabuListSize:
            tabuList.pop()

  return best


# mat = np.array([
#   [5, 9, 8, 10, 1],
#   [9, 3, 10, 1, 8],
#   [9, 4, 5, 8, 6],
#   [4, 8, 8, 7, 2]
# ], dtype=np.int32)

# mat = np.array([
#   [3, 4, 6, 7],
#   [4, 5, 4, 6],
#   [8, 7, 2, 2],
#   [5, 3, 1, 5],
#   [7, 6, 8, 4]
# ])

# mat = np.array([[2, 5, 2, 6, 5],
#                 [4, 5, 5, 7, 5],
#                 [4, 9, 8, 3, 1],
#                 [8, 8, 4, 2, 9],
#                 [1, 6, 2, 1, 9],
#                 [4, 9, 7, 5, 5],
#                 [4, 8, 8, 3, 9],
#                 [7, 2, 4, 8, 4],
#                 [7, 7, 5, 9, 3],
#                 [4, 4, 2, 8, 4]], dtype=np.int32)

mat = generateMatrix(100, 20)

pr = cProfile.Profile()
pr.enable()


best = tabuSearch(mat)
print("best", best, makespan(mat, best))


print("neh_best", NEH(mat))

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
  f.write(s.getvalue())
