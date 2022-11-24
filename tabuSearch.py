import numpy as np
from flowshop import NEH, makespan, hybrid
import cProfile
import pstats
import io
from plot import plot_flowshop


def generateMatrix(jobs, machines, seed):
  """Generate n x m matrix

  Args:
      jobs (number): number of jobs
      machines (number): number of machines
      seed (number): seed

  Returns:
      number[][]: matrix
  """
  np.random.seed(seed)
  return np.random.randint(1, 10, size=(jobs, machines))


def swap(seq, i, j):
  """swap two items in a list

  Args:
      seq (number[]): list
      i (number): first index
      j (number): second index

  Returns:
      number[]: new list
  """
  seq[i], seq[j] = seq[j], seq[i]
  return seq


def insertion(seq, i, j):
  """insert the jth item at index i

  Args:
      seq (number[]): list
      i (number): index
      j (number): index

  Returns:
      number[]: new list
  """

  k = seq[j]
  tmp = seq.copy()
  tmp = list(filter(lambda x: x != k, seq))
  tmp.insert(i, k)

  return tmp


def getRandomIndices(n):
  """ get two random numbers from range 0..n-1

  Args:
      n (number): upper bound

  Returns:
      number[]: two random numbers
  """
  return np.random.choice(np.arange(n), 2, replace=False)


def getNeighborhoodRamdom(seq):
  """generate neighborhood of a list by randomly swapping and inserting elements
  in the list

  Args:
      seq (number[]): job sequence

  Returns:
      number[][]: neighborhood of sequence
  """
  n = len(seq)
  neighborhood = []

  for _ in range(2 * n):
    r = np.random.randint(0, 1)
    i, j = np.random.choice(n, 2)
    tmp = seq.copy()
    if r == 0:
      neighborhood.append(swap(tmp, i, j))
    elif r == 1:
      neighborhood.append(insertion(tmp, i, j))

  return np.unique(neighborhood, axis=0)


def getNeighborhoodBySwap(seq):
  n = len(seq)
  neighborhood = []
  for i in range(n - 1):
    tmp = seq.copy()
    neighborhood.append(swap(tmp, i, i + 1))

  return neighborhood


def getNeighborhoodByInsertion(seq):
  n = len(seq)
  neighborhood = []
  for i in range(n):
    for j in range(n):
      if i != j:
        tmp = seq.copy()
        neighborhood.append(insertion(tmp, j, i))

  return neighborhood


def InTabuList(seq, tabuList):
  """Check if seq is in the tabu list

  Args:
      seq (number[]): sequence
      tabuList (number[][]): tabu list

  Returns:
      boolean: true if the sequence is in the tabu list, false otherwise
  """
  for t in tabuList:
    if np.array_equal(seq, t):
      return True
  return False


mat = generateMatrix(10, 5, 123)
# mat = generateMatrix(10, 10, 123)
# mat = generateMatrix(10, 20, 123)
# mat = generateMatrix(20, 5, 123)
# mat = generateMatrix(20, 10, 123)
# mat = generateMatrix(20, 20, 123)
# mat = generateMatrix(50, 5, 123)
# mat = generateMatrix(50, 10, 123)
# mat = generateMatrix(50, 20, 123)
# mat = generateMatrix(100, 5, 123)
# mat = generateMatrix(100, 10, 123)
# mat = generateMatrix(100, 20, 123)
# mat = generateMatrix(200, 5, 123)
# mat = generateMatrix(200, 10, 123)
# mat = generateMatrix(200, 20, 123)
# mat = generateMatrix(500, 10, 123)
# mat = generateMatrix(500, 20, 123)
# mat = generateMatrix(500, 20, 123)


def tabuSearch(mat, getSchedule, tabuListSize, getNeighborhood, maxNoImprovementCount, plot=False):
  """flow shop tabu search

  Args:
      mat (number[][]): flow shop matrix

  Returns:
      sequence: best job sequence
  """
  initialSequence = getSchedule(mat)

  tabuList = []
  best = initialSequence
  bestMakeSpan = makespan(mat, best)
  noImprovementCounter = 0

  while True:
    improved = False
    print(noImprovementCounter)
    if noImprovementCounter == maxNoImprovementCount:
      break
    neighborhood = getNeighborhood(best)
    for n in neighborhood:
      if not InTabuList(n, tabuList):
        nMakeSpan = makespan(mat, n)
        if nMakeSpan < bestMakeSpan:
          improved = True
          best = n
          tabuList.append(best)
          bestMakeSpan = nMakeSpan
          print("new min: ", bestMakeSpan, " tabu list size: ", len(tabuList))
          if len(tabuList) == tabuListSize:
            tabuList.pop()

    if improved:
      noImprovementCounter = 0
    else:
      noImprovementCounter += 1

  if plot:
    plot_flowshop(mat, best)
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


# print(getNeighborhoodBySwap([0, 1, 2, 3]))
# print(getNeighborhoodByInsertion([0, 1, 2, 3]))

pr = cProfile.Profile()
pr.enable()


best = tabuSearch(mat, hybrid, 7, getNeighborhoodByInsertion, 10, plot=True)
print("best", best, makespan(mat, best))


pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test.txt', 'w+') as f:
  f.write(s.getvalue())
