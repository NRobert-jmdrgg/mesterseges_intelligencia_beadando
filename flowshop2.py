import numpy as np


def makespan(mat, job_seq):
  """calculate the makespan of a flow shop by given job order

  Args:
      mat (number[][]): n x m matrix
      job_seq (number[]): n list the order of jobs

  Returns:
      makespan (number): the time to complete all jobs
  """
  n, m = len(mat), len(mat[0])
  costs = np.zeros((n, m), dtype=np.int32)

  # calculate the first row of the cost matrix
  costs[0][0] = mat[job_seq[0]][0]
  for j in range(1, m):
    costs[0][j] = costs[0][j - 1] + mat[job_seq[0]][j]

  # first column of cost matrix
  for i in range(1, n):
    costs[i][0] = costs[i - 1][0] + mat[i][job_seq[0]]

  # the rest of the cost matrix
  for i in range(1, n):
    for j in range(1, m):
      if costs[i][j - 1] >= costs[i - 1][j]:
        costs[i][j] = costs[i][j - 1] + mat[job_seq[i]][j]
      else:
        costs[i][j] = costs[i - 1][j] + mat[job_seq[i]][j]

  # get the makespan
  print(costs)

  return costs[-1][-1]


def initialJobOrder(mat):
  """get the initial job order of a flow shop problem by summing up the
  processing times of each job and ordering them in descending order

  Args:
      mat (number[][]): flow shop matrix

  Returns:
      number[]: job sequence indexes
  """
  processTimes = np.zeros(len(mat), dtype=np.int32)

  # get the process time of each job
  for i in range(len(mat)):
    processTimes[i] = sum(mat[i])

  # get the job order by descending processing time order
  return np.argsort(processTimes)[::-1]


def insertAtOptimalPosition(mat, job_seq, new_job):
  """insert the k-th job at the position which minimizes makespan

  Args:
      mat (number[][]): flow shop matrix
      job_seq (number[]): current job sequence
      new_job (number): next job
  """
  n = len(job_seq) + 1
  minMakeSpan = np.inf
  minIdx = len(job_seq)

  # check next job at each position
  for i in range(n):
    tmp = job_seq.copy()
    tmp.insert(i, new_job)
    # calculate submatrix makespan
    ms = makespan(mat[tmp], np.arange(n))
    if ms < minMakeSpan:
      minMakeSpan = ms
      minIdx = i

  # insert new job at optimal position
  job_seq.insert(minIdx, new_job)


def NEH(mat):
  """Nawaz, Enscore and Ham algorithm get the minimal job order for a given 
     flow shop matrix

     Args:
         mat (number[][]): flow shop matrix
     Returns:
         number: minimum makespan
  """

  # step 1 Order the n jobs by decreasing total processing time on the machines
  seq = initialJobOrder(mat)  # ok

  # step 2 and 3 Insert the k-th job at the position which minimizes makespan
  # among the k possible positions available
  optimal_order = [seq[0]]
  for i in seq[1:]:
    insertAtOptimalPosition(mat, optimal_order, i)

  # get minimum makespan
  return makespan(mat, optimal_order)


mat = np.array([
  [5, 9, 8, 10, 1],
  [9, 3, 10, 1, 8],
  [9, 4, 5, 8, 6],
  [4, 8, 8, 7, 2]
], dtype=np.int32)

# makespan(mat, [0, 1, 2, 3])
print(NEH(mat))
