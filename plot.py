from plotly.figure_factory import create_gantt
import pandas as pd
import numpy as np


def getGanttIntervals(mat, job_seq):

  n, m = len(mat), len(mat[0])
  interval_beginnings = np.zeros((n, m), dtype=np.int32)
  interval_endings = np.zeros((n, m), dtype=np.int32)

  interval_beginnings[0][0] = 0
  interval_endings[0][0] = mat[job_seq[0]][0]

  for j in range(1, m):
    interval_endings[0][j] = interval_endings[0][j - 1] + mat[job_seq[0]][j]
    interval_beginnings[0][j] = interval_endings[0][j - 1]

  for i in range(1, n):
    interval_endings[i][0] = interval_endings[i - 1][0] + mat[job_seq[i]][0]
    interval_beginnings[i][0] = interval_endings[i - 1][0]

  for i in range(1, n):
    for j in range(1, m):
      interval_endings[i][j] = max(
        interval_endings[i][j - 1], interval_endings[i - 1][j]) + mat[job_seq[i]][j]
      interval_beginnings[i][j] = max(
        interval_endings[i][j - 1], interval_endings[i - 1][j])

  return interval_beginnings, interval_endings


# mat = [
#   [3, 4, 6, 7],
#   [4, 5, 4, 6],
#   [8, 7, 2, 2],
#   [5, 3, 1, 5],
#   [7, 6, 8, 4]
# ]

# seq = [3, 0, 1, 4, 2]


def plot_flowshop(mat, seq):
  begin, end = getGanttIntervals(mat, seq)

  df = pd.DataFrame([
    dict(Start=begin[i][j], Finish=end[i][j], Task=seq[i]) for j in range(len(mat[0])) for i in range(len(mat))
  ])

  fig = create_gantt(df, group_tasks=True)

  fig.update_xaxes(type='linear')

  fig.show()
