import json
import os
import tempfile
from getpass import getpass
from typing import List, Tuple

import clip
import numpy as np
import replicate
import torch
from IPython.display import display
from ortools.linear_solver import pywraplp
from PIL import Image


def zero_rows(matrix):
    for row in matrix:
        min_val = np.min(row)
        row -= min_val


def zero_columns(matrix):
    for i in range(len(matrix[0])):
        min_val = np.min(matrix[:, i])
        matrix[:, i] -= min_val


def cover_zeros(matrix):
    covered_rows = set()
    covered_columns = set()

    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value == 0 and i not in covered_rows and j not in covered_columns:
                covered_rows.add(i)
                covered_columns.add(j)

    return covered_rows, covered_columns


def min_uncovered(matrix, covered_rows, covered_columns):
    min_val = float("inf")
    for i, row in enumerate(matrix):
        if i not in covered_rows:
            for j, value in enumerate(row):
                if j not in covered_columns:
                    min_val = min(min_val, value)

    return min_val


def adjust_matrix(matrix, covered_rows, covered_columns, min_val):
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if i not in covered_rows and j not in covered_columns:
                matrix[i][j] -= min_val
            elif i in covered_rows and j in covered_columns:
                matrix[i][j] += min_val


def hungarian_algorithm(matrix):
    zero_rows(matrix)
    zero_columns(matrix)

    while True:
        covered_rows, covered_columns = cover_zeros(matrix)
        if len(covered_rows) + len(covered_columns) >= len(matrix):
            break

        min_val = min_uncovered(matrix, covered_rows, covered_columns)
        adjust_matrix(matrix, covered_rows, covered_columns, min_val)

    return matrix


def bipartite_matching(sim_matrix):
    cost_matrix = np.max(sim_matrix) - sim_matrix
    cost_matrix = hungarian_algorithm(cost_matrix)
    matches = []

    for i, row in enumerate(cost_matrix):
        for j, value in enumerate(row):
            if value == 0:
                matches.append((i, j))
                break

    return matches
