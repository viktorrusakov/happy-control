import pickle
import sympy as sym
import numpy as np
from functools import reduce
from itertools import groupby


def lie_bracket(element_1, element_2):
    """
    Unfolds a Lie bracket. It is assumed that the second element is homogeneous (the bracket grows to the left).
    Returns a string encoding the result of unfolding: each addend is represented as a sequence of indeces (which
    are separated by '.'), the addends are separated by '|' and there is also a sign before each addend (it is assumed
    to be '+' if there is no sign).

    Example 1:
        lie_bracket('1.2', '3') = [[xi_{1}, xi_{2}], xi_{3}] = '1.2.3|-2.1.3|-3.1.2|3.2.1', where

        '1.2.3|-2.1.3|-3.1.2|3.2.1' = xi_{123} - xi_{213} - xi_{312} + xi_{321}

    Example 2:
        lie_bracket('1.0|-1.1', '3') = [xi_{10} - xi_{11}, xi_3] = '1.0.3|-1.1.3|-3.1.0|3.1.1'
    """

    if '.' not in element_1:
        # if the first element is homogeneous we know the result already

        return '|'.join([element_1 + '.' + element_2, '-' + element_2 + '.' + element_1])

    elif '|' not in element_1:
        # if the first element is another Lie bracket we need to unfold it first

        element_1 = element_1.split('.')
        element_1 = lie_bracket(element_1[0], element_1[1])

    moments = element_1.split('|')
    first_addend = [m + '.' + element_2 for m in moments]
    moments = [m.replace('-', '') if m.startswith('-') else '-' + m for m in moments]
    second_addend = ['-' + element_2 + '.' + m[1:] if m.startswith('-') else element_2 + '.' + m for m in moments]
    res = '|'.join(first_addend + second_addend)
    return res


def unfold_lie_bracket(lie_element):
    """
    Generalization of lie_bracket function - unfolds brackets with nested brackets.
    """

    if len(lie_element) in [1, 2]:
        return lie_element
    elif ']' not in lie_element:
        moment_1, moment_2 = lie_element.split('.')
        return lie_bracket(moment_1, moment_2)
    else:
        bracket = lie_element.split(']')
        res = reduce(lambda x, y: lie_bracket(x, y), bracket)
        return res


def calculate_lie_elements(max_order):
    """
    Calculates Lie elements up to max_order (without including Jacobi identity). Returns a dictionary where key
    represents order and value is a dictionary where key is an encoded Lie element and value is its representation
    in R^p (as numpy array).

    Example:
        max_order = 3
        res = calculate_lie_elements(max_order) =>

        res = {
            '1': {
              '0': np.array([[1]])
            }
            '2': {
              '1': np.array([[1], [0]])
            }
            '3': {
              '2': np.array([[1], [0], [0], [0]]),
              '0.1': np.array([[0], [1], [-1], [0]]),
              '1.0': np.array([[0], [-1], [1], [0]])
            }
        }

        Lie element encoding example:
            1) '1.2' = [xi_{1}, xi_{2}]
            2) '1.2]3]5' = [[[xi_{1}, xi_{2}], xi_{3}], xi_{5}]
    """

    res = {}
    with open('api/moments_grading.pickle', 'rb') as f:
        moments = pickle.load(f)
    for order in range(1, max_order + 1):
        order_moments = moments[order]
        dim = len(order_moments)
        lie_elements = {}
        for index_set in order_moments.keys():
            if '.' not in index_set:
                # homogeneous element can be added already
                lie_elements[index_set] = order_moments[index_set]
                continue
            else:
                index_set = index_set.split('.')
                if index_set[0] == index_set[1]:
                    continue
            # find element of current length from already obtained Lie elements to check for antisymmetry
            # (for outer left elements of the bracket, additionally the other ones have to match)

            with_current_length = filter(lambda x: len(x) == len(index_set), lie_elements.keys())
            for value in with_current_length:
                if index_set[:2] == value[:2][::-1] and index_set[2:] == value[2:]:
                    break
            else:
                if len(index_set) == 2:
                    as_bracket = '.'.join(index_set)
                else:
                    as_bracket = index_set[0] + '.' + ']'.join(index_set[1:])
                lie_repr = np.zeros((dim, 1), dtype=int)
                lie_unfolded = unfold_lie_bracket(as_bracket)
                lie_unfolded = lie_unfolded.split('|')
                for moment in lie_unfolded:
                    if moment.startswith('-'):
                        lie_repr -= order_moments[moment[1:]]
                    else:
                        lie_repr += order_moments[moment]
                lie_elements[as_bracket] = lie_repr

        res[order] = lie_elements
    with open('api/lie_elements.pickle', 'wb') as f:
        pickle.dump(res, f)
    return res


def get_basis_lie_elements(max_order):
    """
    Constructs a basis of graded Lie algebra up to max_order.
    Returns a dictionary where key represents the order of the grading and value is basis data of that grading
    represented as a dictionary where key is encoded Lie element and value (dictionary with key 'repr')
    is its representation in R^p (as numpy array).
    """

    res = {}
    with open('api/lie_elements.pickle', 'rb') as f:
        lie_elements = pickle.load(f)
    for order in range(1, max_order + 1):
        grouped = [list(g) for k, g in groupby(lie_elements[order].items(), key=lambda x: len(x[0]))]
        basis_elements = {}
        for group in grouped:
            lie, cols = zip(*group)
            mat = np.concatenate(cols).reshape((-1, len(cols)), order='F')
            _, inds = sym.Matrix(mat).rref()
            for ind in inds:
                basis_elements[lie[ind]] = {
                    'repr': cols[ind]
                }
        res[order] = basis_elements
    with open('api/lie_basis_new.pickle', 'wb') as lb:
        pickle.dump(res, lb)
    return res


class LieElementsNotFound(Exception):
    pass


class SystemIsTooDeep(Exception):
    pass
