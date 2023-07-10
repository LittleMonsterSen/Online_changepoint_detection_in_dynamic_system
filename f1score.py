#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 15:49:26 2023

@author: senlin
"""

#### metrics to evaluate change point detection performance
def true_positives(T, X, margin=5):
    """Compute true positives without double counting
    >>> true_positives({1, 10, 20, 23}, {3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 8, 20})
    {1, 10, 20}
    >>> true_positives({1, 10, 20, 23}, {1, 3, 5, 8, 20})
    {1, 10, 20}
    >>> true_positives(set(), {1, 2, 3})
    set()
    >>> true_positives({1, 2, 3}, set())
    set()
    """
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def f1_score(annotations, predictions, margin=5, return_PR=False):
    """Compute the F-measure based on human annotations.
    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too
    Remember that all CP locations are 0-based!
    """
    # ensure 0 is in all the sets
    TP = true_positives(annotations, predictions, margin=margin)
    P = len(TP)/len(predictions)
    R = len(TP)/len(annotations)
    F = (2*P*R)/(P+R)
    if return_PR:
        return F, P, R
    return F