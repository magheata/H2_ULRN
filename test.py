# -*- coding: utf-8 -*-
# @Time    : 18/2/21 12:00
# @Author  : Miruna Andreea Gheata
# @Email   : miruna.gheata1@estudiant.uib.cat
# @File    : test.py
# @Software: PyCharm

import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler

# Plots the data points of each class; if provided, it will also plot the decision boundary
def scatter_plot(X, y, title, xlabel, ylabel, pred=None):
    plt.figure(figsize=(15, 10))
    plt.scatter(X[np.where(y == 0)[0]][:, 0], X[np.where(y == 0)[0]][:, 1], marker='+', color='#A2D9CE',
                label='class 0')
    plt.scatter(X[np.where(y == 1)[0]][:, 0], X[np.where(y == 1)[0]][:, 1], marker='*', color='#3CB371',
                label='class 1')
    plt.scatter(X[np.where(y == 2)[0]][:, 0], X[np.where(y == 2)[0]][:, 1], marker='.', color='#90CAF9',
                label='class 2')
    if pred:
        plt.scatter(X[pred][:, 0], X[pred][:, 1], marker='+', color='#FF0000', label='misclassified')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis('equal')
    plt.legend()
    plt.title(title)
    plt.show(block=False)

def get_misclassified(pred):
    missclassified_idxs = []
    for i in range(len(pred) - 1):
        if 3 <= i <= (len(pred) - 3):
            if ((pred[i - 1] != pred[i]) or (pred[i + 1] != pred[i])) and (
                    (pred[i - 2] != pred[i]) or (pred[i + 2] != pred[i])) and (
                    (pred[i - 3] != pred[i]) or (pred[i + 3] != pred[i])):
                missclassified_idxs.append(i)
    return missclassified_idxs


def check_misclassified(pred, missclassified):
    removed = False
    for i in missclassified:
        if ((pred[i - 1] == pred[i]) or (pred[i + 1] == pred[i])) and (
                (pred[i - 2] == pred[i]) or (pred[i + 2] == pred[i])):
            missclassified.remove(i)
            removed = True
    return removed


def proximity_function(x1, x2, theta_Mj):
    dist = ((math.sin(theta_Mj) ** 2) * (x1 ** 2))
    dist = dist + ((math.cos(theta_Mj) ** 2) * (x2 ** 2))
    dist = dist - (2 * math.sin(theta_Mj) * math.cos(theta_Mj) * x1 * x2)
    return dist


def compute_thetaj_crisp(u, j, X):
    num = 0
    den = 0
    for i in range(0, X.shape[0]):
        num = num + (u[i][j] * X[i, 0] * X[i, 1])
    num *= 2

    for i in range(0, X.shape[0]):
        den = den + (u[i][j] * (X[i, 0] ** 2 - X[i, 1] ** 2))

    return 0.5 * math.atan2(num, den)


def compute_thetaj_fuzzy(q, u, j, X):
    num = 0
    den = 0
    for i in range(0, X.shape[0]):
        num = num + (np.power(u[i][j], q) * X[i, 0] * X[i, 1])
    num *= 2

    for i in range(0, X.shape[0]):
        den = den + (np.power(u[i][j], q) * ((X[i, 0] ** 2) - (X[i, 1] ** 2)))

    return 0.5 * math.atan2(num, den)


def compute_cost_crisp(X, u, theta_M):
    cost = 0
    for i in range(X.shape[0]):
        for j in range(theta_M.shape[0]):
            prox = u[i][j] * proximity_function(X[i, 0], X[i, 1], theta_M[j])
            cost = cost + prox
    return cost

def compute_cost_fuzzy(q, X, u, theta_M):
    cost = 0
    for i in range(X.shape[0]):
        for j in range(theta_M.shape[0]):
            prox = np.power(u[i][j], q) * proximity_function(X[i, 0], X[i, 1], theta_M[j])
            cost = cost + prox
    return cost

def assign_cluster_crisp(i, j, u, X, theta_M):
    list_proxs = []
    for idx_cluster in range(len(theta_M)):
        list_proxs.append(proximity_function(X[i, 0], X[i, 1], theta_M[idx_cluster]))
    if list_proxs.index(min(list_proxs)) == j:
        u[i][j] = 1
    else:
        u[i][j] = 0


def assign_cluster_fuzzy(q, i, j, u, X, theta_M):
    k_dist = 0
    prox = proximity_function(X[i, 0], X[i, 1], theta_M[j])

    compute_normal = True

    prox_jk = []
    if prox == 0:
        print("Hola")
        u_value = 1 / len(theta_M)
        for j_aux in range(X.shape[1]):
            u[i][j_aux] = u_value
    else:
        for k in range(len(theta_M)):
            prox_k = proximity_function(X[i, 0], X[i, 1], theta_M[k])
            if prox_k == 0:
                print("Hola")
                u_value = 1 / len(theta_M)
                for j_aux in range(X.shape[1]):
                    u[i][j_aux] = u_value
                compute_normal = False
                break
            else:
                prox_jk.append(prox_k)
                #k_dist = k_dist + proximity_function(X[i, 0], X[i, 1], theta_M[k])

        if compute_normal:
            exp = (1 / (q - 1))
            res = 0
            for c in range(len(prox_jk)):
                res = res + (prox / prox_jk[c])
            #res = prox / k_dist
            den = np.power(res, exp)
            u[i][j] = (1 / den)
            #print(f"prox: {prox}  kdist: {k_dist} res: {res} den: {den} u[i][j]: {u[i][j]}")
    return u

def do_crisp_clust(X, M, n_iter, n_attempts, eps):
    attempts = 0
    N = X.shape[0]
    theta_M = np.zeros((M))
    best_theta_M = None
    best_cost = sys.maxsize
    best_cost_evolution = None
    best_attempt = None
    while attempts < n_attempts:
        cost_evolution = []
        print(f"\nAttempt {attempts}")
        cost = 1
        for j in range(M):
            theta_M[j] = math.radians(random.randint(0, 360))
        u = np.zeros((N, M))
        for t in range(n_iter):
            print(f"        Iteration {t}")
            old_cost = cost
            cost_evolution.append(cost)
            for i in range(N):
                for j in range(M):
                    assign_cluster_crisp(i, j, u, X, theta_M)
            for j in range(M):
                theta_M[j] = compute_thetaj_crisp(u, j, X)
            cost = compute_cost_crisp(X, u, theta_M)
            print("             Cost:", cost)
            if np.abs(old_cost - cost) < eps:
                break
        if cost < best_cost:
            best_cost = cost
            best_theta_M = deepcopy(theta_M)
            best_u = deepcopy(u)
            best_cost_evolution = deepcopy(cost_evolution)
            best_attempt = attempts
        attempts += 1
    return best_u, best_theta_M, best_cost_evolution, best_attempt


def do_fuzzy_clustering(X, M, n_iter, n_attempts, eps, q):
    attempts = 0
    N = X.shape[0]
    theta_M = np.zeros((M))
    best_theta_M = None
    best_cost = sys.maxsize
    best_cost_evolution = None
    best_attempt = None
    while attempts < n_attempts:
        cost_evolution = []
        print(f"\nAttempt {attempts}")
        cost = sys.maxsize
        for j in range(M):
            theta_M[j] = math.radians(random.randint(0, 360))
        u = np.zeros((N, M))
        for t in range(n_iter):
            print(f"        Iteration {t}")
            old_cost = cost
            cost_evolution.append(cost)
            for i in range(N):
                for j in range(M):
                    u = assign_cluster_fuzzy(q, i, j, u, X, theta_M)
            for j in range(M):
                theta_M[j] = compute_thetaj_fuzzy(q, u, j, X)
            cost = compute_cost_fuzzy(q, X, u, theta_M)
            print("             Cost:", cost)
            if np.abs(old_cost - cost) < eps:
                break
        if cost < best_cost:
            best_cost = cost
            best_theta_M = deepcopy(theta_M)
            best_u = deepcopy(u)
            best_cost_evolution = deepcopy(cost_evolution)
            best_attempt = attempts
        attempts += 1
    return best_u, best_theta_M, best_cost_evolution, best_attempt

def get_fuzzy_cluster(u):
    y = np.array(len(u))
    for i in range(len(u)):
        print(u[i])
        y[i] = np.argmax(u[i])
        print(y[i])


if __name__ == "__main__":
    group = '10'
    ds = 3
    data = np.loadtxt('ds103.txt')
    X = data[:, 0:2]
    y = data[:, 2:3]
    scatter_plot(X, y, "Original data points with their respective classes", "x", "y")

    M = 3
    n_attempts = 10
    n_iter = 100
    eps = 0.001
    u, theta_M, cost_evolution, best_attempt = do_crisp_clust(X, M, n_iter, n_attempts, eps)

    clusters_pred = np.zeros((len(u)))
    for i in range(len(u)):
        if u[i][0] == 1.0:
            clusters_pred[i] = 0
        elif u[i][1] == 1.0:
            clusters_pred[i] = 1
        else:
            clusters_pred[i] = 2

    clusters_pred = np.reshape(clusters_pred, (-1, 1))

    missclassified = get_misclassified(np.reshape(clusters_pred, -1))

    removed_missclassified = check_misclassified(np.reshape(clusters_pred, -1), missclassified)

    while removed_missclassified:
        removed_missclassified = check_misclassified(np.reshape(clusters_pred, -1), missclassified)

    scatter_plot(X, y, "Classified data points (Crisp Clustering)", "x", "y", missclassified)

    cost_evolution.pop(0)
    plt.plot(cost_evolution)
    plt.title(f"Cost evolution (Crisp Clustering)")
    plt.ylabel("Cost (scaled)")
    plt.ylim(bottom=0)
    plt.show()

    q = 2

    u, theta_M, cost_evolution, best_attempt = do_fuzzy_clustering(X, M, n_iter, n_attempts, eps, q)

    mms = MinMaxScaler()

    cost_evolution.pop(0)
    transformed_cost = mms.fit_transform(pd.DataFrame(cost_evolution))
    plt.plot(cost_evolution)
    plt.title(f"Cost evolution (Fuzzy Clustering)")
    plt.ylabel("Cost (scaled)")
    plt.ylim(bottom=0)
    plt.show()

    #get_fuzzy_cluster(u)


