#!/usr/bin/env python
#!python
'''
Script to run on cluster for percolation
Usage: python runPercolate.py 1000 1 1 1.5
'''

import sys
import numpy as np
import networkx as nx
import h5py

N     = int(float(sys.argv[1]))
seed  = int(float(sys.argv[2])) + N
usePR = int(float(sys.argv[3]))
r     = float(sys.argv[4])


np.random.seed(seed)

#Add random edge in Erdos renyi model
def add_ER_edge(G, N):
    added = False
    while not added:
        edge = np.random.choice(range(N), size = 2, replace = False)
        if G.has_edge(edge[0], edge[1]):
            continue
        G.add_edge(edge[0], edge[1])
        added = True
    return [edge[0], edge[1]]


def add_PR_edge(G, N, compSZ, compID):
    added = False
    while not added:
        node_list = np.random.choice(range(N), size = 4, replace = False)
        if G.has_edge(node_list[0], node_list[1]) or G.has_edge(node_list[2], node_list[3]):
            continue
        C = [compSZ[compID[node]] for node in node_list]
        if C[0]*C[1] < C[2]*C[3]:
            G.add_edge(node_list[0], node_list[1])
            return [node_list[0], node_list[1]]
        else:
            G.add_edge(node_list[2], node_list[3])
            return [node_list[2], node_list[3]]
            

#For a given graph size threshold and gamma runs simulation for fixed number of steps
def percolate(usePR, N, thresh, gamma, r, snapshots):
    G = nx.empty_graph(N)
    steps = int(N*r)
    compLS = {i:{i} for i in range(N)}
    compID = np.arange(N)
    compSZ = np.ones(N)
    discad = np.zeros(N) #stores list of discarded components
    curMax = 1
    
    if len(snapshots) > 0:
        snapshots = [int(s) for s in snapshots]
        snapshots = list(set(snapshots)) #remove duplicates
        snapshots.sort()
        saveDisc = np.zeros([len(snapshots), N])
        saveSize = np.zeros([len(snapshots), N])    
        snapID = 0
        SnapSize = len(snapshots)
    else:
        saveDisc = []
        saveSize = []
        snapID   = -1
        SnapSize = -1
    
    compSize = np.zeros(steps)

    for i in range(steps):
        if usePR == 1:
            nds = add_PR_edge(G, N, compSZ, compID)
        else:
            nds = add_ER_edge(G, N)
            
            
        if compID[nds[0]] != compID[nds[1]]:
            if compSZ[compID[nds[0]]] >= compSZ[compID[nds[1]]]:
                big = nds[0]
                sma = nds[1]
            else:
                big = nds[1]
                sma = nds[0]
            discad[compID[sma]] = True
            compSZ[compID[big]] += compSZ[compID[sma]]
            if compSZ[compID[big]] > curMax:
                curMax = compSZ[compID[big]]
            oldID = compID[sma]
            for j in compLS[compID[sma]]:
                compID[j] = compID[big]
            compLS[compID[big]] = compLS[compID[big]] | compLS[oldID] 
        compSize[i] = curMax
        
        if compSize[i] >= N**gamma and compSize[i - 1] < N**gamma:
            t0 = i - 1
        if compSize[i] >= N*thresh and compSize[i - 1] < N*thresh:
            t1 = i
        
        if snapID < SnapSize and i == snapshots[snapID]:
            saveDisc[snapID, :] = np.array(discad)
            saveSize[snapID, :] = np.array(compSZ)
            print('step: ' + str(i))
            snapID += 1
    delta = t1 - t0
    return (compSize, delta, t0, t1, saveDisc, saveSize)    
    
  
  

def saveSim(savePath, compsize, r, delta, t0, t1, N, seed, saveDisc, saveSize, snapshots):
    with h5py.File(savePath + '.hdf5', "a") as f:
        f.create_dataset("compsize", data=np.array(compsize))
        f.create_dataset("r", data=r)
        f.create_dataset("delta", data=delta)
        f.create_dataset("t0", data=t0)
        f.create_dataset("t1", data=t1)
        f.create_dataset("N", data=N)
        f.create_dataset("seed", data=seed)                                        
        f.create_dataset("saveDisc", data=np.array(saveDisc))                
        f.create_dataset("saveSize", data=np.array(saveSize))
        f.create_dataset("snapshots", data=np.array(snapshots))       
        
snapshots = list(np.linspace(0,N,(round(r)*100 + 1))) + list(np.arange(0.88*N, 0.9*N, 0.001*N))
compsize, delta, t0, t1, saveDisc, saveSize = percolate(usePR, N, 0.5, 0.5, r, snapshots)

if usePR == 1:
    preString = 'PR'
else:
    preString = 'ER'
    
savePath = preString + '_N' + str(N) + '_seed' + str(seed) + '_r' + str(r)
saveSim(savePath, compsize, r, delta, t0, t1, N, seed, saveDisc, saveSize, snapshots)    
print('DONE')
