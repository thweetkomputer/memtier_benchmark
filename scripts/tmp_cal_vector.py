"""
0.266665757, 0.539760351
0.375206977, 0.760248721
0.512535393, 0.667723775


0.531606436, 0.0392803438
"""

import numpy as np
def cal_cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def cal_ecuclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

def cal_l2sqk(v1, v2):
    return np.linalg.norm(v1 - v2) ** 2

v0 = np.array([0.266665757, 0.539760351])
v1 = np.array([0.375206977, 0.760248721])
v2 = np.array([0.512535393, 0.667723775])

tar = np.array([0.531606436, 0.0392803438])
# print(cal_cosine_similarity(v0, tar))
# print(cal_cosine_similarity(v1, tar))
# print(cal_cosine_similarity(v2, tar))
print(cal_ecuclidean_distance(v0, tar))
print(cal_ecuclidean_distance(v1, tar))
print(cal_ecuclidean_distance(v2, tar))
print(cal_l2sqk(v0, tar))
print(cal_l2sqk(v1, tar))
print(cal_l2sqk(v2, tar))


"""
[2025-04-17T10:34:44.476280 E 268760] [hnsw_index.cpp:117] [result] 0th key: 0 distance: 0.320674
[2025-04-17T10:34:44.476361 E 268760] [hnsw_index.cpp:117] [result] 1th key: 2 distance: 0.395305
[2025-04-17T10:34:44.476390 E 268760] [hnsw_index.cpp:121] [shadow result] 0th key: 2 distance: 0.395305
[2025-04-17T10:34:44.476418 E 268760] [hnsw_index.cpp:121] [shadow result] 1th key: 1 distance: 0.544256
"""

