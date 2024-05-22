import os
os.chdir('/users/fodde/gubas/Didymos')

import numpy as np
import csv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d


def load_shape_from_obj(file_path):
    try:
        vertices = []
        faces = []
        with open(file_path) as f:
            for line in f:
                if line[0] == "v":
                    vertex = list(map(float, line[2:].strip().split()))
                    vertices.append(vertex)
                elif line[0] == "f":
                    face = list(map(int, line[2:].strip().split()))
                    faces.append(face)

        return (np.array(vertices), np.array(faces))

    except FileNotFoundError:
        print(f"{file_path} not found.")
    except:
        print("An error occurred while loading the shape.")


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

sd = load_shape_from_obj('/users/fodde/gubas/Didymos/ShapeFiles/didymos_g_9309mm_spc_obj_0000n00000_v003.obj')
DCM = np.array([[0.016797049063314525, -0.08127894686665554, -0.9965498441819215], 
                [0.7441830641928502, 0.6666646940596693, -0.04183004497806754],
                [-0.6677644989899543, 0.7409128953463926, -0.0716844153084954]]).T
vertices = np.c_[np.zeros(len(sd[0])), DCM.dot(sd[0].T).T*1000]
filename = 'ShapeFiles/didymos_new_vert_large.csv'
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(vertices)

filename = 'ShapeFiles/didymos_new_facet_large.csv'
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(sd[1])

pc = art3d.Poly3DCollection(vertices[sd[1] - 1, 1:], edgecolor="black", alpha=0.5)
ax.add_collection(pc)

sd = load_shape_from_obj('/users/fodde/gubas/Didymos/ShapeFiles/dimorphos_g_1940mm_spc_obj_0000n00000_v004.obj')
DCM = np.array([[0.003491754835678261, 0.011244696597572034, -0.9999306798206545], 
                [-0.13182419747586419, -0.9912051557313173, -0.011606903613783763],
                [0.9912669613217278, -0.13185558786067372, 0.0019787222908395296]]).T
vertices = np.c_[np.zeros(len(sd[0])), DCM.dot(sd[0].T).T*1000]
filename = 'ShapeFiles/dimorphos_new_vert_large.csv'
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(vertices)

filename = 'ShapeFiles/dimorphos_new_facet_large.csv'
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(sd[1])

vertex_didy = np.genfromtxt('/users/fodde/gubas/Didymos/ShapeFiles/Didymos_A_vert_met.csv', usecols=[1,2,3], delimiter=',') 
facet_didy = np.genfromtxt('/users/fodde/gubas/Didymos/ShapeFiles/Didymos_A_facet.csv', usecols=[0,1,2], delimiter=',', dtype=np.int16) 
pc = art3d.Poly3DCollection(vertex_didy[facet_didy - 1] + [900,0,0], edgecolor="black", alpha=0.5)
ax.add_collection(pc)
ax.set_xlim([-400,1200])
ax.set_ylim([-800,800])
ax.set_zlim([-800,800])
plt.show()