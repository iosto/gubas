from matplotlib.lines import lineStyles
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d
from setuptools import find_packages
from matplotlib.colors import LightSource
import scipy
import matplotlib as mpl
import math
import pandas as pd
import pyvista as pv
import gubas_data_reader as gdr

fontsize=26

dim_mesh = pv.read('/users/fodde/gubas/Didymos/ShapeFiles/dimorphos_g_1940mm_spc_obj_0000n00000_v004.obj')
dim_mesh.points[:] = (dim_mesh.points - dim_mesh.center)*1000 + dim_mesh.center
DCM = np.array([[-1,0,0], [0,-1,0],[0,0,1]])
dim_mesh.points[:] = DCM.dot(dim_mesh.points[:].T).T
didy_mesh = pv.read('/users/fodde/gubas/Didymos/ShapeFiles/didymos_g_9309mm_spc_obj_0000n00000_v003.obj')
didy_mesh.points[:] = (didy_mesh.points - didy_mesh.center)*1000 + didy_mesh.center
connecting_line_mesh = pv.Line(didy_mesh.center, dim_mesh.center)
dim_basis_arrows = [pv.Arrow((0,0,0), (1,0,0), scale=300), pv.Arrow((0,0,0), (0,1,0), scale=300), pv.Arrow((0,0,0), (0,0,1), scale=300)]

pl = pv.Plotter(off_screen=False)
pl.set_background('black')
pl.add_mesh(didy_mesh, smooth_shading=True)
pl.add_mesh(dim_mesh, smooth_shading=True)
pl.add_mesh(dim_basis_arrows[0], smooth_shading=True, color = 'red')
pl.add_mesh(dim_basis_arrows[1], smooth_shading=True, color = 'green')
pl.add_mesh(dim_basis_arrows[2], smooth_shading=True, color = 'blue')
pl.add_mesh(connecting_line_mesh, color='gray', line_width=1)

pl.open_gif("super_synchronous.gif")

def zoom_to_data(self, data):
    if not self.camera_set:
        self.view_isometric()
    self.reset_camera(bounds=(-1200, 1200, -1200, 1200, -200, 200))
    self.camera_set = True
    self.reset_camera_clipping_range()

pts_dim = dim_mesh.points
pts_did = didy_mesh.points
pts_didy_arrow = [dim_basis_arrows[0].points, dim_basis_arrows[1].points, dim_basis_arrows[2].points]
states = gdr.read_output_files('86400.0_60.0_60.0_4_2_SuperSynchronous')
eul123_rad, eul123_deg = gdr.euler_angles(states)

f, ax = plt.subplots(1,3,tight_layout=True)
h_line_roll = ax[0].plot(states['time'][:1]/3600, eul123_deg[:1, 0])[0]
h_line_pitch = ax[1].plot(states['time'][:1]/3600, eul123_deg[:1, 1])[0]
h_line_yaw = ax[2].plot(states['time'][:1]/3600, eul123_deg[:1, 2])[0]
ax[0].set_ylim([-90, 90])
ax[0].set_xlim([0, states['time'][-1]/3600])
ax[0].set_xlabel('Time (hrs)')
_ = ax[0].set_ylabel('Roll (deg)')
ax[1].set_ylim([-90, 90])
ax[1].set_xlim([0, states['time'][-1]/3600])
ax[1].set_xlabel('Time (hrs)')
_ = ax[1].set_ylabel('Pitch (deg)')
ax[2].set_ylim([-90, 90])
ax[2].set_xlim([0, states['time'][-1]/3600])
ax[2].set_xlabel('Time (hrs)')
_ = ax[2].set_ylabel('Yaw (deg)')

h_chart = pv.ChartMPL(f, size=(0.66, 0.25), loc=(0.25, 0.75))
h_chart.background_color = (1.0, 1.0, 1.0, 0.4)
pl.add_chart(h_chart)

corotating = True
for idx in range(0,len(states['time'])):
    if idx % 5 == 0:
        # Position data
        pos_dimorphos = states['relative']['pos_Pframe'][idx, :]

        # Frame data
        rot_dimorphos_primary = states['rotmatrix']['SP'][idx, :].reshape((3,3))
        rot_primary_dimorphos = rot_dimorphos_primary.T
        rot_primary_inertial = states['rotmatrix']['PI'][idx, :].reshape((3,3))
        rot_inertial_primary = rot_primary_inertial.T
        ang_vel_rel = np.linalg.norm(np.cross(rot_primary_inertial.dot(states['relative']['pos_Pframe'][idx, :]), rot_primary_inertial.dot(states['relative']['vel_Pframe'][idx, :]))/np.linalg.norm(pos_dimorphos)**2)
        rot_inertial_corot = np.array([[np.cos(-ang_vel_rel*states['time'][idx]), -np.sin(-ang_vel_rel*states['time'][idx]), 0],
                                       [np.sin(-ang_vel_rel*states['time'][idx]), np.cos(-ang_vel_rel*states['time'][idx]), 0],
                                       [0, 0, 1]])
        
        # Update body location and rotation
        if corotating:
            dim_mesh.points = rot_inertial_corot.dot(rot_primary_inertial.dot(rot_dimorphos_primary.dot((pts_dim).T))).T + states['relative']['pos_Pframe'][0, :]
            didy_mesh.points = rot_inertial_corot.dot(rot_primary_inertial.dot(pts_did.T)).T
        else:
            dim_mesh.points = rot_primary_inertial.dot(rot_dimorphos_primary.dot((pts_dim).T)).T  + rot_primary_inertial.dot(pos_dimorphos.T).T
            didy_mesh.points = rot_primary_inertial.dot(pts_did.T).T
        # generate frame arrows
        for i in range(3):
            if corotating:
                dim_basis_arrows[i].points = rot_inertial_corot.dot(rot_primary_inertial.dot(rot_dimorphos_primary.dot((pts_didy_arrow[i]).T))).T + states['relative']['pos_Pframe'][0, :]
            else :
                dim_basis_arrows[i].points = rot_primary_inertial.dot(rot_dimorphos_primary.dot((pts_didy_arrow[i]).T)).T  + rot_primary_inertial.dot(pos_dimorphos.T).T

        connecting_line_mesh.points = pv.Line(didy_mesh.center, dim_mesh.center).points

        # zoom to desired bounds
        zoom_to_data(pl, dim_mesh)

        # Update Euler angles
        h_line_roll.set_xdata(states['time'][: idx + 1]/3600)
        h_line_roll.set_ydata(eul123_deg[: idx + 1, 0])
        h_line_pitch.set_xdata(states['time'][: idx + 1]/3600)
        h_line_pitch.set_ydata(eul123_deg[: idx + 1, 1])
        h_line_yaw.set_xdata(states['time'][: idx + 1]/3600)
        h_line_yaw.set_ydata(eul123_deg[: idx + 1, 2])

        # Write a frame. This triggers a render.
        pl.write_frame()
pl.show()
