import os
os.chdir('/users/fodde/gubas/F3BP')
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
num_streamlines = 10

dim_mesh = pv.read('/users/fodde/gubas/Didymos/ShapeFiles/dimorphos_g_1940mm_spc_obj_0000n00000_v004.obj')
dim_mesh.points[:] = (dim_mesh.points - dim_mesh.center)*1000 + dim_mesh.center
DCM = np.array([[-1,0,0], [0,-1,0],[0,0,1]])
dim_mesh.points[:] = DCM.dot(dim_mesh.points[:].T).T
didy_mesh = pv.read('/users/fodde/gubas/Didymos/ShapeFiles/didymos_g_9309mm_spc_obj_0000n00000_v003.obj')
didy_mesh.points[:] = (didy_mesh.points - didy_mesh.center)*1000 + didy_mesh.center
dim_basis_arrows = [pv.Arrow((0,0,0), (1,0,0), scale=300), pv.Arrow((0,0,0), (0,1,0), scale=300), pv.Arrow((0,0,0), (0,0,1), scale=300)]
third_body = pv.Sphere(radius=10.0)

dim_mesh_cr = dim_mesh.copy()
didy_mesh_cr = didy_mesh.copy()
dim_basis_arrows_cr = [arrow.copy() for arrow in dim_basis_arrows]
third_body_cr = third_body.copy()

pl = pv.Plotter(shape=(1,2), off_screen=False, window_size=(1920, 1080))
pl.subplot(0,0)
pl.set_background('black')
pl.add_mesh(didy_mesh, smooth_shading=True)
pl.add_mesh(dim_mesh, smooth_shading=True)
pl.add_mesh(dim_basis_arrows[0], smooth_shading=True, color = 'red')
pl.add_mesh(dim_basis_arrows[1], smooth_shading=True, color = 'green')
pl.add_mesh(dim_basis_arrows[2], smooth_shading=True, color = 'blue')
pl.add_mesh(third_body, color='gray')
streamlines = []
for i in range(num_streamlines):
    line = pv.Line((0,0,0), (0,0,0))
    pl.add_mesh(line, color="yellow", line_width=1, opacity=(num_streamlines - i)/num_streamlines)
    streamlines.append(line)

pl.subplot(0,1)
pl.set_background('black')
pl.add_mesh(didy_mesh_cr, smooth_shading=True)
pl.add_mesh(dim_mesh_cr, smooth_shading=True)
pl.add_mesh(dim_basis_arrows_cr[0], smooth_shading=True, color = 'red')
pl.add_mesh(dim_basis_arrows_cr[1], smooth_shading=True, color = 'green')
pl.add_mesh(dim_basis_arrows_cr[2], smooth_shading=True, color = 'blue')
pl.add_mesh(third_body_cr, color='gray')
streamlines_cr = []
for i in range(num_streamlines):
    line = pv.Line((0,0,0), (0,0,0))
    pl.add_mesh(line, color="yellow", line_width=2, opacity=(num_streamlines - i)/num_streamlines)
    streamlines_cr.append(line)

pl.open_gif("super_synchronous.gif")

def zoom_to_data(self, data):
    if not self.camera_set:
        self.view_isometric()
    self.reset_camera(bounds=(-1500, 1500, -1500, 1500, -200, 200))
    self.camera_set = True
    self.reset_camera_clipping_range()

pts_dim = dim_mesh.points
pts_did = didy_mesh.points
pts_didy_arrow = [dim_basis_arrows[0].points, dim_basis_arrows[1].points, dim_basis_arrows[2].points]
pts_third = third_body.points
pts_dim_cr = dim_mesh_cr.points
pts_did_cr = didy_mesh_cr.points
pts_didy_arrow_cr = [dim_basis_arrows_cr[0].points, dim_basis_arrows_cr[1].points, dim_basis_arrows_cr[2].points]
pts_third_cr = third_body_cr.points

states = gdr.read_output_files('86400.0_60.0_60.0_4_1_Tumbling')
eul123_rad, eul123_deg = gdr.euler_angles(states)

# f, ax = plt.subplots(1,3,tight_layout=True)
# h_line_roll = ax[0].plot(states['time'][:1]/3600, eul123_deg[:1, 0])[0]
# h_line_pitch = ax[1].plot(states['time'][:1]/3600, eul123_deg[:1, 1])[0]
# h_line_yaw = ax[2].plot(states['time'][:1]/3600, eul123_deg[:1, 2])[0]
# ax[0].set_ylim([-90, 90])
# ax[0].set_xlim([0, states['time'][-1]/3600])
# ax[0].set_xlabel('Time (hrs)')
# _ = ax[0].set_ylabel('Roll (deg)')
# ax[1].set_ylim([-90, 90])
# ax[1].set_xlim([0, states['time'][-1]/3600])
# ax[1].set_xlabel('Time (hrs)')
# _ = ax[1].set_ylabel('Pitch (deg)')
# ax[2].set_ylim([-90, 90])
# ax[2].set_xlim([0, states['time'][-1]/3600])
# ax[2].set_xlabel('Time (hrs)')
# _ = ax[2].set_ylabel('Yaw (deg)')
f, ax = plt.subplots(1,3,tight_layout=True)
h_line_xy = ax[0].plot([0], [0])[0]
h_line_xz = ax[1].plot([0], [0])[0]
h_line_yz = ax[2].plot([0], [0])[0]
ax[0].set_ylim([-1500, 1500])
ax[0].set_xlim([-1500, 1500])
ax[0].set_xlabel('X (m)')
ax[0].set_ylabel('Y (-)')
ax[1].set_ylim([-500, 500])
ax[1].set_xlim([-1500, 1500])
ax[1].set_xlabel('X (m)')
ax[1].set_ylabel('Z (-)')
ax[2].set_ylim([-500, 500])
ax[2].set_xlim([-1500, 1500])
ax[2].set_xlabel('Y (m)')
ax[2].set_ylabel('Z (-)')

h_chart = pv.ChartMPL(f, size=(0.66, 0.25), loc=(0.25, 0.75))
h_chart.background_color = (1.0, 1.0, 1.0, 0.4)
pl.add_chart(h_chart)

tbc = []
tbc_cr = []
int_idx = 0
for idx in range(0,len(states['time'])):
    if idx % 10 == 0:
        # Position data
        pos_dimorphos = states['relative']['pos_Pframe'][idx, :]
        pos_3rd_body = states['3rdBody']['pos_Pframe'][idx, :]

        # Frame data
        rot_dimorphos_primary = states['rotmatrix']['SP'][idx, :].reshape((3,3))
        rot_primary_dimorphos = rot_dimorphos_primary.T
        rot_primary_inertial = states['rotmatrix']['PI'][idx, :].reshape((3,3))
        rot_inertial_primary = rot_primary_inertial.T
        ang_vel_rel = np.linalg.norm(np.cross(rot_primary_inertial.dot(states['relative']['pos_Pframe'][idx, :]), rot_primary_inertial.dot(states['relative']['vel_Pframe'][idx, :]))/np.linalg.norm(pos_dimorphos)**2)
        rot_inertial_corot = np.array([[np.cos(-ang_vel_rel*states['time'][idx]), -np.sin(-ang_vel_rel*states['time'][idx]), 0],
                                       [np.sin(-ang_vel_rel*states['time'][idx]), np.cos(-ang_vel_rel*states['time'][idx]), 0],
                                       [0, 0, 1]])
        
        pl.subplot(0,0)
        # Update body location and rotation
        dim_mesh.points = rot_primary_inertial.dot(rot_dimorphos_primary.dot((pts_dim).T)).T  + rot_primary_inertial.dot(pos_dimorphos.T).T
        didy_mesh.points = rot_primary_inertial.dot(pts_did.T).T
        third_body.points = pts_third + pos_3rd_body

        # update streamlines
        tbc.append(third_body.center)
        for k, mesh in enumerate(streamlines):
            j = len(tbc) - 1
            if k < j:
                pos_3rd_body_prev2 = np.array(tbc[j-k-1])
                pos_3rd_body_prev = np.array(tbc[j-k])
                r1 = pos_3rd_body_prev2.T
                r2 = pos_3rd_body_prev
                mesh.points = pv.Line((r1[0], r1[1], r1[2]), (r2[0], r2[1], r2[2])).points

        # generate frame arrows
        for i in range(3):
                dim_basis_arrows[i].points = rot_primary_inertial.dot(rot_dimorphos_primary.dot((pts_didy_arrow[i]).T)).T  + rot_primary_inertial.dot(pos_dimorphos.T).T

        # zoom to desired bounds
        zoom_to_data(pl, dim_mesh)

        # Update Euler angles
        h_line_xy.set_xdata(np.array(tbc)[:int_idx+1, 0])
        h_line_xy.set_ydata(np.array(tbc)[:int_idx+1, 1])
        h_line_xz.set_xdata(np.array(tbc)[:int_idx+1, 0])
        h_line_xz.set_ydata(np.array(tbc)[:int_idx+1, 2])
        h_line_yz.set_xdata(np.array(tbc)[:int_idx+1, 1])
        h_line_yz.set_ydata(np.array(tbc)[:int_idx+1, 2])

        pl.subplot(0,1)
        dim_mesh_cr.points = rot_inertial_corot.dot(rot_primary_inertial.dot(rot_dimorphos_primary.dot((pts_dim_cr).T))).T + states['relative']['pos_Pframe'][0, :]
        didy_mesh_cr.points = rot_inertial_corot.dot(rot_primary_inertial.dot(pts_did_cr.T)).T
        third_body_cr.points = pts_third_cr + rot_inertial_corot.dot(pos_3rd_body.T).T

        tbc_cr.append(third_body_cr.center)
        for k, mesh in enumerate(streamlines_cr):
            j = len(tbc_cr) - 1
            if k < j:
                pos_3rd_body_prev2 = np.array(tbc_cr[j-k-1])
                pos_3rd_body_prev = np.array(tbc_cr[j-k])
                r1 = pos_3rd_body_prev2.T
                r2 = pos_3rd_body_prev
                mesh.points = pv.Line((r1[0], r1[1], r1[2]), (r2[0], r2[1], r2[2])).points
        
        # generate frame arrows
        for i in range(3):
            dim_basis_arrows_cr[i].points = rot_inertial_corot.dot(rot_primary_inertial.dot(rot_dimorphos_primary.dot((pts_didy_arrow_cr[i]).T))).T + states['relative']['pos_Pframe'][0, :]

        # zoom to desired bounds
        zoom_to_data(pl, dim_mesh)

        pl.write_frame()
        int_idx += 1

pl.show()
