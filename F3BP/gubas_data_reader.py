import numpy as np
import pandas as pd

from scipy.spatial.transform import Rotation as R

def euler_angles(states):
    ####################################
    # Computes the euler angles between the secondary body-fixed
    # reference frame and the rotating orbital reference frame at each instant 
    # of time. 
    #
    # Note on reference frames:
    # Secondary body-fixed reference frame consideres the x-axis along the
    # semi-major axis, the y-axis along the semi-intermediate axis and the
    # z-axis along the semi-minor axis. 
    # Rotating orbital reference frame considers the x-axis in the direction
    # of the LOC and the z-axis on the direction of the total angular
    # momentum, while the y-axis completes the right-hand triad. 
    # credits: Irene Luján Fernández and Iosto Fodde
    #####################################

    # Initialization
    N = len(states['time'])
    eul123_rad = np.zeros((N, 3))
    eul123_deg = np.zeros((N, 3))

    for i in range(N):
        # Orbit rotational (OR) frame
        # x: Line of Centers (LOC) direction
        xOR_P = np.array(states['relative']['pos_Pframe'][i]).reshape(3, 1)
        rotSP = np.array(states['rotmatrix']['SP'][i]).reshape(3, 3)  
        xOR_S = rotSP.T @ xOR_P  # x vector of OR expressed in Secondary RF
        xOR_S = xOR_S / np.linalg.norm(xOR_S)  # x direction of OR expressed in Secondary RF

        # z: total angular momentum direction
        zOR_I = np.array(states['totalangmom'][i]).reshape(3, 1)
        rotPI = np.array(states['rotmatrix']['PI'][i]).reshape(3, 3) 
        zOR_P = rotPI.T @ zOR_I  # z vector of OR expressed in Primary RF
        zOR_S = rotSP.T @ zOR_P  # z vector of OR expressed in Secondary RF
        zOR_S = zOR_S / np.linalg.norm(zOR_S)  # z direction of OR expressed in Secondary RF

        # y: cross product
        yOR_S = np.cross(zOR_S.T, xOR_S.T).T

        # Rotational matrix secondary to orbital rotation
        rotSOR = np.vstack((xOR_S.T, yOR_S.T, zOR_S.T))

        # Euler angles 1-2-3
        rotation = R.from_matrix(rotSOR)
        eul123_rad[i, :] = rotation.as_euler('XYZ')
        eul123_deg[i, :] = np.rad2deg(eul123_rad[i, :])

    return eul123_rad, eul123_deg


def read_output_files(name):
    # Define file names
    files = {
        'Lagrangian': f'LagrangianStateOut_{name}.csv',
        'Hamiltonian': f'HamiltonianStateOut_{name}.csv',
        'EnergyAngMom': f'Energy+AngMom_{name}.csv',
        'Conservation': f'Conservation_Energy+AngMom_{name}.csv',
        'FHamiltonian': f'FHamiltonianStateOut_{name}.csv'
    }

    # Read Lagrangian states
    LagrangianStateOut = pd.read_csv(files['Lagrangian'], header=None).to_numpy()

    # Read Hamiltonian states
    HamiltonianStateOut = pd.read_csv(files['Hamiltonian'], header=None).to_numpy()

    # Read Energy + Angular Momentum
    EnergyAngMom = pd.read_csv(files['EnergyAngMom'], header=None).to_numpy()

    # Read Energy and Angular Momentum Conservation
    ConservationEnergyAngMom = pd.read_csv(files['Conservation'], header=None).to_numpy()

    # Read FHamiltonian
    FHamiltonianStateOut = pd.read_csv(files['FHamiltonian'], header=None).to_numpy()

    # State reorganization
    state = {
        'time': LagrangianStateOut[:, 0],
        'relative': {
            'pos_Pframe': LagrangianStateOut[:, 1:4],
            'vel_Pframe': LagrangianStateOut[:, 4:7],
            'linmom': HamiltonianStateOut[:, 4:7]
        },
        'primary': {
            'angvel_Pframe': LagrangianStateOut[:, 7:10],
            'angmom': HamiltonianStateOut[:, 7:10]
        },
        'secondary': {
            'angvel_Sframe': LagrangianStateOut[:, 10:13],
            'angmom': HamiltonianStateOut[:, 10:13]
        },
        'rotmatrix': {
            'SP': LagrangianStateOut[:, 13:22],
            'PI': LagrangianStateOut[:, 22:31]
        },
        'energy': {
            'mutualpotential': LagrangianStateOut[:, 31],
            'total': EnergyAngMom[:, 0]
        },
        'totalangmom': EnergyAngMom[:, 1:4],
        'fracerror': {
            'energy': ConservationEnergyAngMom[:, 0],
            'angmom': ConservationEnergyAngMom[:, 1]
        },
        '3rdBody': {
            'pos_Pframe': LagrangianStateOut[:, 32:35],
            'vel_Pframe': LagrangianStateOut[:, 35:38],
        }
    }

    return state
