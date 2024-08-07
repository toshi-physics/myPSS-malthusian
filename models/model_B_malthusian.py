import numpy as np
from tqdm import tqdm
import json, argparse, os

from src.field import Field
from src.system import System
from src.explicitTerms import Term
from src.fourierfunc import *
from src.functions import sqrtabs


def main():

    initParser = argparse.ArgumentParser(description='model_ZY')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir
    
    if os.path.isfile(savedir+"/parameters.json"):
	    with open(savedir+"/parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
              
    run       = parameters["run"]
    T         = parameters["T"]        # final time
    dt_dump   = parameters["dt_dump"]
    n_steps   = int(parameters["n_steps"])  # number of time steps
    k_g       = parameters["k_g"]
    rho_end   = parameters["rhoend"]
    rho_seed  = parameters["rhoseed"]   # seeding density, normalised by 100 mm^-2
    rho_min   = parameters["rhomin"]
    rho_max   = parameters["rhomax"]
    a         = parameters["a"]
    d         = parameters["d"]
    
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])

    dt        = T / n_steps     # time step size
    n_dump    = round(T / dt_dump)
    dn_dump   = round(n_steps / n_dump)
    
     # Define the grid size.
    grid_size = np.array([mx, my])
    dr=np.array([dx, dy])

    k_list, k_grids = momentum_grids(grid_size, dr)
    fourier_operators = k_power_array(k_grids)
    # Initialize the system.
    system = System(grid_size, fourier_operators)

    # Create fields that undergo timestepping
    system.create_field('rho', k_list, k_grids, dynamic=True)

    system.create_field('Ident', k_list, k_grids, dynamic=False)
    system.create_field('mu', k_list, k_grids, dynamic=False)
    
    # Create equations, if no function of rho, write None. If function and argument, supply as a tuple. 
    # Write your own functions in the function library or use numpy functions
    # if using functions that need no args, like np.tanh, write [("fieldname", (np.tanh, None))]
    # for functions with an arg, like, np.power(fieldname, n), write [("fieldname", (np.power, n))]
    # Define Identity # The way its written if you don't define a RHS, the LHS becomes zero at next timestep for Static Fields
    system.create_term("Ident", [("Ident", None)], [1, 0])
    
    system.create_term("mu", [("rho", (np.power, 3))], [4, 0, 0, 0])
    system.create_term("mu", [("rho", (np.power, 2))], [-6*(rho_min+rho_max), 0, 0, 0])
    system.create_term("mu", [("rho", None)], [2*(rho_max**2 + rho_min**2 + 4*rho_min*rho_max), 0, 0, 0])
    system.create_term("mu", [("Ident", None)], [-2*(rho_min + rho_max)*rho_min*rho_max, 0, 0, 0])
    system.create_term("mu", [("rho", None)], [d, 1, 0, 0])
    
    # Create terms for rho timestepping
    system.create_term("rho", [("mu", None)], [-a, 1])
    system.create_term("rho", [("rho", None)], [k_g, 0])
    system.create_term("rho", [("rho", (np.power, 2))], [-k_g/rho_end, 0])

    rho     = system.get_field('rho')

    # set init condition and synchronize momentum with the init condition, important!!
    #radius = 16
    #set_rho_islands(rho, 100, rho_seed, grid_size)
    rhoseed = np.random.rand(mx, my) + np.ones([mx, my])
    rhoseed = rhoseed*rho_seed/np.average(rhoseed)
    rho.set_real(rhoseed)
    rho.synchronize_momentum()

    # Initialise identity matrix 
    system.get_field('Ident').set_real(np.ones(shape=grid_size))
    system.get_field('Ident').synchronize_momentum()

    if not os.path.exists(savedir+'/data/'):
        os.makedirs(savedir+'/data/')

    for t in tqdm(range(n_steps)):
    
        system.update_system(dt)

        if t % dn_dump == 0:
            np.savetxt(savedir+'/data/'+'rho.csv.'+ str(t//dn_dump), rho.get_real(), delimiter=',')
            #np.savetxt(savedir+'/data/'+'mu.csv.'+ str(t//dn_dump), system.get_field('mu').get_real(), delimiter=',')

def momentum_grids(grid_size, dr):
    k_list = [np.fft.fftfreq(grid_size[i], d=dr[i])*2*np.pi for i in range(len(grid_size))]
    # k is now a list of arrays, each corresponding to k values along one dimension.

    k_grids = np.meshgrid(*k_list, indexing='ij')
    #k_grids = np.meshgrid(*k_list, indexing='ij', sparse=True)
    # k_grids is now a list of 2D sparse arrays, each corresponding to k values in one dimension.

    return k_list, k_grids

def k_power_array(k_grids):
    k_squared = sum(ki**2 for ki in k_grids)

    k_power_arrays = [k_squared]

    return k_power_arrays

def set_rho_islands(rhofield, ncluster, rhoseed, grid_size):
    centers = grid_size[0]*np.random.rand(ncluster,2); radii = 0.1*np.random.rand(ncluster)*grid_size[0]
    mean = rhoseed; std = rhoseed/2

    tol = 0.001
    x   = np.arange(0+tol, grid_size[0]-tol, 1)
    y   = np.arange(0+tol, grid_size[1]-tol, 1)
    r   = np.meshgrid(x,y)

    rhoinit = np.zeros(grid_size)
    for i in np.arange(ncluster):
        distance = np.sqrt((r[0]-centers[i,0])**2+(r[1]-centers[i,1])**2)
        rhoseeds = np.abs(np.random.normal(mean, std, size=np.shape(distance)))
        rhoinit += np.where(distance < radii[i], rhoseeds*(radii[i]-distance)/radii[i], 1e-3)

    meanrho = np.average(rhoinit)

    rhoinit = rhoinit * rhoseed / meanrho

    print("Average rho at start is", np.average(rhoinit), " for rhoseed=", rhoseed)
    
    rhofield.set_real(rhoinit)
    rhofield.synchronize_momentum()

def set_rho_island(rhofield, rhoseed, radius, grid_size):
    center = grid_size/2
    
    tol = 0.001
    x   = np.arange(0+tol, grid_size[0]-tol, 1)
    y   = np.arange(0+tol, grid_size[1]-tol, 1)
    r   = np.meshgrid(x,y)

    rhoinit = np.zeros(grid_size)
    distance = np.sqrt((r[0]-center[0])**2+(r[1]-center[1])**2)
    rhoinit = np.where(distance < radius, rhoseed*(radius-distance)/radius, 1e-4)
    rhoinit = np.where(rhoinit<1e-4, 1e-4, rhoinit)

    rhofield.set_real(rhoinit)
    rhofield.synchronize_momentum()

if __name__=="__main__":
    main()
