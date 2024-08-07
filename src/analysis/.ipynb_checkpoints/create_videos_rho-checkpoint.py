import numpy as np
import json, argparse, os
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox


def main():

    initParser = argparse.ArgumentParser(description='create_videos')
    initParser.add_argument('-s','--save_dir', help='directory to save data')
    initargs = initParser.parse_args()
    savedir = initargs.save_dir

    if not os.path.exists(savedir+'/videos/'):
        os.makedirs(savedir+'/videos/')
    
    if os.path.isfile(savedir+"/parameters.json"):
        with open(savedir+"/parameters.json") as jsonFile:
              parameters = json.load(jsonFile)
    
    T         = parameters["T"]        # final time
    dt_dump   = parameters["dt_dump"]
    n_steps   = int(parameters["n_steps"])  # number of time steps
    dt        = T / n_steps     # time step size
    n_dump    = round(T / dt_dump)
    
    mx        = np.int32(parameters["mx"])
    my        = np.int32(parameters["my"])
    dx        = np.float32(parameters["dx"])
    dy        = np.float32(parameters["dy"])
    Lx        = mx*dx
    Ly        = my*dy
    
    #setup a meshgrid
    tol = 0.001
    
    x   = np.linspace(0+tol, Lx-tol, mx)
    y   = np.linspace(0+tol, Ly-tol, my)
    xv, yv  = np.meshgrid(x,y, indexing='ij')
    
    times = np.arange(0, n_dump, 1)*dt_dump

    
    figrho, axrho= plt.subplots(figsize=(12, 8), ncols=1)

    n=1
    
    rho = np.loadtxt(savedir+'/data/'+'rho.csv.{:d}'.format(n), delimiter=',')
    
    crho = [axrho.pcolormesh(xv, yv, rho, cmap='viridis', vmin=0, vmax=2)]

    figrho.colorbar(crho[0])
    axrho.set_title(r"$\phi$")
    
    tbaxrho = figrho.add_axes([0.2, 0.93, 0.04, 0.04])
    tbrho = TextBox(tbaxrho, 'time')
    
    def plt_snapshot_rho(val):        
        rho = np.loadtxt(savedir+'/data/'+'rho.csv.{:d}'.format(val), delimiter=',')
        
        crho[0].set_array(rho)
        tbrho.set_val(round(times[val],2))
        
        figrho.canvas.draw_idle()

    
    from matplotlib.animation import FuncAnimation
    animrho = FuncAnimation(figrho, plt_snapshot_rho, frames = n_dump, interval=100, repeat=True)
    animrho.save(savedir+'/videos/'+'rho.mp4')


if __name__=="__main__":
    main()
