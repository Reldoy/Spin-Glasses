import numpy as np
import numba as nb
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib import animation
from time import time
from tqdm import tqdm
from multiprocessing import Pool
import os
import sys

""" 
Class for simulating spin glass systems, in particular
the Edwards-Anderson, or RBIM model.
"""


@nb.njit(error_model='numpy')
def magnetization(config):
    """ Computes total absolute magnetization of config """
    return abs(np.sum(config))

@nb.njit(error_model='numpy')
def energy(config, bonds, L, n):
    """ Computes energy of spin configuration """
    e_sum = 0.0
    for i in range(L):
        for j in range(L):
            e_sum += config[i,j] * (bonds[i,j,0] * config[n[i,j,0,0], n[i,j,0,1]] + \
                                    bonds[i,j,1] * config[n[i,j,1,0], n[i,j,1,1]] + \
                                    bonds[n[i,j,2,0], n[i,j,2,1], 0] * config[n[i,j,2,0], n[i,j,2,1]] + \
                                    bonds[n[i,j,3,0], n[i,j,3,1], 1] * config[n[i,j,3,0], n[i,j,3,1]])
    return -0.5 * e_sum #TODO Divide by 4?

@nb.njit(error_model='numpy')
def metropolis(config, bonds, L, boltz, n):
    """ Local update Monte Carlo method """
    for _ in range(L*L):
        i = np.random.randint(L)
        j = np.random.randint(L)
        dE = bonds[i,j,0] * config[n[i,j,0,0], n[i,j,0,1]] + \
             bonds[i,j,1] * config[n[i,j,1,0], n[i,j,1,1]] + \
             bonds[n[i,j,2,0], n[i,j,2,1], 0] * config[n[i,j,2,0], n[i,j,2,1]] + \
             bonds[n[i,j,3,0], n[i,j,3,1], 1] * config[n[i,j,3,0], n[i,j,3,1]]
        dE *= 2 * config[i,j]
        if np.random.rand() < min(1.0, boltz**dE):
            config[i, j] *= -1   # Spin flip
    return config

@nb.njit(error_model='numpy')
def wolff(config, bonds, L, beta, n):
    ri = np.random.randint(L)
    rj = np.random.randint(L)
    cluster = []
    cluster.append((ri,rj))
    visited = []
    while len(cluster) > 0:
        si, sj = cluster[np.random.randint(len(cluster))] # choose random site in cluster
        n = [((si+1)%L,sj), ((si-1)%L,sj), (si,(sj+1)%L), (si,(sj-1)%L) ]
        J_s = [bonds[si,sj,0],
               bonds[si,sj,1],
               bonds[n[1][0], n[1][1], 0],
               bonds[n[3][0], n[3][1], 1]] # bonds connected to site s
        for k in range(4): # Loop over neighbors
            boltz = max(0, 1-np.exp(-2*beta*J_s[k]*config[si,sj]*config[n[k][0],n[k][1]]))# Update for each candidate spin
            if n[k] not in cluster and n[k] not in visited and np.random.rand() < boltz:
                cluster.append(n[k]) # TODO: vectorize
        config[si, sj] *= -1
        cluster.remove((si, sj))
        visited.append((si, sj))
    return config

@nb.njit(error_model="numpy",parallel=True) # TODO: fix parallelization
def run_simulation(bonds, L, T, neighbors, eqSteps, mcSteps):
    """ Simulator copied from wolffEndeavorStackexchange.py
        Consider using this going forward, make your own """
    nt=T.shape[0]

    E,M,C,X,B = np.empty(nt), np.empty(nt), np.empty(nt), np.empty(nt), np.empty(nt)

    n1, n2  = 1.0/(mcSteps*L*L), 1.0/(mcSteps*mcSteps*L*L)
    #Computation time heavily temperature dependent
    #Shuffle values to equalize workload
    #np.random.shuffle unsupported by Numba, use Python callback
    with nb.objmode(ind='int32[::1]'): #what's this?
        ind = np.arange(nt)
        np.random.shuffle(ind)

    ind_rev=np.argsort(ind)
    T=T[ind]

    for tt in nb.prange(nt):
        config = 1 -2*np.random.randint(0, 2, (L, L))
        E1 = M1 = E2 = M2 = 0
        iT=1.0/T[tt]; iT2=iT*iT;
        boltz = np.exp(-iT)

        # If metro, do ising.boltz_arr[tt]
        # If wolff, do iT
        for i in range(eqSteps):           # equilibrate
            config=metropolis(config, bonds, L, boltz, neighbors)       # Monte Carlo moves/sweeps

        for i in range(mcSteps):
            config=metropolis(config, bonds, L, boltz, neighbors)            
            Ene = energy(config, bonds, L, neighbors)     # calculate the energy
            Mag = magnetization(config)   # calculate the magnetisation

            E1 += Ene
            M1 += Mag
            M2 += Mag*Mag
            #M4 += M2*M2
            E2 += Ene*Ene

        E[tt] = n1*E1
        M[tt] = n1*M1
        C[tt] = (n1*E2 - n2*E1*E1)*iT2
        X[tt] = (n1*M2 - n2*M1*M1)*iT
        #B[tt] = 1.0 - n2*M4/(3.0*n1*M2*M2)
        #print ("Temp:",T[tt],":", E[tt], M[tt],C[tt],X[tt])

        #undo the shuffling
    return E[ind_rev],M[ind_rev],C[ind_rev],X[ind_rev], B[ind_rev]

def phase_transition(T,C,X):
    """ Yields estimates of critical temperature
        based on divergences of heat capacity and susceptibility.
        May not be appropriate for nontrivial p. """
    C_maxind = np.argmax(C)
    X_maxind = np.argmax(X)
    T_c_C = T[C_maxind]
    T_c_X = T[X_maxind]
    return T_c_C, T_c_X

def Nishimori(p):
    """ Nishimori line for Bernoulli distribution, valid for 0<p<1/2 """
    return -2. / np.log(1./(1-p) - 1.)

def Tc_exact():
    """ Lars Onsager's solution for
        the critical temperature """
    return 2./np.log(1+np.sqrt(2))

def domain_wall(config):
    """ Computes Domain Wall Free Energy,
        i.e. energy cost of introducing a domain wall
        TO BE IMPLEMENTED """
    pass

def movie_maker(configs, fig=None, ax=None):
    """ Display animation using input figure + axes.
        if none given, create custom
        TODO save video """
    if fig == None and ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    im = ax.imshow(configs[0], animated=True)

    def init():
        im.set_array(configs[0])
        return im,

    def update_frame(i):
        im.set_array(configs[i])
        return im,

    anim = animation.FuncAnimation(fig,
                                   update_frame,
                                   init_func=init,
                                   frames=len(configs),
                                   interval=20,
                                   blit=True
                                   )
    # anim.save('ising_animation.mp4', writer="ffmpeg")
    return anim

# ------------- CLASS --------------------

class SpinGlass:
    def __init__(self, L=10, J=1.0, p=0.0):
        """
        L: Lattice size
        J: Interaction strength
        p: antiferromagnetic bond density
        """
        # TODO: is this necessary?
        self.bond_rng = sc.stats.bernoulli(p)  # Initialize bernoulli generator
        self.L = L
        self.N_sites = self.L**2
        self.J = J

    def setup(self, T_start, T_end, N_T, p=0.0):
        """
        Initialize spin configuration using
        uniformly distributed +1 and -1 spins

        Initialize bonds in (2L,2L) matrix of +J, -J.
        Spin i,j is conventionally associated with horizontal right link (2i,2j)
        and vertical down link (2i,2j+1).
        The other two links will be tied to different spin representatives.

        Construct boltzmann lookup table
        
        Construct nearest neighbor lookup table
        """
        self.randomize_spins()
        self.randomize_bonds(p)
        T_arr = np.linspace(T_start, T_end, N_T)  # Temperatures
        # Make variables available to rest of class
        self.T_arr = T_arr
        self.construct_boltzes(self.T_arr)
        self.construct_neighbors(self.L)

    def construct_neighbors(self, L):
        """ Makes array/lookup table where
            first two indices represent site,
            latter two a 4x2 matrix of said site's neighbors.
            Ordering is clockwise starting with RIGHT"""
        Nmat = np.zeros((L, L, 4, 2), dtype=int)
        for i in range(L):
            for j in range(L):
                Nmat[i, j] = np.asarray([[(i + 1) % L, j],
                                         [(i - 1) % L, j],
                                         [i, (j + 1) % L],
                                         [i, (j - 1) % L]])
        self.neighs = Nmat

    def construct_boltzes(self, T_arr):
        """ For input temperature array,
            outputs lookup array of Boltzmann factors """
        boltz_arr = np.empty((len(T_arr), 6))
        dE_flip = self.J * \
            np.array([-8, -4, 0, 4, 8]
                     )  # Possible energy costs of random spin flips
        # Boltzmann factor lookup table
        boltz_arr[:,:-1] = np.exp(-np.outer(1./T_arr, dE_flip))
        boltz_arr[boltz_arr > 1.0] = 1.0  # Refinement
        boltz_arr[:, 2] = 1.0  # Deal with nans from div by zero
        boltz_arr[:,-1] = 1.0 - np.exp(-2.*self.J/T_arr) #For use in cluster build
        self.boltz_arr = boltz_arr   #For use in metropolis

    def randomize_spins(self):
        """ Initializes or resets spin configuration to spins that are \pm 1 with prob 0.5 """
        self.config = 1 -2*np.random.randint(0, 2, (self.L, self.L))

    def randomize_bonds(self, p):
        """ Initializes or resets bond configuration to Bernoulli distributed bonds  """
        self.bond_rng = sc.stats.bernoulli(p)  # Update distr
        self.bonds = 1 - 2*self.J*self.bond_rng.rvs((self.L, self.L, 2))

    def thermo_plotter(self, T, E, M, C, X, fig=None, ax=None):
        """ Displays a plot of EMCXB
            Maybe introduce an "add_to_plot" method """
        if fig == None and ax == None:
            fig, ax = plt.subplots(2, 3, figsize=(12, 6))
        # ------ ENERGY ------- #
        ax[0][0].plot(T, E, '-')
        ax[0][0].set_title("Energy")
        # ----MAGNETIZATION-----#
        ax[0][1].plot(T, M, '-')
        ax[0][1].set_title("Magnetization")
        # ---SP HEAT CAPACITY---#
        ax[1][0].plot(T, C, '-')
        ax[1][0].set_title("Heat capacity")
        # ---SP SUSCEPTIBILITY--#
        ax[1][1].plot(T, X, '-')
        ax[1][1].set_title("Susceptibility")
        # ---DOMAIN WALL FREE ENERGY---#
        # ax[1][2].plot(self.T_arr, self.DWF, '-')
        # ax[1][2].set_title("Domain Wall Energy")
        # --------CONFIG--------#
        #ax[0][2].imshow(self.config)
        #ax[0][2].set_title("Spins")
        plt.savefig(f"thermo_data.png")
        plt.show()

if __name__ == '__main__':
    """
    TODO:
    - Do user input
    - Test antiferro

    p=0.0 benchmark:
        L=16: (2.3265306122448983, 2.416326530612245)
        L=32: (2.2816326530612248, 2.3265306122448983)
    """
    # --------EQUILIBRATION--------
    # ising = SpinGlass(L=64)
    # ising.setup(1.5, 3.0, 1, p=0.1)
    # eqSteps = 2000
    # E = np.zeros((eqSteps))
    # t1 = time()
    # for _ in range(eqSteps):           # equilibrate
    #     E[_] = energy(ising.config,
    #                   ising.bonds,
    #                   ising.L,
    #                   ising.neighs)
    #     config=metropolis(ising.config,
    #                       ising.bonds,
    #                       ising.L,
    #                       np.exp(-1./1.1),
    #                       ising.neighs)       # Monte Carlo moves
    # plt.plot(E)
    # --------- SIMULATION --------
    # ising = SpinGlass(L=64)
    # ising.setup(0.4, 3.0, 50, p=0.08)
    # t1 = time()
    # E, M, C, X, B = run_simulation(ising.bonds,
    #                                 ising.L,
    #                                 ising.T_arr,
    #                                 ising.neighs,
    #                                 eqSteps=2000,
    #                                 mcSteps=2000)
    # print("Runtime: ", time() - t1)
    # print(phase_transition(ising.T_arr, C, X))
    # ---------- BINDERS -----------
    # L_list = [4, 6, 8, 10, 12]
    # fig, ax = plt.subplots()
    # for L in L_list:
    #     ising = SpinGlass(L=L)
    #     ising.setup(0.8, 3.0, 50, p=0.0)
    #     t1 = time()
    #     E, M, C, X, B = run_simulation(ising.bonds,
    #                                 ising.L,
    #                                 ising.T_arr,
    #                                 ising.neighs,
    #                                 eqSteps=2**16,
    #                                 mcSteps=2**16,
    #                                 APBC=False)
    #     ax.plot(ising.T_arr, B)
    # print(time() - t1)
    # plt.show()
    print("Exited successfully")
