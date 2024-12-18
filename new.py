#!/Users/stefano/anaconda3/bin/python
import numpy as np

#storage of atomic masses values
atomic_mass = { 'H':1.008,
                'C':12.011,
                'O':15.999,
                'N':14.0067,
                'Al':26.9815,
                "He" : 4.00260,
                "Li" : 7.0,
                "Be" : 9.0121,
                "B"  : 10.81,
                "F"  : 18.9984,
                "Ne" : 20.180,
                "Na" : 22.9897,
                "Mg" : 24.305,
                "Si" : 28.085,
                "P"  : 30.97376,
                "S"  : 32.07,
                "Cl" : 35.45,
                "Ar" : 39.9,
                "K"  : 39.0983,
                "Ca" : 40.08,
                "Fe" : 55.84,
                "Cu" : 63.55,
                "Zn" : 65.4,
                "Br" : 79.90,
                "Ag" : 107.868,
                "Sn" : 118.71,
                "I"  : 126.9045,
                "Xe" : 131.29,
                "Cs" : 132.9}

def update(cm, sum_cm, res_name, res_names_frame):
    cm = np.append(cm, np.reshape(sum_cm[0:3] / sum_cm[3], (1, 3)), axis=0)
    res_names_frame.append(res_name)
    return cm, res_names_frame

class trajectory:
    def __init__(self, filename, format="gro"):
        self.format = format  # later it will be use to implement different trajectory file formats
        self.logger = Logger()
        self.traj = self.load_gro(filename) # array [x,y,z] (n_frames, n_mols), [res_name] (1, n_mols)
        self.msd = []                       # array [msd] (1,t)

    def update(self):
        cm = np.append(cm, np.reshape(sum_cm[0:3] / sum_cm[3], (1, 3)), axis=0)
        res_names_frame.append(res_name)
        return cm, sum_cm, masses, res_name, res_names_frame
    def load_gro(self,filename):
        #trajectory loading routine
        #GROMACS gro format trajectory
        #   output:
        #   data (n_frame, n_mols, xyz), residue names (n_frames,n_mols)
        self.logger.print(f"Trajectory loading: {self.format} format")
        try:
            with open(filename, 'r') as file:
                # repeat for every frame up to the EOF
                CM = []         # it will append the center of mass for each residue
                res_names = []  # array of every frame
                while True:
                    read_line(file)                         # first line is a comment
                    n_atoms = int(read_line(file).strip())  #atoms in the frame
                    sum_cm = np.zeros(4)                    # x,y,z, molecular mass
                    res_num_count = 1                       # counter of residues, starts from 1
                    cm = np.empty((0, 3), float)        # x,y,z  one for each residue in the frame
                    res_names_frame = []                    # residue names of the frame
                    res_name_last = []                      #keeps resname in memory
                    for i in range(0, n_atoms): #atoms in frame
                        x, y, z, res_num, res_name, atom_name = parsline(file)
                        if not res_name_last: #if empty(first atom)
                            res_name_last = res_name
                        if res_num_count == res_num:  # same residue
                            sum_cm[0] += x * atomic_mass[atom_name]
                            sum_cm[1] += y * atomic_mass[atom_name]
                            sum_cm[2] += z * atomic_mass[atom_name]
                            sum_cm[3] += atomic_mass[atom_name]
                            if i == n_atoms - 1:  # last line before end of frame
                                cm, res_names_frame = update(cm, sum_cm, res_name, res_names_frame)
                            else:
                                pass
                        else:  # new molecule found
                            res_num_count = res_num            # update the counter
                            cm, res_names_frame = update(cm, sum_cm, res_name, res_names_frame)
                            #restart sum_cm
                            res_name_last = res_name        #update memory for name
                            sum_cm = np.array(
                                [x * atomic_mass[atom_name], y * atomic_mass[atom_name], z * atomic_mass[atom_name],
                                 atomic_mass[atom_name]])  # start new molecule
                            if i == n_atoms - 1: #if last residue is only one atom
                                cm, res_names_frame = update(cm, sum_cm, res_name, res_names_frame)
                    read_line(file)  # last line have cell dimensions
                    res_names.append(res_names_frame)
                    CM.append(cm)
        except End_of_Loop:
            CM = np.array(CM)
            res_names = np.array(res_names)
            self.logger.print(f"Total Frame Processed: {CM.shape[0]}\nNumber of Molecules: {CM.shape[1]}\nMolecular Types: {np.unique(res_names)}")
            return np.array(CM), np.array(res_names)

class Logger():
    def __init__(self):
        import logging
        self.logger = logging.getLogger('log')
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        if not len(self.logger.handlers):
            consoleHandler = logging.StreamHandler()
            consoleHandler.setLevel(logging.INFO)
            consoleHandler.setFormatter(formatter)
            file_handler = logging.FileHandler('logs.log')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(consoleHandler)
    def print(self,message):
        self.logger.info(message)

class End_of_Loop(Exception): pass #To being able of closing the loop outside the loop functions

def parsline(file):
    line=file.readline()
    if not line:
        raise End_of_Loop
    else:
        line = line.strip('\n')
        #gromacs line format "%5d%-5s%5s%5d%8.3f%8.3f%8.3f%8.4f%8.4f%8.4f"
        res_num = int(line[0:5])
        res_name, atom_name  = [ line[c:c+5].strip()  for c in range(5,5*2+1,5)]
        atom_name = ''.join([ i for i in atom_name if not i.isdigit()]) #remove digits from the atom name
        atom_num = int(line[5*3:5*3+5])
        x, y, z = [ float(line[c:c+8])  for c in range(5*4,5*4+8*3,8)]
        #end gromacs
        return x, y, z, res_num, res_name, atom_name

def read_line(f):
    #Is needed to check when the file finish and the exception need to being handled
    line=f.readline()
    #check is line is empty meaning that the file end is reach
    if not line:
        raise End_of_Loop
    else:
        return line

def msd_ii(coord,slice_dimension,skip=0):
    #Self diffusion (i=j) MSD
    #coord from the same molecule residue type. Have [Frame, N, 3] shape
    #skip are the line skipped while sliding the window matrix
    #slice_dimension is the correlation depth (in frame lines number) of the sliding window
    import numpy as np
    #MSD Vectorization
    f, n = coord[:,:,0].shape                                       # number of columns n (molecules) and number of frames f
    #  sliding windows dimensions : frames, N
    n_windows = ((f - slice_dimension) // (skip + 1 )) + 1      # +1 is needed for taking into account the first frame
    R=np.zeros((n_windows,n*slice_dimension))
    #loop repeated for the 3 component [X,Y,Z]
    for i in range(coord.shape[-1]):
        X = coord[:,:,i]
        first_idx = np.arange(slice_dimension*n)                      # First index array of the sliding window, flattened
        idx_matrix = first_idx[None,:] + n*(skip+1)*np.arange(n_windows)[:,None]
        windows = X.flatten()[idx_matrix]
        X0 = - windows[:,:n]
        X0 = np.hstack([X0 for i in range(slice_dimension)])
        r = np.sum((windows,X0),axis=0)
        r_quad = np.multiply(r,r)
        R = np.sum((R,r_quad),axis=0)
    #mean over all the windows
    msd = np.mean(R,axis=0).reshape(slice_dimension,n)
    #mean over all molecules
    msd = np.mean(msd,axis=1)
    #Mean square deviation in nm^2, instead t needs to be ask to the user
    return msd*1000000 #nm^2 -> pm^2

def regression(msd, t, scaling=0.3):
    #msd in pm^2
    #t in ps
    from sklearn.linear_model import LinearRegression
    # Use skitlearn tools for making the linear regression
    # it takes the subset of time vector and MSD to perform linear regression
    idx = int(len(t) * scaling)
    t_pred = t[idx:].reshape((-1, 1))
    msd_subset = msd[idx:]
    # LINEAR REGRESSION
    model_c = LinearRegression().fit(t_pred, msd_subset)
    slope = model_c.coef_
    D = model_c.coef_ / 6
    intercept = model_c.intercept_
    log = Logger()
    log.print(f"regression depth : {100 - scaling * 100} %")
    log.print(f"slope : {model_c.coef_} ")
    log.print(f"Intercept: {model_c.intercept_} ")
    log.print(f"D : {model_c.coef_ / 6} pm^2/ps \n")
    # generate msd_predition point for plotting with the same spacing and interval of t_prediction
    msd_pred = model_c.predict(t_pred)
    print(D)
    return [msd_pred, t_pred], [slope, intercept]

if __name__ == "__main__":
    #TESTING
    obj = trajectory("test_files/#test.gro.1#")
    coord, res_names = obj.traj
    #print("Matrix: (Frame, Molecules, Coordinates)")
    #print(coord.shape)
    #print("--------")
    #print(res_names)
    #print(coord)
    #import numpy as np
    #MSD = msd_ii(coord[:,res_names=="emi",:],slice_dimension=100,skip=0)
    #t=np.arange(len(MSD))
    #print(MSD)
    #print("__________")
    #msd_pred, var_regression =regression(MSD, t)
    #PLOTTING
    # import matplotlib.pyplot as plt
    # plt.plot(t,MSD, label="MSD")
    # import pandas as pd
    # travis = pd.read_csv("test_files/msd_C6H11N2_#2.csv", header=0, delimiter=";")
    # travis.columns=["r","msd","int"]
    # print(travis)
    # plt.plot(travis["r"],travis["msd"],label="TRAVIS")
    # plt.legend()
    # plt.show()