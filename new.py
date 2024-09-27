#!/Users/stefano/anaconda3/bin/python

atomic_mass = { 'H':1.008, 'C':12.011, 'O':15.999, 'N':14.0067, 'Cl':35.453, 'Al':26.9815, 'F':19.9984 }

class trajectory:
    def __init__(self, filename, format="gro"):
        self.traj = []              # array [x,y,z,q,atom]
        self.filename = filename    #trajectory filename
        self.format = format        # later it will be use to implement different trajectory file formats
        self.logger = Logger()
    def load(self):
        #trajectory loading routine
        import numpy as np
        try:
            with open(self.filename, 'r') as file:
                # repeat for every frame up to the EOF
                CM = []  # easier solution for appending np.arrays to an empty array without defining dimension first-hand.
                while True:
                    read_line(file)  # first line is a comment
                    n_atoms = int(read_line(file).strip())
                    sum_cm = np.zeros(4)  # x,y,z, mass
                    res_num_count = 1
                    cm = np.empty((0, 3), float)  # x,y,z
                    res_names = []
                    for i in range(0, n_atoms):
                        x, y, z, res_num, res_name, atom_name = parsline(file)
                        if res_num_count == res_num:  # same molecule
                            sum_cm[0] += x * atomic_mass[atom_name]
                            sum_cm[1] += y * atomic_mass[atom_name]
                            sum_cm[2] += z * atomic_mass[atom_name]
                            sum_cm[3] += atomic_mass[atom_name]
                            if i == n_atoms - 1:  # last line before end of frame
                                cm = np.append(cm, np.reshape(sum_cm[0:3] / sum_cm[3], (1, 3)), axis=0)
                                res_names.append(res_name)
                            else:
                                pass
                        else:  # new molecule
                            res_num_count = res_num  # update the counter
                            cm = np.append(cm, np.reshape(sum_cm[0:3] / sum_cm[3], (1, 3)),
                                           axis=0)  # save center of mass for the molecule
                            sum_cm = np.array(
                                [x * atomic_mass[atom_name], y * atomic_mass[atom_name], z * atomic_mass[atom_name],
                                 atomic_mass[atom_name]])  # start new molecule
                            res_names.append(res_name)
                    read_line(file)  # last line is a cell dimensions
                    CM.append(cm)
        except End_of_Loop:
            self.logger.print("End of file reached")
            self.traj = np.array(CM), np.array(res_names)


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




def msd(coord,slice_dimension,jump):
    import numpy as np
    #MSD Vectorization
    n = coord[:,:,0].shape[1]                                       # number of columns (molecules)
    slice_dimension = 3                                                           # number or frames in the sliding windows
    jump = 2                                                        # frame skipped while sliding the windows
    window_dim = (slice_dimension, n)                                             #  sliding windows dimensions : frames, N
    n_windows = ((coord.shape[0] - window_dim[0]) // jump) + 1      # +1 is needed for taking into account the first frame
    print("Original Matrix:",coord[:,:,0].shape,"\tsliding window:",window_dim,"\tf:",slice_dimension,"\tjump:",jump, "\tn_windows:",n_windows)

    R=np.zeros((n_windows,n*slice_dimension))
    #loop repeated for the 3 component [X,Y,Z]
    for i in range(coord.shape[-1]):
        X = coord[:,:,i]
        first_idx = np.arange(window_dim[0]*window_dim[1])                      # First index array of the sliding window, flattened
        idx_matrix = first_idx[None,:] + n*jump*np.arange(n_windows)[:,None]
        windows = X.flatten()[idx_matrix]
        X0 = - windows[:,0:n]
        #X0 = np.repeat(X0, slice_dimension, axis=1)
        X0 = np.hstack([X0 for i in range(slice_dimension)])
        print(X0[0,1])
        print(windows[0,1])
        print(",,,,,,PROBLEMA E' QUI!!!!! NON USA STACK HORIZONATALE")
        XR=np.sum((windows,X0),axis=0)
        XR = np.multiply(XR,XR)
        R=np.sum((R,XR),axis=0)
    #mean over all the windows

    msd = np.mean(R,axis=0).reshape(slice_dimension,n)
    print(msd)
    print("All first array should be zero!!!!!")
    #mean over all molecules
    msd = np.mean(msd,axis=1)
    #Mean square deviation in nm^2, instead t needs to be ask to the user
    print(msd)
    return msd


#TESTING
traj = trajectory("test_files/trj.gro")
traj.load()
coord, res_names = traj.traj
print(coord.shape)
print("--------")

MSD = msd(coord[:,res_names=="al2",:],3,2)
print("FIRST NUMBER WHOULD BE ZERO!!!!!")
