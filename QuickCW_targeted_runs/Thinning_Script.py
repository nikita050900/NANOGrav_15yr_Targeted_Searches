#Credit to Bence Bencsy

import sys
import h5py
import numpy as np

fileList = np.loadtxt('./results/3c66B_1bill_targeted_detect/3c66b_toThin.txt', dtype = 'str') #list of all files in data set
print(fileList)

for i in fileList:
    infile = i
    first_n_param = 8
    outfile = './results/3c66B_1bill_targeted_detect/THIN/thinned_'+i.split('/')[3]
    thin = 500
    
    print(infile)
    print(first_n_param)
    print(outfile)
    
    with h5py.File(infile, 'r') as f:
        Ts = f['T-ladder'][...]
        samples_cold = f['samples_cold'][:,:,:]
        print(samples_cold[-1].shape)
        log_likelihood = f['log_likelihood'][:1,:]
        print(log_likelihood.shape)
        par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
        acc_fraction = f['acc_fraction'][...]
        fisher_diag = f['fisher_diag'][...]
    
    with h5py.File(outfile, 'w') as f:
        f.create_dataset('samples_cold', data=samples_cold[:,::thin,:first_n_param], compression="gzip", chunks=True)
        f.create_dataset('log_likelihood', data=log_likelihood[:,:], compression="gzip", chunks=True)
        f.create_dataset('par_names', data=np.array(par_names, dtype='S'))
        f.create_dataset('acc_fraction', data=acc_fraction)
        f.create_dataset('fisher_diag', data=fisher_diag)
        f.create_dataset('T-ladder', data=Ts)


# i='./results/HS1630_1bill_targeted_detect/MassPriorChange/quickCW_15yr_HS1630+2355_1bill_targeteddetectNone.h5'
# infile = i
# first_n_param = 8
# outfile = './results/HS1630_1bill_targeted_detect/MassPriorChange/THIN/thin'+i.split('/')[4]

# print(infile)
# print(first_n_param)
# print(outfile)

# with h5py.File(infile, 'r') as f:
#     Ts = f['T-ladder'][...]
#     samples_cold = f['samples_cold'][:,:,:]
#     print(samples_cold[-1].shape)
#     log_likelihood = f['log_likelihood'][:1,:]
#     print(log_likelihood.shape)
#     par_names = [x.decode('UTF-8') for x in list(f['par_names'])]
#     acc_fraction = f['acc_fraction'][...]
#     fisher_diag = f['fisher_diag'][...]

# with h5py.File(outfile, 'w') as f:
#     f.create_dataset('samples_cold', data=samples_cold[:,::thin,:first_n_param], compression="gzip", chunks=True)
#     f.create_dataset('log_likelihood', data=log_likelihood[:,:], compression="gzip", chunks=True)
#     f.create_dataset('par_names', data=np.array(par_names, dtype='S'))
#     f.create_dataset('acc_fraction', data=acc_fraction)
#     f.create_dataset('fisher_diag', data=fisher_diag)
#     f.create_dataset('T-ladder', data=Ts)