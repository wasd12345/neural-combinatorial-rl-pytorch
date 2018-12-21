import matplotlib.pyplot as plt
import os
import numpy as np


if __name__ == '__main__':
    

    TASK = 'highlowhigh_10' #'tsp_5' #'tsp_50' #'sort_10'
    path = os.path.join('outputs',TASK,'0')
    #path = os.path.join('outputs','sort_10','0')
    
    
    files = os.listdir(path)
    files = [i for i in files if i.endswith('.npy')]
    s = [i.find('_',5)+1 for i in files]
    e = [i.find('.npy') for i in files]
    ints = [int(ff[s[i]:e[i]]) for i,ff in enumerate(files)]
    _ = np.argsort(ints)

    files = np.array(files)[_]
    train_files = [os.path.join(path,i) for i in files if i.startswith('R_train')]
    val_files = [os.path.join(path,i) for i in files if i.startswith('R_val')]
    

    for i, ff in enumerate(train_files):
        x = np.load(ff)
        if i==0:
            train = x
        else:
            train = np.vstack((train,x))
    

    plt.figure()
#    for hh in range(train.shape[1]):
#            plt.plot(train[:,hh])
    plt.plot(train.mean(axis=1),color='k')
    #plt.plot(np.median(train,axis=1),color='b')
    plt.title('TSP 50, Average Tour Length (Training)')
    plt.xlabel('Step')
    plt.ylabel('Average Tour Length')
    plt.show()

    
    







    for i, ff in enumerate(val_files):
        x = np.load(ff)
        if i==0:
            val = x
        else:
            val = np.concatenate((train,x))
    
    
    

    plt.figure()
    #plt.plot(train.mean(axis=1),color='k')
    plt.plot(val,color='k')
    plt.title('TSP 50, Average Tour Length (Validation)')
    plt.xlabel('Step')
    plt.ylabel('Average Tour Length')
    plt.show()    