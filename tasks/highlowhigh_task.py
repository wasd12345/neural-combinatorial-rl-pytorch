# Generate high-low-high data and store in .txt
# Define the reward function 

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from tqdm import trange, tqdm
import os
import sys


def reward(sample_solution, USE_CUDA=False):
    """
    The reward for the high-low-high task is related to the Hamming Distance
    from the correct ordering. This makes it a purely supervised reward signal
    which is only applicable to small toy problems like this where an exact 
    solution can be obtained cheaply. Nonetheless, it is interesting to try.
    

    Input sequences must all be the same length.


    Args:
        sample_solution: list of len sourceL of [batch_size]
        Tensors
    Returns:
        [batch_size] containing trajectory rewards
    """
    
    batch_size = sample_solution[0].size(0)
    sourceL = len(sample_solution)

    longest = Variable(torch.ones(batch_size, 1), requires_grad=False)
    current = Variable(torch.ones(batch_size, 1), requires_grad=False)

    if USE_CUDA:
        longest = longest.cuda()
        current = current.cuda()

    #sequence reward
    for i in range(1, sourceL):
        #For even parity: [the "high" elements]
        if i%2==0:
            # compare solution[i-1] < solution[i] 
            res = torch.lt(sample_solution[i-1], sample_solution[i])
            # if res[i,j] == 1, increment length of current sorted subsequence
            current += res.float()
            # else, reset current to 1
            current[torch.eq(res, 0)] = 1
            #current[torch.eq(res, 0)] -= 1
            # if, for any, current > longest, update longest
            mask = torch.gt(current, longest)
            longest[mask] = current[mask]
        #For odd parity: [the "low" elements]
        elif i%2==1:
            # compare solution[i-1] > solution[i] 
            res = torch.gt(sample_solution[i-1], sample_solution[i])
            # if res[i,j] == 1, increment length of current sorted subsequence
            current += res.float()
            # else, reset current to 1
            current[torch.eq(res, 0)] = 1
            #current[torch.eq(res, 0)] -= 1
            # if, for any, current > longest, update longest
            mask = torch.gt(current, longest)
            longest[mask] = current[mask]            

    sequence_reward = -torch.div(longest, sourceL)


    #Hamming distance reward
    s = [mm for mm in range(sourceL)]
    inds = []
    par = 0
    for dd in range(sourceL):
        inds.extend([s.pop(par-1)])
        par = (par+1)%2
    
    #!!!!! ASSUMES all solutions in this batch are same length
    tt = torch.stack(sample_solution,dim=2)
    tt_sort, _ = torch.sort(tt,dim=2)
    gt = tt_sort[:,0,inds]
#    print(tt_sort)
#    print(gt)
    
    hamming_reward = -torch.div(torch.sum(torch.eq(tt,gt),dim=2).float(), float(sourceL))
#    print('sample_solution',sample_solution)
#    print('hamming_reward',hamming_reward)
    LAMBDA = .25
    final_reward = torch.lerp(sequence_reward, hamming_reward, LAMBDA)
#    print('final reward', final_reward)
    return final_reward


def create_dataset(
        train_size,
        val_size,
        #test_size,
        data_dir,
        data_len,
        seed=None,
        max_offset=0,
        scale=1
        ):

    if seed is not None:
        torch.manual_seed(seed)
    
    train_task = 'high-low-high-size-{}-len-{}-train.txt'.format(train_size, data_len)
    val_task = 'high-low-high-size-{}-len-{}-val.txt'.format(val_size, data_len)
    
    train_fname = os.path.join(data_dir, train_task)
    val_fname = os.path.join(data_dir, val_task)

    
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    else:
        if os.path.exists(train_fname) and os.path.exists(val_fname):
            return train_fname, val_fname
    
    train_set = open(os.path.join(data_dir, train_task), 'w')
    val_set = open(os.path.join(data_dir, val_task), 'w') 
    #test_set = open(os.path.join(data_dir, test_task), 'w')
    
    def to_string(tensor):
        """
        Convert a a torch.LongTensor 
        of size data_len to a string 
        of integers separated by whitespace
        and ending in a newline character
        """
        line = ''
        for j in range(data_len-1):
            line += '{} '.format(tensor[j])
        line += str(tensor[-1].item()) + '\n'
        return line
    
    print('Creating training data set for {}...'.format(train_task))
    
    # Generate a training set of size train_size
    for i in trange(train_size):
        x = torch.randperm(data_len*scale)
        x = x[:data_len]
        if max_offset>0:
            x += torch.randint(0,data_len,size=(1,)).long()
        train_set.write(to_string(x))

    print('Creating validation data set for {}...'.format(val_task))
    
    for i in trange(val_size):
        x = torch.randperm(data_len*scale)
        x = x[:data_len]
        if max_offset>0:
            x += torch.randint(0,data_len,size=(1,)).long()        
        val_set.write(to_string(x))

#    print('Creating test data set for {}...'.format(test_task))
#
#    for i in trange(test_size):
#        x = torch.randperm(data_len)
#        test_set.write(to_string(x))

    train_set.close()
    val_set.close()
#    test_set.close()
    return train_fname, val_fname

class HighLowHighDataset(Dataset):

    def __init__(self, dataset_fname):
        super(HighLowHighDataset, self).__init__()
       
        print('Loading training data into memory')
        self.data_set = []
        with open(dataset_fname, 'r') as dset:
            lines = dset.readlines()
            for next_line in tqdm(lines):
                toks = next_line.split()
                sample = torch.zeros(1, len(toks)).long()
                for idx, tok in enumerate(toks):
                    sample[0, idx] = int(tok)
                self.data_set.append(sample)
        
        self.size = len(self.data_set)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_set[idx]

if __name__ == '__main__':

    if int(sys.argv[1]) == 1:
        MAX_OFFSET=5
        SCALE=3
        create_dataset(1000, 100, 'data', 10, 123, max_offset=MAX_OFFSET, scale=SCALE)
    elif int(sys.argv[1]) == 2:

        hlh_data = HighLowHighDataset(os.pardir,'data', 'high-low-high-size-1000-len-10-train.txt',
            'high-low-high-size-100-len-10-val.txt')
        
        for i in range(len(hlh_data)):
            print(hlh_data[i])
