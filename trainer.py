#!/usr/bin/env python

import argparse
import os
from tqdm import tqdm 

import pprint as pp
import numpy as np
import matplotlib.pyplot as plt

import torch
print(torch.__version__)
import torch.optim as optim
import torch.autograd as autograd
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboard_logger import configure, log_value

from neural_combinatorial_rl import NeuralCombOptRL
from plot_attention import plot_attention


def str2bool(v):
      return v.lower() in ('true', '1')


if __name__ == "__main__":
    
    
    """parser = argparse.ArgumentParser(description="Neural Combinatorial Optimization with RL")
    
    # Data
    parser.add_argument('--task', default='sort_10', help="The task to solve, in the form {COP}_{size}, e.g., tsp_20")
    parser.add_argument('--batch_size', default=128, help='')
    parser.add_argument('--train_size', default=1000000, help='')
    parser.add_argument('--val_size', default=10000, help='')
    # Network
    parser.add_argument('--embedding_dim', default=128, help='Dimension of input embedding')
    parser.add_argument('--hidden_dim', default=128, help='Dimension of hidden layers in Enc/Dec')
    parser.add_argument('--n_process_blocks', default=3, help='Number of process block iters to run in the Critic network')
    parser.add_argument('--n_glimpses', default=2, help='No. of glimpses to use in the pointer network')
    parser.add_argument('--use_tanh', type=str2bool, default=True)
    parser.add_argument('--tanh_exploration', default=10, help='Hyperparam controlling exploration in the pointer net by scaling the tanh in the softmax')
    parser.add_argument('--dropout', default=0., help='')
    parser.add_argument('--terminating_symbol', default='<0>', help='')
    parser.add_argument('--beam_size', default=1, help='Beam width for beam search')
    
    # Training
    parser.add_argument('--actor_net_lr', default=1e-4, help="Set the learning rate for the actor network")
    parser.add_argument('--critic_net_lr', default=1e-4, help="Set the learning rate for the critic network")
    parser.add_argument('--actor_lr_decay_step', default=5000, help='')
    parser.add_argument('--critic_lr_decay_step', default=5000, help='')
    parser.add_argument('--actor_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--critic_lr_decay_rate', default=0.96, help='')
    parser.add_argument('--reward_scale', default=2, type=float,  help='')
    parser.add_argument('--is_train', type=str2bool, default=True, help='')
    parser.add_argument('--n_epochs', default=1, help='')
    parser.add_argument('--random_seed', default=24601, help='')
    parser.add_argument('--max_grad_norm', default=2.0, help='Gradient clipping')
    parser.add_argument('--use_cuda', type=str2bool, default=True, help='')
    parser.add_argument('--critic_beta', type=float, default=0.9, help='Exp mvg average decay')
    
    # Misc
    parser.add_argument('--log_step', default=50, help='Log info every log_step steps')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--run_name', type=str, default='0')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--epoch_start', type=int, default=0, help='Restart at epoch #')
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--disable_tensorboard', type=str2bool, default=False)
    parser.add_argument('--plot_attention', type=str2bool, default=False)
    parser.add_argument('--disable_progress_bar', type=str2bool, default=False)
    
    args = vars(parser.parse_args())"""
    
    
    
    
    args = {
    'task': 'sort_10',
#    'task': 'tsp_50',
#    'task': 'tsp_20',
#    'task': 'tsp_5',
#    'task': 'highlowhigh_10',
    'batch_size': 12,
    'train_size': 10000,#000,#1000000,
    'val_size': 1000,#10000,
    # Network
    'embedding_dim': 128, #Dimension of input embedding
    'hidden_dim': 128,#Dimension of hidden layers in Enc/Dec')
    'n_process_blocks': 3, #Number of process block iters to run in the Critic network')
    'n_glimpses': 2, #No. of glimpses to use in the pointer network')
    'use_tanh': True,
    'tanh_exploration': 10, #Hyperparam controlling exploration in the pointer net by scaling the tanh in the softmax')
    'dropout': 0.,
    'terminating_symbol': '<0>',
    'beam_size': 1, #Beam width for beam search'
    
    # Training
    'actor_net_lr': 1e-4, #Set the learning rate for the actor network")
    'critic_net_lr': 1e-4, #Set the learning rate for the critic network")
    'actor_lr_decay_step': 1000,
    'critic_lr_decay_step': 5000,
    'actor_lr_decay_rate': 0.96,
    'critic_lr_decay_rate': 0.96,
    'reward_scale': 2.,
    'is_train': True,
    'n_epochs': 1,
    'random_seed': 24601,
    'max_grad_norm': 2.0, #Gradient clipping')
    'use_cuda': False,#True,
    'critic_beta': 0.9, #Exp mvg average decay')
    
    # Misc
    'log_step': 50, #Log info every log_step steps')
    'log_dir': 'logs',
    'run_name': '0',
    'output_dir': 'outputs',
    'epoch_start': 0, #Restart at epoch #')
    'load_path': '',
    'disable_tensorboard': False,
    'plot_attention': False,
    'disable_progress_bar': False
    }
    
    
    
    NUM_WORKERS=0 #Justdo for now since Windows is having issues with python multiprocessing... #!!!!!!!!!!!
    
    
    #For translated sorting task
    MAX_OFFSET=5 #0

    #For non-consecutive integer sorting task for "sort_{N}"
    #Draw N integers without replacement, but from {0, ..., N*SCALE -1}
    #instead of consecutive {0, ..., N -1}
    SCALE=5
    
    #Save figures of TSP tours (for 2D TSP only)
    SAVE_TSP_TOURS = True
    #Save rewards for every instance during training/validation
    SAVE_OUT = True
    
    #Type of critic to use in actor-ctiric method.
    #'EMA' for exponential mving average
    #'net' for neural network critic
    critic_type = 'EMA' #'net'
    
    
    #If using Proximal Policy Optimization (PPO)
    USE_PPO = False#True #False#True
    PPO_OBJECTIVE = 'vanilla' #'clipped'
    PPO_CLIPPED_EPSILON = .2
    PPO_ITERS_PER_STEP = 5

    
    
    # Pretty print the run args
    pp.pprint(args)
    
    # Set the random seed
    torch.manual_seed(int(args['random_seed']))
    
    # Optionally configure tensorboard
    if not args['disable_tensorboard']:
        configure(os.path.join(args['log_dir'], args['task'], args['run_name']))
    
    # Task specific configuration - generate dataset if needed
    task = args['task'].split('_')
    COP = task[0]
    data_dir = 'data/' + COP
    
    #!!!!!!!!!!!!!!!!!!! consider range of instance sizes for training
    INSTANCE_SIZE = int(task[1])
    INSTANCE_SIZE_FIXED = False
    INSTANCE_SIZE_MIN = 1 #Integer >=1
    INSTANCE_SIZE_MAX = 2*INSTANCE_SIZE
    
    
    
    if COP == 'sort':
        import tasks.sorting_task as sorting_task
        
        input_dim = 1
        reward_fn = sorting_task.reward
        train_fname, val_fname = sorting_task.create_dataset(
            int(args['train_size']),
            int(args['val_size']),
            data_dir,
            data_len=INSTANCE_SIZE,
            max_offset=MAX_OFFSET,
            scale=SCALE
            )
        training_dataset = sorting_task.SortingDataset(train_fname)
        val_dataset = sorting_task.SortingDataset(val_fname)
    elif COP == 'highlowhigh':
        import tasks.highlowhigh_task as highlowhigh_task
        
        input_dim = 1
        reward_fn = highlowhigh_task.reward
        train_fname, val_fname = highlowhigh_task.create_dataset(
            int(args['train_size']),
            int(args['val_size']),
            data_dir,
            data_len=INSTANCE_SIZE,
            max_offset=0,#MAX_OFFSET,
            scale=1#SCALE
            )
        training_dataset = highlowhigh_task.HighLowHighDataset(train_fname)
        val_dataset = highlowhigh_task.HighLowHighDataset(val_fname)        
    elif COP == 'tsp':
        import tasks.tsp_task as tsp_task
    
        input_dim = 2 #consider multiple dimensions...
        reward_fn = tsp_task.reward
        val_fname = tsp_task.create_dataset(
            problem_size=str(INSTANCE_SIZE),
            data_dir=data_dir)
        training_dataset = tsp_task.TSPDataset(train=True, size=INSTANCE_SIZE,
             num_samples=int(args['train_size']))
        val_dataset = tsp_task.TSPDataset(train=True, size=INSTANCE_SIZE,
                num_samples=int(args['val_size']))
    else:
        print('Currently unsupported task!')
        exit(1)
    
    # Load the model parameters from a saved state
    if args['load_path'] != '':
        print('  [*] Loading model from {}'.format(args['load_path']))
    
        model = torch.load(
            os.path.join(
                os.getcwd(),
                args['load_path']
            ))
        model.actor_net.decoder.max_length = INSTANCE_SIZE
        model.is_train = args['is_train']
    else:
        # Instantiate the Neural Combinatorial Opt with RL module
        model = NeuralCombOptRL(
            input_dim,
            int(args['embedding_dim']),
            int(args['hidden_dim']),
            INSTANCE_SIZE, # decoder len #!!!!!!!!!!!!!!!!!! will need to be dynamic
            args['terminating_symbol'],
            int(args['n_glimpses']),
            int(args['n_process_blocks']), 
            float(args['tanh_exploration']),
            args['use_tanh'],
            int(args['beam_size']),
            reward_fn,
            args['is_train'],
            args['use_cuda'])
    
    
    save_dir = os.path.join(os.getcwd(),
               args['output_dir'],
               args['task'],
               args['run_name'])    
    
    try:
        os.makedirs(save_dir)
    except:
        pass
    

    actor_optim = optim.Adam(model.actor_net.parameters(), lr=float(args['actor_net_lr']))
    
    actor_scheduler = lr_scheduler.MultiStepLR(actor_optim,
            range(int(args['actor_lr_decay_step']), int(args['actor_lr_decay_step']) * 1000,
                int(args['actor_lr_decay_step'])), gamma=float(args['actor_lr_decay_rate']))
    
    if critic_type=='net':
        critic_mse = torch.nn.MSELoss()
        critic_optim = optim.Adam(model.critic_net.parameters(), lr=float(args['critic_net_lr']))        
        critic_scheduler = lr_scheduler.MultiStepLR(critic_optim,
                range(int(args['critic_lr_decay_step']), int(args['critic_lr_decay_step']) * 1000,
                    int(args['critic_lr_decay_step'])), gamma=float(args['critic_lr_decay_rate']))
    
    training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
        shuffle=True, num_workers=NUM_WORKERS)
    
    validation_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=NUM_WORKERS)
    
    if critic_type=='EMA':
        critic_exp_mvg_avg = torch.zeros(1)
        beta = args['critic_beta']
    
    if args['use_cuda']:
        model = model.cuda()
        if critic_type=='net':
            critic_mse = critic_mse.cuda()
        if critic_type=='EMA':
            critic_exp_mvg_avg = critic_exp_mvg_avg.cuda()
    
    step = 0
    val_step = 0
    
    if not args['is_train']:
        args['n_epochs'] = '1'
     
    R_train = []
    R_val = []
    
    epoch = int(args['epoch_start'])
    for i in range(epoch, epoch + int(args['n_epochs'])):
        
        if args['is_train']:
            # put in train mode!
            model.train()
    
            # sample_batch is [batch_size x input_dim x sourceL]
            for batch_id, sample_batch in enumerate(tqdm(training_dataloader,
                    disable=args['disable_progress_bar'])):
    
    
                bat = Variable(sample_batch)
                if args['use_cuda']:
                    bat = bat.cuda()
    
                #If using Proximal Policy Optimization (PPO), reuse same data
                #for a few update steps (more data efficient).
                #**Note: this is a bit different because we have a stochastic 
                #decoding step in our model, so not a direct comparison between
                #old and new policy results...
                K_sub_iters = PPO_ITERS_PER_STEP if USE_PPO else 1
                for ppo_iter in range(K_sub_iters):
                    R, probs, actions, actions_idxs = model(bat)
                    # - R is [1 x batchsize] the reward per example in batch
                    # - probs is list of tensors. List len is instance_size, each 
                    #tensor is batchsize. The probability of each of the set elements
                    # - actions is list of tensors. List len is instance_size, each 
                    #tensor is batchsize x dimension. The action (which set element) to choose.
                    #e.g. for sorting, dimension is just 1 (an integer), vs. for 
                    #2D TSP, dimension is 2 (x,y).
                    #TRAINING analysis
                    #Optionally save out rewards"
                    if SAVE_OUT:
                        R_train.append(list(R.data.numpy()))
                          
                
                    if batch_id == 0:
                        critic_exp_mvg_avg = R.mean()
                    else:
                        critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * R.mean())
        
                    advantage = R - critic_exp_mvg_avg
#                    print('advantage',advantage)
#                    print('R',R)
#                    print('critic_exp_mvg_avg',critic_exp_mvg_avg)
                    
                    logprobs = 0
                    nll = 0
                    for prob in probs: 
                        # compute the sum of the log probs
                        # for each tour in the batch
                        logprob = torch.log(prob)
                        nll += -logprob
                        logprobs += logprob
                   
                    # guard against nan
                    nll[(nll != nll).detach()] = 0.
                    # clamp any -inf's to 0 to throw away this tour
                    logprobs[(logprobs < -1000).detach()] = 0.
        

                    #PPO:
                    if USE_PPO:
                        if ppo_iter==0:
                            logprobs_prev=logprobs
                        ratio = logprobs / logprobs_prev
                        if PPO_OBJECTIVE == 'vanilla':
                            actor_loss = ratio * advantage
                        elif PPO_OBJECTIVE == 'clipped':
                            PPO_CLIPPED_EPSILON
                            actor_loss = torch.min(advantage*ratio,
                                                   advantage*torch.clamp(ratio,
                                                                         1.-PPO_CLIPPED_EPSILON,
                                                                         1.+PPO_CLIPPED_EPSILON)
                                                   )
                        
                        
                        actor_loss = actor_loss.mean()
                    
                    #vs. regular old actor-critic:
                    elif not USE_PPO: #or iter = =0 ???????????
                        # multiply each time step by the advanrate
                        reinforce = advantage * logprobs                        
                        actor_loss = reinforce.mean()
                    
                    #Keep the logprobs to compare to next iteration of policy
                    logprobs_prev = logprobs.clone()
                    
                    actor_optim.zero_grad()
                   
                    if not USE_PPO:
                        actor_loss.backward()
                        
                    elif USE_PPO:
                        if ppo_iter == K_sub_iters-1:
                            actor_loss.backward()
                        else:
                            actor_loss.backward(retain_graph=True)
                        
                        
                    # clip gradient norms
                    torch.nn.utils.clip_grad_norm_(model.actor_net.parameters(),
                            float(args['max_grad_norm']), norm_type=2)
        
                    actor_optim.step()
                    
                    #Only update learning rate once per PPO batch
                    #(on last sub iteration of PPO):
                    if ppo_iter == K_sub_iters-1:
                        actor_scheduler.step() #!!!!!!!!!!! Move move this outside this loop to have one step per epoch??
        
                    critic_exp_mvg_avg = critic_exp_mvg_avg.detach()
        
                    #critic_scheduler.step() #!!!!!!!!!!! Move move this outside this loop to have one step per epoch?? 
                    #!!!!!!! Maybe this is why he says his critic was bad? Too many lr steps since doing per batch, not per eopch?
        
                    #R = R.detach()
                    #critic_loss = critic_mse(v.squeeze(1), R)
                    #critic_optim.zero_grad()
                    #critic_loss.backward()
                    
                    #torch.nn.utils.clip_grad_norm_(model.critic_net.parameters(),
                    #        float(args['max_grad_norm']), norm_type=2)
        
                    #critic_optim.step()
                
                step += 1
                
                if not args['disable_tensorboard']:
                    log_value('avg_reward', R.mean().item(), step)
                    log_value('actor_loss', actor_loss.item(), step)
                    #log_value('critic_loss', critic_loss.item(), step)
                    log_value('critic_exp_mvg_avg', critic_exp_mvg_avg.item(), step)
                    log_value('nll', nll.mean().item(), step)
    
                if step % int(args['log_step']) == 0:
                    print('epoch: {}, train_batch_id: {}, avg_reward: {} +/= {}'.format(
                        i, batch_id, R.mean().item(), R.std().item()))
                    example_output = []
                    example_input = []
                    for idx, action in enumerate(actions):
                        if task[0] == 'tsp':
                            example_output.append(actions_idxs[idx][0].item())
                        else:
                            example_output.append(action[0].item())  # <-- ?? 
                        example_input.append(sample_batch[0, :, idx])
    
                    if SAVE_OUT:
                        for mm in range(10):
                            try:
                                np.save(os.path.join(save_dir,f'R_train_{step}.npy'),np.array(R_train))
                                break
                            except:
                                continue
                        #Clear the rewards lists                  
                        R_train = []
#                    print('Example train input: {}'.format(example_input))
                    print('Example train output: {}'.format(example_output))
                    
                    
                    #For TSP 2D, save figs of the tours
                    if task[0]=='tsp' and SAVE_TSP_TOURS:
                        x = [nn[0].item() for nn in example_input]
                        y = [nn[1].item() for nn in example_input]
                        plt.figure()
                        plt.title('Example 2D TSP Tour',fontsize=20)
                        for cc in range(len(example_output)):
                            plt.plot([x[cc],x[cc-1]],[y[cc],y[cc-1]],marker='o',color='k',linestyle='--')
                        #And the final leg:
                        plt.plot([x[example_output[-1]],x[example_output[0]]],
                                 [y[example_output[-1]],y[example_output[0]]],
                                 marker='o',color='k',linestyle='--')
                        #Tour start/end point:
                        plt.plot(x[example_output[0]],y[example_output[0]],marker='o',color='r')
                        for mm in range(10):
                            try:
                                plt.savefig(os.path.join(save_dir, f'TSP_Tour_train_{step}_0.png'))
                                break
                            except:
                                continue                        
                        plt.close('all')
                        
                    
                    #TRAINING analysis
                    #Optionally save some things for analysis:
                    #example_input
                    #example_output
                    #avg_reward [to get mean and variance]
            #        if SAVE_OUT:
            #            ...                    
    
    
    
        # Use beam search decoding for validation
        model.actor_net.decoder.decode_type = "beam_search"
        
        print('\n~Validating~\n')
    
        example_input = []
        example_output = []
        avg_reward = []
    
        # put in test mode!
        model.eval()
    
        for batch_id, val_batch in enumerate(tqdm(validation_dataloader,
                disable=args['disable_progress_bar'])):
            bat = Variable(val_batch)
    
            if args['use_cuda']:
                bat = bat.cuda()
    
            R, probs, actions, action_idxs = model(bat)
            
            if SAVE_OUT:
                R_val.append(list(R.data.numpy()))
                
            avg_reward.append(R[0].item())
            val_step += 1.
    
            if not args['disable_tensorboard']:
                log_value('val_avg_reward', R[0].item(), int(val_step))
    
            if val_step % int(args['log_step']) == 0:
                example_output = []
                example_input = []
                for idx, action in enumerate(actions):
                    if task[0] == 'tsp':
                        example_output.append(action_idxs[idx][0].item())
                        example_input.append(bat[0, :, idx])                        
                    else:
                        example_output.append(action[0].item())
                        example_input.append(bat[0, :, idx].item())

                print('Step: {}'.format(batch_id))
                print('Example test input: {}'.format(example_input))
                print('Example test output: {}'.format(example_output))
                print('Example test reward: {}'.format(R[0].item()))
        
                if SAVE_OUT:
                    for mm in range(10):
                        try:
                            np.save(os.path.join(save_dir,f'R_val_{step}.npy'),np.array(R_val))
                            break
                        except:
                            continue
                    #Clear the rewards lists                  
                    R_val = []  
            
                if args['plot_attention']:
                    probs = torch.cat(probs, 0)
                    plot_attention(example_input,
                            example_output, probs.data.cpu().numpy())
                    
                #For TSP 2D, save figs of the tours
                if task[0]=='tsp' and SAVE_TSP_TOURS:
                    x = [nn[0].item() for nn in example_input]
                    y = [nn[1].item() for nn in example_input]
                    plt.figure()
                    plt.title('Example 2D TSP Tour',fontsize=20)
                    for cc in range(len(example_output)):
                        plt.plot([x[cc],x[cc-1]],[y[cc],y[cc-1]],marker='o',color='k',linestyle='--')
                    #And the final leg:
                    plt.plot([x[example_output[-1]],x[example_output[0]]],
                             [y[example_output[-1]],y[example_output[0]]],
                             marker='o',color='k',linestyle='--')
                    #Tour start/end point:
                    plt.plot(x[example_output[0]],y[example_output[0]],marker='o',color='r')
                    for mm in range(10):
                        try:
                            plt.savefig(os.path.join(save_dir, 'TSP_Tour_val_{}.png'.format(int(val_step))))
                            break
                        except:
                            continue                      
                    plt.close('all')
                    
                    
        print('Validation overall avg_reward: {}'.format(np.mean(avg_reward)))
        print('Validation overall reward var: {}'.format(np.var(avg_reward)))
        
        

        #VALIDATION analysis
        #Optionally save some things for analysis:
        #example_input
        #example_output
        #avg_reward [to get mean and variance]
#        if SAVE_OUT:
#            ...

            
            
        if args['is_train']:
            model.actor_net.decoder.decode_type = "stochastic"
             
            print('Saving model...')
         
            torch.save(model, os.path.join(save_dir, 'epoch-{}.pt'.format(i)))
    
            # If the task requires generating new data after each epoch, do that here!
            if not INSTANCE_SIZE_FIXED:
                INSTANCE_SIZE = int(torch.randint(INSTANCE_SIZE_MIN,INSTANCE_SIZE_MAX,size=(1,)))#!!!!!!!!!!!
            if COP == 'tsp':
                training_dataset = tsp_task.TSPDataset(train=True, size=INSTANCE_SIZE,
                    num_samples=int(args['train_size']))
                training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
                    shuffle=True, num_workers=1)
            if COP == 'sort':
                train_fname, _ = sorting_task.create_dataset(
                    int(args['train_size']),
                    int(args['val_size']),
                    data_dir,
                    data_len=INSTANCE_SIZE,
                    max_offset=MAX_OFFSET,
                    scale=SCALE
                    )
                training_dataset = sorting_task.SortingDataset(train_fname)
                training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
                        shuffle=True, num_workers=1)
            if COP == 'highlowhigh':
                train_fname, _ = highlowhigh_task.create_dataset(
                    int(args['train_size']),
                    int(args['val_size']),
                    data_dir,
                    data_len=INSTANCE_SIZE,
                    max_offset=MAX_OFFSET,
                    scale=SCALE
                    )
                training_dataset = highlowhigh_task.HighLowHighDataset(train_fname)
                training_dataloader = DataLoader(training_dataset, batch_size=int(args['batch_size']),
                        shuffle=True, num_workers=1)                
