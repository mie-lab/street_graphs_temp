# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 18:53:55 2019

@author: henry
"""
import os, sys
sys.path.append(os.getcwd())
import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)

    import torch
    from torch.optim.lr_scheduler import StepLR
    from datetime import datetime
    import numpy as np
    import time
    import pickle
    import sys, os
    sys.path.append(os.getcwd())
    sys.path.append(os.path.join(os.getcwd(), 'ye'))
    sys.path.append(os.path.join(os.getcwd(),'henry', 'graphs'))
    sys.path.append(os.path.join(os.getcwd(),'experiments_for_JMLR'))

    from videoloader import trafic4cast_dataset
    from kipfdepth1_config import config
    from visual_TB import Visualizer
    from earlystopping import EarlyStopping
    import json

    from graph_models import KipfNet
    from datetime import datetime

    from torch_geometric.data import Data
    import matplotlib.pyplot as plt

    from graph_utils import csr_to_torch, coo_to_torch, image_to_vector, \
     vector_to_image, create_adj_matrix, transform_shape_train, \
     transform_shape_test, blockify_A, blockify_data, unblockify_target, \
     retransform_unblockify_target

    from sklearn.neighbors import NearestNeighbors

def trainNet(model, train_loader, val_loader, val_loader_ttimes, \
                device, adj, nn_ixs, edge_index, coords=None):
        
    # Print all of the hyper parameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", config['dataloader']['batch_size'])
    print("epochs=", config['num_epochs'])
    print("learning_rate=", config['optimizer']['lr'])
    print("mask_threshold=", config['mask_threshold'])
    print("nh1=", config['model']['KipfNet']['nh1'])
    print("K=", config['model']['KipfNet']['K'])
    print("K_mix=", config['model']['KipfNet']['K_mix'])
    print("inout_skipconn=", config['model']['KipfNet']['inout_skipconn'])
    print("=" * 30)
    # define the optimizer & learning rate 
    #optim = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])#
    if config['optimizer_name'] == 'SGD':
        optim = torch.optim.SGD(model.parameters(), **config['optimizer'])
    elif config['optimizer_name'] == 'ADAM':
        optim = torch.optim.Adam(model.parameters(), **config['optimizer'])

    # scheduler = StepLR(optim, step_size=config['lr_step_size'], gamma=config['lr_gamma'])
    
    nh1 = config['model']['KipfNet']['nh1']
    K = config['model']['KipfNet']['K'] 
    K_mix = config['model']['KipfNet']['K_mix']
    inout_skipconn = config['model']['KipfNet']['inout_skipconn']

    log_dir = 'runs/graphs/' + 'KipfNet' + '_nh1=' + str(nh1) \
                + '_K=' + str(K) + '_Kmix=' + str(K_mix) \
                + '_skip_conn' + str(inout_skipconn) \
                + '_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") \
                + '-'.join(config['dataset']['cities'])
    writer = Visualizer(log_dir)

    # dump config file  
    with open(os.path.join(log_dir,'config.json'), 'w') as fp:
        json.dump(config, fp)

    # Time for printing
    training_start_time = time.time()
    globaliter = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config['patience'], verbose=True)
#    adj = adj.to(device)
   
    # Loop for n_epochs
    for epoch_idx, epoch in enumerate(range(config['num_epochs'])):
        
        writer.write_lr(optim, globaliter)

        # train for one epoch
        globaliter = train(model, train_loader, optim, device, writer, epoch, globaliter,
         adj, nn_ixs, edge_index, coords=coords)

        # At the end of the epoch, do a pass on the validation set
        # val_loss = validate(model, val_loader, device, writer, globaliter, adj, nn_ixs, edge_index)
        val_loss_testtimes = validate(model, val_loader_ttimes, device, writer,
                                globaliter, adj, nn_ixs, edge_index, if_testtimes=True, coords=coords)
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss_testtimes, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
        if config['debug'] and epoch_idx >= 0:
            break

        # scheduler.step()
        

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

#    model.load_state_dict(torch.load('runs/checkpoint.pt'))

    # remember to close
    writer.close()

def get_graphdata_obj(inputs, edge_index, y, num_features=38, num_classes=9):
    graphdata = Data(x=inputs, edge_index=edge_index, y=y)
    # graphdata.num_features = num_features
    # graphdata.num_classes = num_classes
    
    return graphdata
        

def train(model, train_loader, optim, device, writer, \
             epoch, globaliter, adj, nn_ixs, edge_index, coords=None):
    batch_size = config['dataloader']['batch_size']
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)
 
    # define start time
    start_time = time.time()
    

    for i, data in enumerate(train_loader, 0):
        
#        print("validation batch {}/{}".format(i,len(train_loader)))
        inputs, Y, features = data
        inputs = inputs/255
        globaliter += 1
        effective_batch_size = inputs.shape[0]
        # padd the input data with 0 to ensure same size after upscaling by the network
        # feature_vec = feature_vec['feature_vector'].float().to(device)

        if coords is not None:
            inputs = torch.cat((inputs, coords[0], coords[1]), 1)

        inputs = image_to_vector(inputs,nn_ixs)
        Y = image_to_vector(Y,nn_ixs)
        
        inputs, Y = blockify_data(inputs, Y, batch_size)
        Y = Y.float()
               
        # the Y remains the same dimension
        inputs = inputs.float().to(device) 
        Y = Y.float().to(device) 


        graphdata = get_graphdata_obj(inputs, edge_index, Y, num_features=36, num_classes=9)
        # Set the parameter gradients to zero
        optim.zero_grad()

        # Forward pass, backward pass, optimize
        prediction = model(graphdata)


        
        # crop the output for comparing with true Y
        loss_size = torch.nn.functional.mse_loss(prediction, Y)

        loss_size.backward()
        optim.step()

        # Print statistics
        running_loss += loss_size.item()
        if (i+1) % config['print_every_step'] == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch+1, int(100 * (i+1) / n_batches), running_loss / config['print_every_step'], time.time() - start_time))

            # write the train loss to tensorboard
            running_loss_norm = running_loss / config['print_every_step']
            writer.write_loss_train(running_loss_norm, globaliter)

            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()
            

    return globaliter


def validate(model, val_loader, device, writer, globaliter, adj, nn_ixs, 
            edge_index, if_testtimes=False, coords=None):
    batch_size = config['dataloader']['batch_size']
    total_val_loss = 0
    if if_testtimes:
        prefix = 'testtimes'
    else:
        prefix = ''

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):



            val_inputs_image, val_y1, feature_vec = data
            val_inputs_image = val_inputs_image/255
            # feature_vec = feature_vec['feature_vector'].float().to(device)

            # add coords
            if coords is not None:
                val_inputs_image = torch.cat((val_inputs_image, coords[0], coords[1]), 1)

            val_inputs_image = image_to_vector(val_inputs_image,nn_ixs)
            # tested            
            val_y2 = image_to_vector(val_y1,nn_ixs)
            
            val_inputs_block, val_y_block = blockify_data(val_inputs_image, val_y2, batch_size)
                   
            # the Y remains the same dimension
            val_inputs_block = val_inputs_block.float().to(device) 
            val_y_block = val_y_block.float().to(device) 
    
            val_graphdata = get_graphdata_obj(val_inputs_block, edge_index, val_y_block,
                                                num_features=36, num_classes=9)
            # Set the parameter gradients to zero
            # Forward pass, backward pass, optimize
            prediction_block = model(val_graphdata)
            
            # crop the output for comparing with true Y
            prediction_block = torch.clamp(prediction_block, 0, 255, out=None)
            val_loss_size = torch.nn.functional.mse_loss(prediction_block, val_graphdata.y)
            total_val_loss += val_loss_size.item()


            
            # each epoch select one prediction set (one batch) to visualize
            if i  == 0:
                val_output = retransform_unblockify_target(prediction_block.cpu().detach().numpy(),
                                                           nn_ixs=nn_ixs,
                                                           batch_size=batch_size,
                                                           dataset=val_loader.dataset)
                val_output = torch.from_numpy(val_output)
                
                val_y = retransform_unblockify_target(val_y_block.cpu().detach().numpy(),
                                                      nn_ixs=nn_ixs,
                                                      batch_size=batch_size,
                                                      dataset=val_loader.dataset )
                val_y = torch.from_numpy(val_y)
                
                

                # we have to unblockify the data before we can do the backtransformation from vector to image
                
                writer.write_image(val_output, globaliter,if_predict=True, if_testtimes=if_testtimes)
                writer.write_image(val_y, globaliter,if_predict=False, if_testtimes=if_testtimes)

    val_loss = total_val_loss / len(val_loader)
    print("Validation loss = {:.2f}".format(val_loss))
    # write the validation loss to tensorboard
    writer.write_loss_validation(val_loss, globaliter, if_testtimes=if_testtimes)
    return val_loss


if __name__ == "__main__":
    
    
    dataset_train = trafic4cast_dataset(split_type='training', **config['dataset'],
                                        reduce=True, filter_test_times=True)
    dataset_val = trafic4cast_dataset(split_type='validation', **config['dataset'], 
                                        reduce=True, filter_test_times=False)

    dataset_val_ttimes = trafic4cast_dataset(split_type='validation', **config['dataset'], 
                                        reduce=True, filter_test_times=True)

    
    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True,
                                               **config['dataloader'])
    val_loader = torch.utils.data.DataLoader(dataset_val, shuffle=True,
                                             **config['dataloader'])
    val_loader_ttimes = torch.utils.data.DataLoader(dataset_val_ttimes, shuffle=True,
                                             **config['dataloader'])
    
    print('start graph stuff')
#     define the network structure -- partial UNet
    n_features = 38
    h=495
    w=436
    b = config['dataloader']['batch_size']

    x_coords = np.expand_dims(np.arange(w), 0)
    x_coords = np.repeat(x_coords, h, 0)
    x_coords = np.expand_dims(x_coords, 0)
    x_coords = np.expand_dims(x_coords, 0)
    x_coords = np.repeat(x_coords, b, 0) / w
    
    y_coords = np.expand_dims(np.arange(h), 1)
    y_coords = np.repeat(y_coords, w, 1)
    y_coords = np.expand_dims(y_coords, 0)
    y_coords = np.expand_dims(y_coords, 0)
    y_coords = np.repeat(y_coords, b, 0) / h
    
    
    np_coords = np.asarray((np.squeeze(x_coords[0,...]),np.squeeze(y_coords[0,...])))
      
    x_coords = torch.from_numpy(x_coords).to(dtype=torch.float)
    y_coords = torch.from_numpy(y_coords).to(dtype=torch.float)
    coords = (x_coords,y_coords)
#
#
#   
#
#    
#    # 
#    # model = GUNet(num_features=36, num_classes=9).to(device)
#    # 
#    
##    model = GraphUNet(in_channels=36,hidden_channels=4,out_channels=9,depth=1).to(device)
#    
    adj, nn_ixs, G, mask = create_adj_matrix(city=config['dataset']['cities'][0],
                                mask_threshold=config['mask_threshold'])
#    
#    adj = adj.to(device)
    
    
    if config['dataloader']['batch_size'] > 1:
        adj = blockify_A(adj, config['dataloader']['batch_size'])
        
    device = torch.device(config['device_num'])
    edge_tuple = adj.nonzero()
    edge_array = np.stack(edge_tuple)
    edge_array = edge_array.astype(np.int64)
    edge_array_np = edge_array
    edge_index = torch.LongTensor(edge_array)
    edge_index = edge_index.to(device)
   
    

    print(config['device_num'])


    for nh1 in [16, 32, 48]:
        for K in [2, 4, 6, 8]:
            for K_mix in [1,2,4]:
                for inout_skipconn in [True, False]:
                    config['model']['KipfNet']['nh1'] = nh1
                    config['model']['KipfNet']['K'] = K
                    config['model']['KipfNet']['K_mix'] = K_mix
                    config['model']['KipfNet']['inout_skipconn'] = inout_skipconn


                    model = KipfNet(num_features=config['model']['in_channels'], 
                        num_classes=9,
                        **config['model']['KipfNet']).to(device)
                        
                   
                    try:
                        trainNet(model, train_loader, val_loader, val_loader_ttimes, device,
                                 adj, nn_ixs, edge_index, coords=coords)

                    except RuntimeError:
                        print('Out of Memory error for ', nh1, K)


                    # test

