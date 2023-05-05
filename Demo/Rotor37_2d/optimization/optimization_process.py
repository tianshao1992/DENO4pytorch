def opt_grad(netmodel, device, target, optimizer, scheduler, initial_input = None):
    """
    Args:
        netmodel: Network
        lossfunc: Loss function
        optimizer: optimizer
        scheduler: scheduler
    """
    # Initial value
    target_cur = 0


    # start optimization
    for ii in range(1000):
        input = input.to(device)
        output = output.to(device)
        pred = netmodel(input)

        optimal = target(pred) # 将

        # grid_size = 64
        # weighted_lines = 3
        # weighted_cof = 1.5
        # temp1 = np.ones([weighted_lines, grid_size])*weighted_cof
        # temp2 = np.ones([grid_size-weighted_lines*2, grid_size])
        # weighted_mat = np.concatenate((temp1,temp2,temp1),axis=0).reshape(output[0].shape)
        # weighted_mat = np.tile(weighted_mat[None,:],(output.shape[0],1))

        optimizer.zero_grad()
        optimal.backward() # 自动微分
        optimizer.step()


    scheduler.step()
    return optimal