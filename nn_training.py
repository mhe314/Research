
def model_trainer(path_dataset):
    '''train model
    
    Args:
        dataset_path: [String] folder to save dataset, please name it as "dataset";

    Returns:
        None, but save model to current_folder + "results/mode.pkl"
    '''
    # configeration
    config = Configer()

    dataset_train = MyDataset(path_dataset, 'train')
    dataset_test = MyDataset(path_dataset, 'test')
    print(f'[DATASET] The number of paired data (train): {len(dataset_train)}')
    print(f'[DATASET] The number of paired data (test): {len(dataset_test)}')
    print(f'[DATASET] Piano_shape: {dataset_train[0][0].shape}, guitar_shape: {dataset_train[0][1].shape}')

    # dataset
    train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=config.batch_size, shuffle=True)

    net = SimpleNet(config.p_length, config.g_length)
    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=int(config.epoch/4.), gamma=0.3)

    # Note that this part is about model_trainer
    loss_list = []
    for epoch_idx in range(config.epoch):
        # train
        for step, (piano_sound, guitar_sound, _) in enumerate(train_loader):
            inputs = piano_sound.to(device)
            targets = guitar_sound.to(device)
            inputs = inputs.reshape(inputs.shape[0], 4, -1)
            targets = targets.reshape(inputs.shape[0], 4, -1)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        # eval
        if epoch_idx % int(config.epoch/10.) == 0:
            net.eval()
            for step, (inputs, targets, _) in enumerate(train_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                inputs = inputs.reshape(inputs.shape[0], 4, -1)
                targets = targets.reshape(inputs.shape[0], 4, -1)
                outputs = net(inputs)
            loss = criterion(outputs, targets)
            print(f'epoch: {epoch_idx}/{config.epoch}, loss: {loss.item()}')

    # save model
    torch.save(net.state_dict(), path_dataset.replace('dataset', 'results')+'/model.pkl')

    # plot loss history
    fig = plt.figure()
    plt.plot(loss_list, 'k')
    plt.ylim([0, 0.02])
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.tight_layout()
    st.pyplot()
    #plt.savefig('results-MDS/MDS_loss.jpg', doi=300)

model_trainer(path_dataset)
