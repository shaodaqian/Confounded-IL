import numpy as np
from models.DMLIV_models import MLP,MixtureGaussian_MLP
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader,random_split,ConcatDataset

from models.utils import SimpleDataset, mixture_of_gaussian_loss
from models.utils import device


def naive_learner(prev_states,actions,n_epochs,lr=0.0005,batch_size=256,dropout=0.01,weight_decay=1e-5,hiddens=None,verbose=True):
    if hiddens is None:
        hiddens=[8,16,32]

    if len(prev_states.shape)==1:
        prev_states=prev_states.reshape(-1,1)
    if len(actions.shape)==1:
        actions=actions.reshape(-1,1)

    train_data=SimpleDataset(torch.tensor(prev_states,dtype=torch.float32),torch.tensor(actions,dtype=torch.float32))
    train_loader=DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            )

    model = MLP(input_channel=prev_states.shape[-1],
                output_channel=actions.shape[-1],
                hiddens=hiddens,
                dropout=dropout).to(device)
    # print(model)
    # train the model
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)

    for epoch in range(n_epochs):
        for i, (prev_s,a) in enumerate(train_loader):
            y_pred = model(prev_s.to(device))
            loss = loss_fn(y_pred, a.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    model.eval()

    return model


def history_learner(
        prev_states,
        actions,
        n_epochs,
        policy_history,
        lr=0.0005,
        batch_size=256,
        dropout=0.01,
        weight_decay=1e-5,
        use_action=True,
        hiddens=None,
        verbose=True):

    if hiddens is None:
        hiddens=[8,16,32]

    if len(prev_states.shape)==1:
        prev_states=prev_states.reshape(-1,1)
    if len(actions.shape)==1:
        actions=actions.reshape(-1,1).astype(np.float32)
    state_dim = prev_states.shape[-1]
    action_dim = actions.shape[-1]

    history = []
    for i in range(policy_history):
        if i==0:
            states_temp=prev_states
        else:
            states_temp=np.concatenate((np.repeat(prev_states[0:1,:],i,axis=0),prev_states[:-i,:]),axis=0)
        history.append(states_temp)
        if use_action:
            actions_temp=np.concatenate((np.zeros((i+1,action_dim)),actions[:-(i+1),:]),axis=0)
            history.append(actions_temp)

    X=np.concatenate(history,axis=1,dtype=np.float32)

    X[:,0]=(X[:,0]-X[:,state_dim]-0.0)*20
    X=X[:,:state_dim]


    train_data=SimpleDataset(torch.tensor(X),torch.tensor(actions))
    train_loader=DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            )

    if use_action:
        input_channel = policy_history*(state_dim+action_dim)
    else:
        input_channel = policy_history*state_dim
    model = MLP(input_channel=state_dim,hiddens=hiddens,output_channel=action_dim, dropout=dropout).to(device)
    # model=torch.compile(model)
    # train the model
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay=weight_decay)

    for epoch in range(n_epochs):
        for i, (prev_s,a) in enumerate(train_loader):
            y_pred = model(prev_s.to(device))
            loss = loss_fn(y_pred, a.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if verbose:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    model.eval()

    return model

def IV_learner(
        prev_states,
        actions,
        n_epochs,
        ue_k,
        lr=0.0005,
        batch_size=256,
        dropout=0.01,
        weight_decay=1e-5,
        hiddens=None,
        n_components=5,
        n_samples=1,
        verbose=True):

    if hiddens is None:
        hiddens=[8,16,32]

    if len(prev_states.shape)==1:
        prev_states=prev_states.reshape(-1,1)
    if len(actions.shape)==1:
        actions=actions.reshape(-1,1).astype(np.float32)

    inp_channel = prev_states.shape[-1]
    out_channel = actions.shape[-1]

    prev_k_states=np.concatenate((np.repeat(prev_states[0:1,:], ue_k,axis=0), prev_states[:-ue_k,:]), axis=0)

    p_k_s=torch.tensor(prev_k_states,dtype=torch.float32)
    p_s=torch.tensor(prev_states,dtype=torch.float32)
    y=torch.tensor(actions,dtype=torch.float32)

    train_data=SimpleDataset(p_k_s,p_s,y)
    train_loader=DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            )

    s_model = MixtureGaussian_MLP(input_channel=inp_channel,output_channel=inp_channel,hiddens=hiddens,dropout=dropout,n_components=n_components).to(device)
    # a_model = MLP(input_channel=1, hiddens=[8,16,32],dropout=0.0).to(device)
    policy_model = MLP(input_channel=inp_channel,output_channel=out_channel,hiddens=hiddens,dropout=dropout).to(device)

    # print(model)
    # train the model
    mse_loss = nn.MSELoss()
    mixture_loss = mixture_of_gaussian_loss
    optimizer_s = optim.AdamW(s_model.parameters(), lr=lr,weight_decay=weight_decay)
    # optimizer_a = optim.AdamW(a_model.parameters(), lr=lr)
    optimizer_policy = optim.AdamW(policy_model.parameters(), lr=lr,weight_decay=weight_decay)


    for epoch in range(n_epochs):
        total_loss=0
        for i, (prev_k_s,prev_s,a) in enumerate(train_loader):
            s_pred = s_model(prev_k_s.to(device))
            target=prev_s.to(device)
            # target[:,0]=0.0
            loss = mixture_loss(s_pred, target)
            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()
        if verbose:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    s_model.eval()


    for epoch in range(n_epochs):
        for i, (prev_k_s,prev_s,a) in enumerate(train_loader):
            s_sample=torch.zeros_like(prev_k_s,device=device)
            for _ in range(n_samples):
                s_sample+=s_model.sample(prev_k_s.to(device))
            s_sample/=n_samples
            # a_sample=a_model(prev_k_s)
            a_sample=a
            y_pred = policy_model(s_sample)
            loss = mse_loss(y_pred, a_sample.to(device))

            optimizer_policy.zero_grad()
            loss.backward()
            optimizer_policy.step()
        if verbose:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')

    policy_model.eval()

    return policy_model

def IV_history_learner(
        prev_states,
        actions,
        n_epochs,
        ue_k,
        sample_k,
        policy_history,
        lr=0.0005,
        batch_size=256,
        dropout=0.01,
        s_drop_out=None,
        weight_decay=1e-5,
        s_weight_decay=None,
        n_components=5,
        n_samples=1,
        use_action=True,
        hiddens=None,
        verbose=True):

    assert sample_k<=policy_history
    assert sample_k<=ue_k

    if hiddens is None:
        hiddens=[8,16,32]
    if s_drop_out is None:
        s_drop_out=dropout
    if s_weight_decay is None:
        s_weight_decay=weight_decay

    if len(prev_states.shape)==1:
        prev_states=prev_states.reshape(-1,1)
    if len(actions.shape)==1:
        actions=actions.reshape(-1,1).astype(np.float32)

    state_dim = prev_states.shape[-1]
    action_dim = actions.shape[-1]

    history = []
    for i in range(policy_history):
        if i==0:
            states_temp=prev_states
        else:
            states_temp=np.concatenate((np.repeat(prev_states[0:1,:],i,axis=0),prev_states[:-i,:]),axis=0)
        history.append(states_temp)
        if use_action:
            actions_temp=np.concatenate((np.zeros((i+1,action_dim)),actions[:-(i+1),:]),axis=0)
            history.append(actions_temp)

    history=np.concatenate(history,axis=1,dtype=np.float32)



    prev_k_states=np.concatenate((np.repeat(prev_states[0:1,:], ue_k,axis=0), prev_states[:-ue_k,:]), axis=0)
    p_k_s=torch.tensor(prev_k_states,dtype=torch.float32)

    y=torch.tensor(actions,dtype=torch.float32)
    history=torch.tensor(history,dtype=torch.float32)

    train_data=SimpleDataset(p_k_s,history,y)
    train_loader=DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            pin_memory=True,
                            )

    if use_action:
        inp_channel = policy_history*(state_dim+action_dim)
        sample_channel=sample_k*state_dim+(sample_k-1)*action_dim
    else:
        inp_channel = policy_history*state_dim
        sample_channel=sample_k*state_dim

    state_channel=p_k_s.shape[-1]

    # print('input channel:',inp_channel,'sample channel:',sample_channel,'state channel:',state_channel,'action channel:',action_dim)

    policy_model = MLP(input_channel=state_dim,hiddens=hiddens,output_channel=action_dim, dropout=dropout).to(device)
    optimizer_policy = optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=weight_decay)

    s_model = MixtureGaussian_MLP(input_channel=state_channel,output_channel=sample_channel,hiddens=hiddens,dropout=s_drop_out,n_components=n_components).to(device)
    optimizer_s = optim.AdamW(s_model.parameters(), lr=lr, weight_decay=s_weight_decay)

    mse_loss = nn.MSELoss()
    mixture_loss = mixture_of_gaussian_loss

    # s_model=torch.compile(s_model,fullgraph=False)
    for epoch in range(n_epochs):
        for i, (prev_k_s,history,a) in enumerate(train_loader):
            s_pred = s_model(prev_k_s.to(device))
            target=history[:,:sample_channel].to(device)
            target[:,0]=0.0
            loss = mixture_loss(s_pred, target)

            optimizer_s.zero_grad()
            loss.backward()
            optimizer_s.step()
        if verbose:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    s_model.eval()
    # print(sample_channel,'number of features for GMM model to predict')


    for epoch in range(n_epochs):
        for i, (prev_k_s,history,a) in enumerate(train_loader):
            s_sample=torch.zeros_like(history[:,:sample_channel],device=device)
            for _ in range(n_samples):
                s_sample+=s_model.sample(prev_k_s.to(device))
            s_sample/=n_samples
            # a_sample=a_model(prev_k_s.to(device))
            a_sample=a
            IV_history=torch.cat([s_sample,history[:,sample_channel:].to(device)],dim=1)
            IV_history[:, 0] = (IV_history[:, ue_k*state_dim] - IV_history[:, (ue_k+1)*state_dim] - 0.0) * 20
            IV_history = IV_history[:, :state_dim]

            y_pred = policy_model(IV_history)
            loss = mse_loss(y_pred, a_sample.to(device))

            optimizer_policy.zero_grad()
            loss.backward()
            optimizer_policy.step()
        if verbose:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    policy_model.eval()

    return policy_model

def DMLIV_history_learner(
        prev_states,
        actions,
        n_epochs,
        ue_k,
        sample_k,
        policy_history,
        lr=0.0005,
        batch_size=256,
        dropout=0.01,
        weight_decay=1e-5,
        n_components=5,
        n_samples=1,
        k_fold=6,
        use_action=True,
        hiddens=None,
        verbose=True):

    assert sample_k<=policy_history
    assert sample_k<=ue_k


    if hiddens is None:
        hiddens=[8,16,32]

    if len(prev_states.shape)==1:
        prev_states=prev_states.reshape(-1,1)
    if len(actions.shape)==1:
        actions=actions.reshape(-1,1).astype(np.float32)

    state_dim = prev_states.shape[-1]
    action_dim = actions.shape[-1]

    history = []
    for i in range(policy_history):
        if i==0:
            states_temp=prev_states
        else:
            states_temp=np.concatenate((np.repeat(prev_states[0:1,:],i,axis=0),prev_states[:-i,:]),axis=0)
        history.append(states_temp)
        if use_action:
            actions_temp=np.concatenate((np.zeros((i+1,action_dim)),actions[:-(i+1),:]),axis=0)
            history.append(actions_temp)

    history=np.concatenate(history,axis=1,dtype=np.float32)


    prev_k_history=[]
    for i in range(1):
        states_temp=np.concatenate((np.repeat(prev_states[0:1,:], ue_k+i,axis=0), prev_states[:-(ue_k+i),:]), axis=0)
        prev_k_history.append(states_temp)
        if use_action:
            actions_temp=np.concatenate((np.zeros((ue_k+i+1,action_dim)),actions[:-(ue_k+i+1),:]),axis=0)
            prev_k_history.append(actions_temp)

    prev_k_history=np.concatenate(prev_k_history,axis=1,dtype=np.float32)

    prev_k_states=np.concatenate((np.repeat(prev_states[0:1,:], ue_k,axis=0), prev_states[:-ue_k,:]), axis=0)
    p_k_s=torch.tensor(prev_k_states,dtype=torch.float32)

    y=torch.tensor(actions,dtype=torch.float32)
    history=torch.tensor(history,dtype=torch.float32)

    train_data=SimpleDataset(p_k_s,history,y)


    if use_action:
        inp_channel = policy_history*(state_dim+action_dim)
        sample_channel=sample_k*state_dim+(sample_k-1)*action_dim
    else:
        inp_channel = policy_history*state_dim
        sample_channel=sample_k*state_dim

    state_channel=p_k_s.shape[-1]

    datasets = random_split(train_data, [1 / k_fold] * k_fold)
    state_models = []
    y_models = []
    for fold in range(k_fold):
        train_data_fold = []
        for d in range(k_fold):
            if d != fold:
                train_data_fold.append(datasets[d])

        train_data_fold = ConcatDataset(train_data_fold)
        train_loader = DataLoader(train_data_fold,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  pin_memory=True,
                                  )
        # print(len(train_data))

        s_model = MixtureGaussian_MLP(input_channel=state_channel, output_channel=sample_channel, hiddens=hiddens,
                                      dropout=dropout, n_components=n_components).to(device)
        # a_model = MLP(input_channel=state_channel,output_channel=action_dim, hiddens=hiddens,dropout=dropout).to(device)

        # train the model
        mse_loss = nn.MSELoss()
        mixture_loss = mixture_of_gaussian_loss
        optimizer_s = optim.AdamW(s_model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer_a = optim.AdamW(a_model.parameters(), lr=lr, weight_decay=weight_decay)

        # s_model=torch.compile(s_model,fullgraph=False)
        for epoch in range(n_epochs):
            for i, (prev_k_s, history, a) in enumerate(train_loader):
                s_pred = s_model(prev_k_s.to(device))
                target = history[:, :sample_channel].to(device)
                # target[:,0]=0.0
                loss = mixture_loss(s_pred, target)

                optimizer_s.zero_grad()
                loss.backward()
                optimizer_s.step()
            if verbose:
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
        s_model.eval()
        # print(sample_channel,'number of features for GMM model to predict')

        state_models.append(s_model)
        # y_models.append(a_model)



    policy_model = MLP(input_channel=state_dim, hiddens=hiddens, output_channel=action_dim, dropout=dropout).to(device)
    optimizer_policy = optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=weight_decay)


    train_loaders=[]
    for data in datasets:
        train_loaders.append(torch.utils.data.DataLoader(
        data, batch_size=batch_size,
        shuffle=True,  # 'True' to check training progress with validation function.
        pin_memory=True))

    for epoch in range(n_epochs):
        for fold,train_loader in enumerate(train_loaders):
            for i, (prev_k_s,history,a) in enumerate(train_loader):
                s_sample=torch.zeros_like(history[:,:sample_channel],device=device)
                for _ in range(n_samples):
                    s_sample+=state_models[fold].sample(prev_k_s.to(device))
                s_sample/=n_samples
                a_sample=a
                IV_history=torch.cat([s_sample,history[:,sample_channel:].to(device)],dim=1)
                # IV_history=s_sample
                IV_history[:, 0] = (IV_history[:, ue_k*state_dim] - IV_history[:, (ue_k+1)*state_dim] - 0.0) * 20
                IV_history = IV_history[:, :state_dim]

                y_pred = policy_model(IV_history)
                loss = mse_loss(y_pred, a_sample.to(device))

                optimizer_policy.zero_grad()
                loss.backward()
                optimizer_policy.step()
        if verbose:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    policy_model.eval()

    return policy_model
