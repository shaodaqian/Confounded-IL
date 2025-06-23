from __future__ import annotations
import torch
from torch import nn
import torch.optim as optim
import logging
import numpy as np

from models.DFIV_model import DFIVModel
from models.utils import linear_reg_loss, device, SimpleDataset,mixture_of_gaussian_loss

from models.DMLIV_models import MLP,MixtureGaussian_MLP
from torch.utils.data import Dataset, DataLoader,random_split,ConcatDataset


logger = logging.getLogger()


class DFIVTrainer(object):
    def __init__(self, lam1=0.1,lam2=0.1,stage1_iter=20,stage2_iter=1,n_epoch=100,split_ratio=0.5,
                 treatment_weight_decay=0.0,instrumental_weight_decay=0.00001):
        # configure training params
        self.lam1 = lam1
        self.lam2 = lam2
        self.stage1_iter = stage1_iter
        self.stage2_iter = stage2_iter
        self.n_epoch = n_epoch
        self.split_ratio = split_ratio
        self.treatment_weight_decay = treatment_weight_decay
        self.instrumental_weight_decay = instrumental_weight_decay
        self.add_stage1_intercept = True
        self.add_stage2_intercept = True

    def train(self, prev_states,actions,ue_k,policy_history,lr=0.0005,
              dropout=0.01,hiddens=None,out_d=16,verbose=True,use_action=True):
        """

        Parameters
        ----------
        rand_seed: int
            random seed
        verbose : int
            Determine the level of logging
        Returns
        -------
        oos_result : float
            The performance of model evaluated by oos
        """

        if hiddens is None:
            hiddens = [8, 16, 32]


        if len(prev_states.shape) == 1:
            prev_states = prev_states.reshape(-1, 1)
        if len(actions.shape) == 1:
            actions = actions.reshape(-1, 1).astype(np.float32)

        state_dim = prev_states.shape[-1]
        action_dim = actions.shape[-1]

        if use_action:
            inp_channel = policy_history * (state_dim + action_dim)
        else:
            inp_channel = policy_history * state_dim

        batch_size=prev_states.shape[0]
        print(state_dim,action_dim,inp_channel,batch_size)
        # batch_size=1000

        history = []
        for i in range(policy_history):
            if i == 0:
                states_temp = prev_states
            else:
                states_temp = np.concatenate((np.repeat(prev_states[0:1, :], i, axis=0), prev_states[:-i, :]), axis=0)
            history.append(states_temp)
            if use_action:
                actions_temp = np.concatenate((np.zeros((i + 1, action_dim)), actions[:-(i + 1), :]), axis=0)
                history.append(actions_temp)

        history = np.concatenate(history, axis=1, dtype=np.float32)

        prev_k_states = np.concatenate((np.repeat(prev_states[0:1, :], ue_k, axis=0), prev_states[:-ue_k, :]), axis=0)
        p_k_s = torch.tensor(prev_k_states, dtype=torch.float32)

        y = torch.tensor(actions, dtype=torch.float32)
        history = torch.tensor(history, dtype=torch.float32)

        train_data = SimpleDataset(p_k_s, history, y)
        train_1st_t, train_2nd_t = random_split(train_data,[0.5,0.5])
        # train_both = torch.utils.data.TensorDataset(train_1st_t, train_2nd_t)

        self.lam1 *= len(train_1st_t)
        self.lam2 *= len(train_2nd_t)
        print(self.lam1,'lam1')

        train_1st_t = DataLoader(train_1st_t,batch_size=batch_size, shuffle=True, pin_memory=True)
        train_2nd_t = DataLoader(train_2nd_t, batch_size=batch_size, shuffle=True,pin_memory=True)

        self.treatment_net = MLP(input_channel=inp_channel, output_channel=action_dim*out_d, hiddens=hiddens, dropout=dropout).to(device)
        self.instrumental_net = MLP(input_channel=state_dim, output_channel=action_dim*out_d*2, hiddens=hiddens,dropout=dropout).to(device)

        self.treatment_opt = torch.optim.AdamW(self.treatment_net.parameters(),lr=lr,weight_decay=self.treatment_weight_decay)
        self.instrumental_opt = torch.optim.AdamW(self.instrumental_net.parameters(),lr=lr,weight_decay=self.instrumental_weight_decay)

        for t in range(self.n_epoch):
            self.instrumental_net.train(False)

            train_2nd_t_iter = iter(train_2nd_t)

            for _, (prev_k_s1, history1, a1) in enumerate(train_1st_t):
                (prev_k_s2, history2, a2) = next(train_2nd_t_iter)

                self.stage1_update(prev_k_s1,history1,a1, False)
                self.stage2_update(prev_k_s1,history1,a1,prev_k_s2,history2,a2,False)
            if verbose:
                print(f"Epoch {t} ended")

        mdl = DFIVModel(self.treatment_net, self.instrumental_net,self.add_stage1_intercept, self.add_stage2_intercept,None)
        mdl.fit_t(train_1st_t, train_2nd_t, self.lam1, self.lam2)

        return mdl

    def stage1_update(self, prev_k_s1,history1,a1, verbose):
        self.treatment_net.train(False)
        self.instrumental_net.train(True)

        treatment_feature = self.treatment_net(history1.to(device)).detach()
        for _ in range(self.stage1_iter):
            self.instrumental_opt.zero_grad()
            instrumental_feature = self.instrumental_net(prev_k_s1.to(device))
            feature = DFIVModel.augment_stage1_feature(instrumental_feature,
                                                       self.add_stage1_intercept)
            loss = linear_reg_loss(treatment_feature, feature, self.lam1)
            loss.backward()
            if verbose:
                print(f"stage1 learning: {loss.item()}")
            self.instrumental_opt.step()

    def stage2_update(self,prev_k_s1,history1,a1,prev_k_s2,history2,a2,verbose):
        self.treatment_net.train(True)
        self.instrumental_net.train(False)


        # have instrumental features
        instrumental_1st_feature = self.instrumental_net(prev_k_s1.to(device)).detach()
        instrumental_2nd_feature = self.instrumental_net(prev_k_s2.to(device)).detach()

        # have covariate features
        covariate_2nd_feature = None

        for _ in range(self.stage2_iter):
            self.treatment_opt.zero_grad()
            treatment_1st_feature = self.treatment_net(history1.to(device))
            res = DFIVModel.fit_2sls(treatment_1st_feature,
                                     instrumental_1st_feature,
                                     instrumental_2nd_feature,
                                     covariate_2nd_feature,
                                     a2.to(device),
                                     self.lam1, self.lam2,
                                     self.add_stage1_intercept,
                                     self.add_stage2_intercept)
            loss = res["stage2_loss"]
            loss.backward()
            if verbose:
                print(f"stage2 learning: {loss.item()}")
            self.treatment_opt.step()