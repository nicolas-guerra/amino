import h5py
import torch
import numpy as np
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, norm, inputs_bool, device, which, mod, noise=0., samples=4096, data_path="data/Poisson_n70_L20_w2pi_FullDtN.h5"):
        print("Training with ", samples, " samples")
        #self.file_data = "data/Poisson_n70_L20_w2pi_FullDtN.h5"
        self.file_data = data_path
        self.mod = mod
        self.noise = noise
        if which == "training":
            self.length = samples

            self.start = 0
            self.which = which
            print("Using ", self.length, " Training Samples")
        elif which == "validation":
            self.length = 1024
            self.start = 4096
            self.which = which
        else:
            self.length = 2048
            self.start = 4096 + 1024
            self.which = which

        self.reader = h5py.File(self.file_data, 'r')
        self.mean_inp_dbc = torch.from_numpy(self.reader['mean_inp_dbc_fun'][:, :]).type(torch.float32)
        self.mean_inp_nbc = torch.from_numpy(self.reader['mean_inp_nbc_fun'][:, :]).type(torch.float32)
        self.mean_out = torch.from_numpy(self.reader['mean_out_fun'][:, :]).type(torch.float32)
        self.std_inp_dbc = torch.from_numpy(self.reader['std_inp_dbc_fun'][:, :]).type(torch.float32)
        self.std_inp_nbc = torch.from_numpy(self.reader['std_inp_nbc_fun'][:, :]).type(torch.float32)
        self.std_out = torch.from_numpy(self.reader['std_out_fun'][:, :]).type(torch.float32)
        self.min_data_dbc = self.reader['min_inp_dbc'][()]
        self.min_data_nbc = self.reader['min_inp_nbc'][()]
        self.max_data_dbc = self.reader['max_inp_dbc'][()]
        self.max_data_nbc = self.reader['max_inp_nbc'][()]
        self.min_model = self.reader['min_out'][()]
        self.max_model = self.reader['max_out'][()]
        if self.mod == "nio" or self.mod == "fcnn" or self.mod == "don":
            self.inp_dbc_dim_branch = 4
            self.inp_nbc_dim_branch = 4
            self.n_fun_samples = 20
        else:
            self.inp_dbc_dim_branch = 276
            self.inp_nbc_dim_branch = 272
            self.n_fun_samples = 20

        self.norm = norm
        self.inputs_bool = inputs_bool

        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        inputs_dbc = torch.from_numpy(self.reader[self.which]['sample_' + str(index)]["input_dbc"][:]).type(torch.float32)
        inputs_nbc = torch.from_numpy(self.reader[self.which]['sample_' + str(index)]["input_nbc"][:]).type(torch.float32)
        labels = torch.from_numpy(self.reader[self.which]['sample_' + str(index)]["output"][:]).type(torch.float32)

        inputs_dbc = inputs_dbc * (1 + self.noise * torch.randn_like(inputs_dbc))
        inputs_nbc = inputs_nbc * (1 + self.noise * torch.randn_like(inputs_nbc))

        if self.norm == "norm":
            #inputs_dbc = self.normalize(inputs_dbc, self.mean_inp_dbc, self.std_inp_dbc)
            inputs_nbc = self.normalize(inputs_nbc, self.mean_inp_nbc, self.std_inp_nbc)
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "norm-inp":
            inputs_dbc = self.normalize(inputs_dbc, self.mean_inp_dbc, self.std_inp_dbc)
            inputs_nbc = self.normalize(inputs_nbc, self.mean_inp_nbc, self.std_inp_nbc)
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "norm-out":
            inputs_dbc = 2 * (inputs_dbc - self.min_data_dbc) / (self.max_data_dbc - self.min_data_dbc) - 1.
            inputs_nbc = 2 * (inputs_nbc - self.min_data_nbc) / (self.max_data_nbc - self.min_data_nbc) - 1.
            labels = self.normalize(labels, self.mean_out, self.std_out)
        elif self.norm == "minmax":
            inputs_dbc = 2 * (inputs_dbc - self.min_data_dbc) / (self.max_data_dbc - self.min_data_dbc) - 1.
            inputs_nbc = 2 * (inputs_nbc - self.min_data_nbc) / (self.max_data_nbc - self.min_data_nbc) - 1.
            labels = 2 * (labels - self.min_model) / (self.max_model - self.min_model) - 1.
        elif self.norm == "none":
            inputs_dbc = inputs_dbc
            inputs_nbc = inputs_nbc
            labels = labels
        else:
            raise ValueError('Not a Valid Norm')

        if self.mod == "nio" or self.mod == "fcnn" or self.mod == "don":
            inputs_dbc = inputs_dbc.view(4, 69, 20)
            inputs_dbc_trim = inputs_dbc[:,:68,:] # Trim corners of DBC data to match dimensions of NBC
            inputs_nbc = inputs_nbc.view(4, 68, 20)

            inputs_concat = torch.cat((inputs_dbc_trim, inputs_nbc), dim=2) # Concat DBC and NBC data to one input
        else:
            inputs_dbc = inputs_dbc.view(1, 4, 69, 20).permute(3, 0, 1, 2)
            inputs_dbc_trim = inputs_dbc[:,:,:,:68]
            inputs_nbc = inputs_nbc.view(1, 4, 68, 20).permute(3, 0, 1, 2)

            inputs_concat = torch.cat((inputs_dbc_trim, inputs_nbc), dim=2) # Concat DBC and NBC data to one input

        return inputs_concat, labels
        # return {
        #     "inputs_dbc": inputs_dbc,
        #     "inputs_nbc": inputs_nbc,
        #     "labels": labels
        # }

    def normalize(self, tensor, mean, std):
        return (tensor - mean) / (std + 1e-16)

    def denormalize(self, tensor):
        if self.norm == "norm" or self.norm == "norm-out":
            return tensor * (self.std_out + 1e-16).to(self.device) + self.mean_out.to(self.device)
        elif self.norm == "none":
            return tensor
        else:
            return (self.max_model - self.min_model) * (tensor + torch.tensor(1., device=self.device)) / 2 + self.min_model

    def get_grid(self):

        # Reshape grid from (2,70,70) to (70,70,2)
        grid = np.zeros((70,70,2))
        grid[:,:,0] = self.reader['grid'][0,:, :]
        grid[:,:,1] = self.reader['grid'][1,:, :]

        # Convert to torch tensor
        grid = torch.from_numpy(grid).type(torch.float32)

        return grid.unsqueeze(0)
