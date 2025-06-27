import ase
import os
import pandas as pd
import numpy as np
import torch
import h5py
import lightning as L

#loader to import dicts:
from cace.data.atomic_data import AtomicData
from cace.tools.torch_geometric import Dataset, DataLoader

def from_h5key(h5key,h5fn,cutoff=None):
    with h5py.File(h5fn, "r") as f:
        data = f[h5key]
        hartree_to_ev = 27.2114
        bohr_to_angstrom = 0.529177

        #Make atoms object
        els = np.array(data["atomic_numbers"])
        pos = np.array(data["positions"]) * bohr_to_angstrom
        pos = pos - pos.mean(axis=0) #center
        atoms = ase.Atoms(numbers=els,positions=pos)
        ad = AtomicData.from_atoms(atoms,cutoff=cutoff) #makes graph structure

        ad.energy = torch.Tensor(np.array(data["energy"])) * hartree_to_ev
        ad.force = torch.Tensor(np.array(data["force"])) * hartree_to_ev/bohr_to_angstrom
        ad.dipole = torch.Tensor(np.array(data["dipole"]))[None,:] * bohr_to_angstrom
        ad.quadrupole = torch.Tensor(np.array(data["quadrupole"]))[None,...] * bohr_to_angstrom**2
        ad.mbi_charges = torch.Tensor(np.array(data["mbis_charges"])).squeeze()
        return ad

class SpiceDataset(Dataset):
    def __init__(self,root="data/aodata.h5",cutoff=4.0,
                transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.cutoff = cutoff

    def len(self):
        with h5py.File(self.root, "r") as f:
            return len(f.keys())
    
    def get(self, idx):
        return from_h5key(f"c{idx}",h5fn=self.root,cutoff=self.cutoff)

class SpiceInMemoryDataset(Dataset):
    def __init__(self,root="data/spice-dipep.h5",cutoff=4.0,
                transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.cutoff = cutoff
        self.prepare_data()

    def get_h5(self,i):
        return from_h5key(f"c{i}",h5fn=self.root,cutoff=self.cutoff)
    
    def prepare_data(self):
        with h5py.File(self.root, "r") as f:
            data_len = len(f.keys())
        self.dataset = [self.get_h5(i) for i in range(data_len)]

    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        return self.dataset[idx]

class SpiceData(L.LightningDataModule):
    def __init__(self, root="data/spice-dipep.h5", cutoff=4.0, in_memory=False, drop_last=True,
                 batch_size=32, valid_p=0.1, test_p=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.valid_p = valid_p
        self.test_p = test_p
        self.cutoff = cutoff
        self.drop_last = drop_last
        self.in_memory = in_memory
        try:
            self.num_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        except:
            self.num_cpus = os.cpu_count()
        self.prepare_data()
    
    def prepare_data(self):
        if not self.in_memory:
            dataset = SpiceDataset(self.root,cutoff=self.cutoff)
        else:
            dataset = SpiceInMemoryDataset(self.root,cutoff=self.cutoff)
        torch.manual_seed(12345)
        dataset = dataset.shuffle()
        cut1 = int(len(dataset)*(1-self.valid_p-self.test_p))
        cut2 = int(len(dataset)*(1-self.test_p))
        self.train = dataset[:cut1]
        self.val = dataset[cut1:cut2]
        self.test = dataset[cut2:]

    def train_dataloader(self):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, drop_last=self.drop_last,
                                  shuffle=True, num_workers = self.num_cpus)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val, batch_size=self.batch_size, drop_last=False,
                                shuffle=False, num_workers = self.num_cpus)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test, batch_size=self.batch_size, drop_last=False,
                                shuffle=False, num_workers = self.num_cpus)
        return test_loader
