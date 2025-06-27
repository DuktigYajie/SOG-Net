import torch
import torch.nn as nn
from itertools import product
from typing import Dict
import pytorch_finufft
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SOGPotential(nn.Module):
    def __init__(self,
                 N_dl=1,  # Fourier modes
                 bandwidth_num = 12,
                 external_field = None, # external field
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z
                 charge_neutral_lambda: float = None,
                 remove_self_interaction=False,
                 feature_key: str = 'q',
                 output_key: str = 'SOG_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 Periodic: bool = False,
                 ):
        super().__init__()
        self.N_dl = N_dl
        self.bandwidth_num = bandwidth_num
        # Create bandwidth 
        # self.bandwidth = torch.linspace(-5, 1.2, self.bandwidth_num)  # Exponential decay
        # Parameters to learn during training
        self.shift_1 = torch.nn.Parameter(torch.linspace(-0.5, 1.0, self.bandwidth_num, dtype=torch.float32))
        self.amplitude_1 = torch.nn.Parameter(torch.ones(self.bandwidth_num, dtype=torch.float32))

        # self.shift_1 = torch.nn.Parameter(torch.linspace(-3.0, 2.0, self.bandwidth_num, dtype=torch.float32))
        # self.amplitude_1 = torch.nn.Parameter(torch.tensor([-7.0450, 11.4645, -4.9724, 0.4311, 0.1973, -0.1282, 0.4223, 1.3309, 3.2130, 8.1743, 19.3299, 55.2736],dtype=torch.float32))# dimer-CC
        # self.amplitude_1 = torch.nn.Parameter(torch.tensor([0.2750, 0.1375, 0.0688, 0.0344, 0.0172, 0.0086, 0.0043, 0.0021, 0.0011, 0.0005, 0.0003, 0.0001], dtype=torch.float32))
        # self.shift_1 = torch.nn.Parameter(torch.tensor([2.8, 5.7, 11.4, 22.7, 45.5, 91.0, 182.0, 364.0, 728.0, 1456.0, 2912.0, 5823.9],dtype=torch.float32))
        
        # self.amplitude_1 = torch.tensor([0.2750, 0.1375, 0.0688, 0.0344, 0.0172, 0.0086, 0.0043, 0.0021, 0.0011, 0.0005, 0.0003, 0.0001], dtype=torch.float32).to(device)
        # self.shift_1 = torch.tensor([2.8, 5.7, 11.4, 22.7, 45.5, 91.0, 182.0, 364.0, 728.0, 1456.0, 2912.0, 5823.9],dtype=torch.float32).to(device)
 
        self.Periodic = Periodic
        #print("shift_begin:",self.shift_1)

        self.norm_factor = torch.tensor(1.0)# self.norm_factor = torch.nn.Parameter(torch.tensor(1.0))
        self.ene_factor = torch.nn.Parameter(torch.tensor(0.0))#self.ene_factor = torch.tensor(0.0) # self.ene_factor = torch.nn.Parameter(torch.tensor(0.0))


        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.model_outputs = [output_key]
        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.compute_field = compute_field
        if self.compute_field:
            self.model_outputs.append(feature_key+'_field')
        self.charge_neutral_lambda = charge_neutral_lambda

        self.dl = self.N_dl
        self.sigma = 1.0
        self.exponent = 1 ##6
        self.sigma_sq_half = self.sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        #self.norm_factor = 1.0 
        self.k_sq_max = (self.twopi / self.dl) ** 2

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        # this is just for compatibility with the previous version
        if hasattr(self, 'exponent') == False:
            self.exponent = 1
        if hasattr(self, 'compute_field') == False:
            self.compute_field = False
        
        # box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        box = data['cell'].view(-1, 3, 3)
        #print("cell:",box)

        r = data['positions'] # (total_atom_number_of_all_configurations_in_batch, 3)
        #print(r.shape)

        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        #print(q.shape) # (total_atom_number_of_all_configurations_in_batch, number_of_q_layers)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices. Batch_now saves the corresponding configuration index [0 0 ... 0 1 ... 1 2 ... 2]. Unique is used to get the total number of configurations in the batch
        #print(batch_now)
        #print(batch_now.shape, unique_batches.shape)
        results = []
        field_results = []
        for i in unique_batches:
            mask = (batch_now == i)  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now, box_now = r[mask], q[mask], box[i] # Extract the atomic information for each configuration.
            #print("Mask:",mask,i,r_raw_now.shape, q_now.shape, box_now.shape,r.shape,q.shape,box.shape)
            #print("box_now:", box_now)
            box_diag = box[i].diagonal(dim1=-2, dim2=-1)
            if not hasattr(self, 'Periodic'):
                self.Periodic = False
            if self.Periodic:
                pot, field = self.compute_potential_SOG(r_raw_now, q_now, box_now, self.compute_field)
                # pot, field = self.compute_potential_SOG_pure(r_raw_now, q_now, box_now, self.compute_field) #For pure SR NaCl and Molten NaCl
            else:
                pot, field = self.compute_potential_Gaussian_realspace(r_raw_now, q_now, self.compute_field)
                #pot = self.compute_potential_SOG(r_raw_now, q_now, box_now, self.compute_field)
            # pot, field = self.compute_potential_Gaussian_realspace(r_raw_now, q_now, self.compute_field)
            if self.exponent == 1 and hasattr(self, 'external_field') and self.external_field is not None:
                # if self.external_field_direction is an integer, then external_field_direction is the direction index
                if isinstance(self.external_field_direction, int):
                    direction_index_now = self.external_field_direction
                    # if self.external_field_direction is a string, then it is the key to the external field
                else:
                    try:
                        direction_index_now = int(data[self.external_field_direction][i])
                    except:
                        raise ValueError("external_field_direction must be an integer or a key to the external field")
                if isinstance(self.external_field, float):
                    external_field_now = self.external_field
                else:
                    try:
                        external_field_now = data[self.external_field][i]
                    except:
                        raise ValueError("external_field must be a float or a key to the external field")
                box_now = box_now.diagonal(dim1=-2, dim2=-1)
                pot_ext = self.add_external_field(r_raw_now, q_now, box_now, direction_index_now, external_field_now)
            else:
                pot_ext = 0.0

            if hasattr(self, 'charge_neutral_lambda') and self.charge_neutral_lambda is not None:
                q_mean = torch.mean(q[mask])
                pot_neutral = self.charge_neutral_lambda * (q_mean)**2.
                #print(pot_neutral, pot)
            else:
                pot_neutral = 0.0
            
            #print("pot", pot)
            #print("pot_ext",pot_ext)
            #print("pot_neutral",pot_neutral)
            #print(pot + pot_ext + pot_neutral)
            # results.append(pot + pot_ext + pot_neutral)
            # print("SOG:",(pot + pot_ext + pot_neutral).shape)
            if not hasattr(self, 'ene_factor'):
                self.ene_factor = 0.0
            results.append(pot + self.ene_factor)
        #print(results[0].shape,results[1].shape,results[2].shape, pot.shape)
        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            field_results.append(field)
            data[self.feature_key+'_field'] = torch.cat(field_results, dim=0)
        return data

    def compute_potential_SOG(self, r_raw, q, box, compute_field=False):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device
        
        # print(self.shift_1)
        
        cell_inv = torch.linalg.inv(box)
        G = 2 * torch.pi * cell_inv.T

        norms = torch.norm(box, dim=1)
        Lx = box[0,0]
        Ly = box[1,1]
        Lz = box[2,2]
        N_dl_x = (torch.ceil(Lx / self.N_dl)).int()
        N_dl_y = (torch.ceil(Ly / self.N_dl)).int()
        N_dl_z = (torch.ceil(Lz / self.N_dl)).int()
        #print(N_dl_x, N_dl_y,N_dl_z)
        n1 = torch.arange(-N_dl_x, N_dl_x + 1, device=device)
        n2 = torch.arange(-N_dl_y, N_dl_y + 1, device=device)
        n3 = torch.arange(-N_dl_z, N_dl_z + 1, device=device)
        print(n1,n2,n3)

        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        kvec = (nvec.float() @ G).to(device)
        
        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)
        
        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        exp_ikr = torch.exp(1j * k_dot_r)
        q_expanded = q.unsqueeze(-1)  # 将 q 的形状从 (16, 3) 扩展为 (16, 3, 1)
        # 扩展 exp_ikr 的维度
        exp_ikr_expanded = exp_ikr.unsqueeze(1)  # 将 exp_ikr 的形状从 (16, 4484342) 扩展为 (16, 1, 4484342)
        # 逐元素乘法
        product = q_expanded * exp_ikr_expanded  # 形状为 (16, 3, 4484342)
        # 计算结构因子 S(k)
        S_k = torch.sum(product, dim=0)  # 形状为 (3, 4484342)
        
        min_term = -1 / torch.exp(-2 * self.shift_1)  # 计算指数衰减
        min_term = min_term.view(1, 1, 1, -1)  # 扩展维度
        kfac = self.amplitude_1.view(1, 1, 1, -1) * torch.exp(k_sq.unsqueeze(-1) * min_term)  # 计算 SOG 频谱响应
        # print("multiplier:",multiplier.shape,multiplier)
        kfac = kfac.sum(dim=-1)  # 在 SOG 频谱维度求和
        #kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        
        volume = torch.det(box)
        pot = (factors * kfac * torch.abs(S_k)**2).sum() / volume
        
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)
        
        # if compute_field:
        #     sk_field = 2 * kfac * torch.conj(S_k)
        #     q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume
        # if self.remove_self_interaction and self.exponent == 1:
        #     pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)
        #     q_field -= q * (2 / (self.sigma * (2 * torch.pi)**1.5))
        if compute_field:
            sk_field = 2 * kfac * torch.conj(S_k)
            q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume
        if self.remove_self_interaction and self.exponent == 1:
            #print("Here is self-interaction")
            #pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)  
            diag_sum = kfac.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2 * volume)
            pot -= torch.sum(q**2)*diag_sum
            q_field -= q * (2 *diag_sum)
        return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor

    def compute_potential_SOG_pure(self, r_raw, q, box, compute_field=False):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device
        
        #print(self.shift_1)
        
        cell_inv = torch.linalg.inv(box)
        G = 2 * torch.pi * cell_inv.T
        
        #print("Here",box,cell_inv,G)

        # norms = torch.norm(box, dim=1)
        # Lx = box[0,0]
        # Ly = box[1,1]
        # Lz = box[2,2]
        # N_dl_x = (torch.ceil(Lx / self.N_dl)).int()
        # N_dl_y = (torch.ceil(Ly / self.N_dl)).int()
        # N_dl_z = (torch.ceil(Lz / self.N_dl)).int()
        # #print(N_dl_x, N_dl_y,N_dl_z)
        # n1 = torch.arange(-N_dl_x, N_dl_x + 1, device=device)
        # n2 = torch.arange(-N_dl_y, N_dl_y + 1, device=device)
        # n3 = torch.arange(-N_dl_z, N_dl_z + 1, device=device)
        
        norms = torch.norm(box, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)
        # print(Nk,n1,n2,n3)

        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        kvec = (nvec.float() @ G).to(device)
        
        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)

        mask = (k_sq > 0)
        
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)
        
        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        exp_ikr = torch.exp(1j * k_dot_r)
        q_expanded = q.unsqueeze(-1)  # 将 q 的形状从 (16, 3) 扩展为 (16, 3, 1)
        # 扩展 exp_ikr 的维度
        exp_ikr_expanded = exp_ikr.unsqueeze(1)  # 将 exp_ikr 的形状从 (16, 4484342) 扩展为 (16, 1, 4484342)
        # 逐元素乘法
        product = q_expanded * exp_ikr_expanded  # 形状为 (16, 3, 4484342)
        # 计算结构因子 S(k)
        S_k = torch.sum(product, dim=0)  # 形状为 (3, 4484342)
        
        min_term = -1 / torch.exp(-2 * self.shift_1)  # 计算指数衰减
        min_term = min_term.view(1, 1, 1, -1)  # 扩展维度
        kfac = self.amplitude_1.view(1, 1, 1, -1) * torch.exp(k_sq.unsqueeze(-1) * min_term)  # 计算 SOG 频谱响应
        # print("multiplier:",multiplier.shape,multiplier)
        kfac = kfac.sum(dim=-1)  # 在 SOG 频谱维度求和
        #kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq

        volume = torch.det(box)
        pot = (factors * kfac * torch.abs(S_k)**2).sum() / (2*volume)

        # kfac1 = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        # pot1 = (factors * kfac1 * torch.abs(S_k)**2).sum() / volume
        # print("pot error:",pot.item(),pot1.item(),(pot-pot1).item())

        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)
        if compute_field:
            sk_field = 2 * kfac * torch.conj(S_k)
            q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume
            print("compute field")
        if self.remove_self_interaction and self.exponent == 1:
            #print("Here is self-interaction")
            #pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)  
            diag_sum = kfac.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2 * volume)
            pot -= torch.sum(q**2)*diag_sum
            q_field -= q * (2 *diag_sum)
            print("remove_self_interaction")
        # if(self.remove_zero_mode):
        #     kfac_zero = self.amplitude_1.sum() * torch.sum(q ** 2) / (2*volume)
        #     pot = pot + kfac_zero


        return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor

    def compute_potential_Gaussian_realspace(self, r_raw, q, compute_field=False):
        #print(r_raw.shape)
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        #print(r_ij.shape)
        r_ij_norm = torch.norm(r_ij, dim=-1)

        # min_term = -4*torch.exp(-2 * self.shift_1)  # NaCl charge transfer model use this
        min_term = -1/self.shift_1**2 # Dipeptides model use this
        #print(self.shift_1)
        # print(min_term,"herr",min_term.shape)
        min_term = min_term.view(1, 1, -1)  
        # print(min_term.shape)
        convergence_func_ij = self.amplitude_1.view(1, 1, -1) * torch.exp( torch.square(r_ij_norm).unsqueeze(2) * min_term)
        convergence_func_ij = torch.sum(convergence_func_ij, dim=2)
        # print(convergence_func_ij.shape)
        #print(convergence_func_ij.shape)
        # epsilon = 1e-6
        # r_p_ij = 1.0 / (r_ij_norm + epsilon)
        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
        
        # print(q.shape,compute_field,self.remove_self_interaction)

        Ewald_convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
        Ewald_r_p_ij = 1.0 / (r_ij_norm + 1e-6)
        Ewald_pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * Ewald_r_p_ij.unsqueeze(2) * Ewald_convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0

        idx = torch.arange(convergence_func_ij.size(0))
        convergence_func_ij[idx, idx] = 0

        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0        
        # print("Here:",convergence_func_ij.unsqueeze(2).shape,convergence_func_ij.unsqueeze(2))
        pot = pot.to(torch.float32)

        # print("Ewald vs SOG1:",(Ewald_r_p_ij.unsqueeze(2) * Ewald_convergence_func_ij.unsqueeze(2)), (convergence_func_ij.unsqueeze(2)),(Ewald_r_p_ij.unsqueeze(2) * Ewald_convergence_func_ij.unsqueeze(2)).shape,(convergence_func_ij.unsqueeze(2)).shape)
        #print(pot.shape)
        q_field = torch.zeros_like(q, dtype=q.dtype, device=q.device) # Field due to q
        # Compute field if requested
        if compute_field:
            # [n_node, 1 , n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
            #q_field = torch.sum(q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi
            q_field = torch.sum(q.unsqueeze(1) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False and self.exponent == 1:
            pot += torch.sum(q ** 2) * self.amplitude_1.sum() / self.twopi / 2.0 #torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field + q * self.amplitude_1.sum() / self.twopi 

            Ewald_pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2.

        return pot* self.norm_factor, q_field.unsqueeze(1) * self.norm_factor 
