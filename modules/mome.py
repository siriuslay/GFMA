import torch
import torch.nn as nn
# from Model.mask_model import MaskModel
from torch.distributions.normal import Normal
# from utils.util import evaluate_ptm, evaluate_moe
from modules import MaskGNN_dict


class MoME(nn.Module):

    def __init__(self, args, pretrained_model_list=None, noisy_gating=True, ptm_arch_list=None):
        super(MoME, self).__init__()
        self.conf = args
        self.noisy_gating = noisy_gating
        self.apply_mask = args.apply_mask
        self.training = args.training
        self.experts = pretrained_model_list
        self.experts_arch = ptm_arch_list
        self.mask_experts = torch.nn.ModuleList()
        self.experts_num = len(pretrained_model_list)
        self.k = args.num_experts
        assert (self.k <= len(pretrained_model_list))
        self.w_gate = nn.Parameter(torch.zeros(args.dim_feat, self.experts_num), requires_grad=True)
        # args.dim_feat / args.gcls * self.experts_num
        # init.xavier_uniform_(self.w_gate)  # initial the gate
        self.w_noise = nn.Parameter(torch.zeros(args.dim_feat, self.experts_num), requires_grad=True)

        self.softmax = nn.Softmax(1)
        self.softplus = nn.Softplus()

        print("Building experts...")
        for idx, expert in enumerate(self.experts):
            if self.apply_mask:
                # self.mask_experts.append(self.build_mask_model(args, expert))
                self.mask_experts.append(self.build_mask_gnn(expert, self.experts_arch[idx]))
        print("Experts has been built")

        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    def forward(self, graphs, features, edge_weight=None):

        batch_indices = graphs.batch_num_nodes()
        start_idx = 0
        features_graphs = torch.zeros(graphs.batch_size, features.shape[1]).to(features.device)
        for idx, num_nodes in enumerate(batch_indices):
            end_idx = start_idx + num_nodes
            features_graphs[idx] = torch.sum(features[start_idx:end_idx], dim=0).clone()
            start_idx = end_idx

        gates, load = self.noisy_top_k_gating(features_graphs, self.training)

        expert_out_list = []
        mask_expert_list = []
        masks_loss = 0
        for expert in self.experts:
            output_i = expert(graphs, features, edge_weight)
            expert_out_list.append(output_i)

        # Gate(experts_output)
        # expert_outputs = torch.stack(expert_out_list, dim=1)
        # mean_experts_out = torch.mean(expert_outputs, dim=1)

        # cat_experts_out = torch.cat(expert_out_list, dim=-1)
        # gates, load = self.noisy_top_k_gating(cat_experts_out, self.training)

        importance = gates.sum(0)
        gate_loss = self.cv_squared(importance) + self.cv_squared(load)

        if self.apply_mask:
            for expert in self.mask_experts:
                mask_output_i = expert(graphs, features, edge_weight)
                mask_expert_list.append(mask_output_i)
                mask_loss = 0
                for name, param in expert.named_parameters():
                    if param.requires_grad:
                        mask_loss += self.RegLoss(param, 2)
                masks_loss += mask_loss
            mask_expert_outputs = torch.stack(mask_expert_list, dim=1)
            y = gates.unsqueeze(dim=-1) * mask_expert_outputs
            expert_out_list = mask_expert_list
        else:
            expert_outputs = torch.stack(expert_out_list, dim=1)
            y = gates.unsqueeze(dim=-1) * expert_outputs

        y = y.mean(dim=1)

        return y, gates, expert_out_list, gate_loss, masks_loss

    def build_mask_gnn(self, model, model_arch):
        model_state_dict = model.state_dict()
        # input_dim, gnnlayer_width, output_dim, gnnlayer_num, normlayer_num, linear_num \
        #     = self.get_params(model_state_dict)
        # 实例化maskgcn
        mask_gnn = MaskGNN_dict[model_arch](self.conf, mask_scale=self.conf.mask_scale, threshold=self.conf.threshold,
                           apply_mask=self.conf.apply_mask)
        # mask_gcn = MaskGCN(num_layers=gnnlayer_num + linear_num, input_dim=input_dim, hidden_dim=gnnlayer_width,
        #                    output_dim=output_dim, mask_scale=self.conf.mask_scale, threshold=self.conf.threshold,
        #                    apply_mask=self.conf.apply_mask, dev=self.conf.gpu)
        mask_model_dict = mask_gnn.state_dict()
        mask_model_dict.update(model_state_dict)
        mask_gnn.load_state_dict(mask_model_dict)

        for name, param in mask_gnn.named_parameters():
            param.requires_grad_(False)
            if 'mask_real' in name:
                param.requires_grad_(True)

        enabled = set()
        for name, param in mask_gnn.named_parameters():
            if param.requires_grad:
                enabled.add(name)

        print(f"Parameters to be updated: {enabled}")
        trainable_param = sum(p.numel() for p in mask_gnn.parameters() if p.requires_grad)
        print(f"number of trainable mask parameters:{trainable_param}")

        return mask_gnn

    def build_mask_model(self, args, pretrained_model):

        mask_model = MaskModel(args, pretrained_model)

        enabled = set()
        for name, param in mask_model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")
        trainable_param = sum(p.numel() for p in mask_model.parameters() if p.requires_grad)
        print(f"number of trainable mask parameters:{trainable_param}")

        return mask_model

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):

        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.experts_num), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.experts_num and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def cv_squared(self, x):

        eps = 1e-10
        # if only experts_num = 1
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):

        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf(((clean_values - threshold_if_in)/noise_stddev))
        prob_if_out = normal.cdf(((clean_values - threshold_if_out)/noise_stddev))
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def get_params(self, state_dict):

        input_dim = state_dict["gcnlayers.0.weight"].shape[0]
        gnnlayer_width = state_dict["gcnlayers.0.weight"].shape[1]
        gnnlayer_num = len(
            [k for k in state_dict.keys() if k.startswith("gcn") and k.endswith("weight")])
        normlayer_num = len([k for k in state_dict.keys() if k.startswith("norm") and k.endswith("weight")])
        output_dim = state_dict["linears_prediction.weight"].shape[0]
        linear_num = len([k for k in state_dict.keys() if k.startswith("linear") and k.endswith("weight")])

        return input_dim, gnnlayer_width, output_dim, gnnlayer_num, normlayer_num, linear_num

    def RegLoss(self, param, k, t=0.2, eta=0.5):
        assert k in [1, 2]
        param = param.view(-1)
        if self.conf.mask_loss == 100:
            param_size = param.size(0)
            count_close_to_one = torch.sum((torch.abs(param - 1) < t).float())
            reg_loss = torch.abs(param.sum() / param_size - eta) + torch.abs(count_close_to_one / param_size - eta)
        elif self.conf.mask_loss == 0:
            reg_loss = torch.norm(param, 2)
        else:
            reg_loss = torch.norm(param-1, 2)
        return reg_loss