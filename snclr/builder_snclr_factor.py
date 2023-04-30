import torch
import torch.nn as nn


from .nn_memory_norm_bank_multi_keep import NNMemoryBankModuleNormMultiKeep
from .ntx_ent_loss import NTXentLoss


class SNCLR(nn.Module):
    """
    Build a SNCLR model with a base encoder, a momentum encoder, and two MLPs
    """

    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, hidden_dim=384,
                 bank_size=98304, model='res', world_size=8, threshold=False, topk=5,
                 batch=128):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(SNCLR, self).__init__()
        self.T = T
        self.topk = topk
        self.batch = batch
        self.threshold = threshold
        self.world_size = world_size

        self.scale = dim ** -0.5

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

        self.memory_bank = NNMemoryBankModuleNormMultiKeep(size=bank_size, topk=self.topk)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def att_clc(self, q_pre, k_pre):
        q = q_pre
        k = k_pre

        # q = q * self.scale
        att_mask = self.generation_mask(self.topk, self.batch).cuda()
        attn = q @ k.transpose(-2, -1)

        attn = attn + att_mask
        attn = nn.Softmax(dim=-1)(attn)

        return attn

    def generation_mask(self, topk, batch):
        mask_ = torch.eye(batch)
        mask_ = (mask_ - 1) * 999
        mask = mask_.repeat(topk,1).reshape(topk,batch,-1).permute(2,1,0).reshape(batch,topk*batch)

        tmp = -999 * torch.ones(batch, topk*batch*self.world_size) # to set the value after softmax to 0
        tmp[:, topk*batch*torch.distributed.get_rank(): topk*batch*(torch.distributed.get_rank()+1)] = mask

        return tmp

    def soft_neighbors_constrastive(self, q, k, k_keep, q_pre):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        k_keep = nn.functional.normalize(k_keep, dim=1)
        q_pre = nn.functional.normalize(q_pre, dim=1)

        # gather all targets
        k = concat_all_gather(k)
        k_keep = concat_all_gather(k_keep)

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k_keep]) / self.T  # 128， 20480 gor 32 gpus   128, 10240  128,

        logits = logits.log_softmax(dim=-1)

        labels = self.att_clc(q_pre, k)
        b, all_b = labels.shape

        if self.threshold:
            sim_threshold = 1 / self.topk
            mask_threshold = (labels > sim_threshold).type(torch.long)
            labels = labels * mask_threshold


        one_label = torch.eye(b).cuda()
        insert_label = torch.zeros(b, b * self.world_size).cuda()
        insert_label[:, b * torch.distributed.get_rank(): b * (torch.distributed.get_rank() + 1)] = one_label
        fina_label = torch.cat((insert_label.reshape(b,-1,b).unsqueeze(dim=-1), labels.reshape(b, -1, b, self.topk)), dim = -1).reshape(b, -1)

        return torch.mean(torch.sum(-logits * fina_label, dim=-1), dim=-1) * (2 * self.T)

    def forward(self, x1, x2, m, epoch):
        q1_pre = self.base_encoder(x1)
        q2_pre = self.base_encoder(x2)
        q1 = self.predictor(q1_pre)
        q2 = self.predictor(q2_pre)

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1_ = self.momentum_encoder(x1)
            k2_ = self.momentum_encoder(x2)
            k1, k1_keep = self.memory_bank(k1_, update=False)
            k2, k2_keep = self.memory_bank(k2_, update=True)

        loss_con = self.soft_neighbors_constrastive(q1, k2, k2_keep, q1_pre) + self.soft_neighbors_constrastive(q2, k1, k1_keep, q2_pre)
        loss_sim = 0 * loss_con
        loss = loss_con + loss_sim
        return loss, loss_con, loss_sim


class SNCLR_ViT(SNCLR):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        print(f'dim {hidden_dim}-{mlp_dim}-{dim}')
        del self.base_encoder.head, self.momentum_encoder.head

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


