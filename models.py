import torch.nn as nn
import torch.nn.functional as F
import torch
from dgllife.model.gnn import GCN
from ACmix import ACmix
from Intention import BiIntention
from scipy.sparse import coo_matrix


def binary_cross_entropy(pred_output, labels):
    loss_fct = torch.nn.BCELoss()
    m = nn.Sigmoid()
    n = torch.squeeze(m(pred_output), 1)
    loss = loss_fct(n, labels)
    return n, loss


def cross_entropy_logits(linear_output, label, weights=None):
    class_output = F.log_softmax(linear_output, dim=1)
    n = F.softmax(linear_output, dim=1)[:, 1]
    max_class = class_output.max(1)
    y_hat = max_class[1]  # get the index of the max log-probability
    if weights is None:
        loss = nn.NLLLoss()(class_output, label.type_as(y_hat).view(label.size(0)))
    else:
        losses = nn.NLLLoss(reduction="none")(class_output, label.type_as(y_hat).view(label.size(0)))
        loss = torch.sum(weights * losses) / torch.sum(weights)
    return n, loss


def entropy_logits(linear_output):
    p = F.softmax(linear_output, dim=1)
    loss_ent = -torch.sum(p * (torch.log(p + 1e-5)), dim=1)
    return loss_ent


class BINDTI(nn.Module):
    def __init__(self, device='cuda', **config):
        super(BINDTI, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        protein_num_head = config['PROTEIN']['NUM_HEAD']
        cross_num_head = config['CROSSINTENTION']['NUM_HEAD']
        cross_emb_dim = config['CROSSINTENTION']['EMBEDDING_DIM']
        cross_layer = config['CROSSINTENTION']['LAYER']

        self.drug_extractor = MolecularGCN(in_feats=drug_in_feats, dim_embedding=drug_embedding,
                                            padding=drug_padding,
                                            hidden_feats=drug_hidden_feats)
        self.protein_extractor = ProteinACmix(protein_emb_dim, num_filters, protein_num_head, protein_padding)

        self.structAware = StructuralAttributeEnhancement(protein_emb_dim, protein_emb_dim)

        self.cross_intention = BiIntention(embed_dim=cross_emb_dim, num_head=cross_num_head, layer=cross_layer,
                                           device=device)
        self.cross_intention2 = CrossAttentionFusion(dim=256, num_heads=cross_num_head)
        self.linear_v_d = nn.Linear(128, 256)
        self.mlp_classifier = MLPDecoder(mlp_in_dim, mlp_hidden_dim, mlp_out_dim, binary=out_binary)

    def forward(self, bg_d, v_p, t=None, mode="train"):

        v_d, e_v_d = self.drug_extractor(bg_d)  # v_d.shape(64, 290, 128) e_v_d.shape(64, 290, 128) 节点增强后的特征，用于（残差连接）
        v_d = v_d + e_v_d
        v_p, e_v_p = self.protein_extractor(v_p)  # v_p.shape:(64, 1200, 128) e_v_p.shape:(64, 128, 128)
        e_v_p = self.structAware(e_v_p)

        f, v_d, v_p, att = self.cross_intention(drug=v_d, protein=v_p)  # f:[64, 256]
        e_f, e_v_d, e_v_p, e_att = self.cross_intention(drug=e_v_d, protein=e_v_p)

        v_d, e_v_d = self.linear_v_d(v_d), self.linear_v_d(e_v_d)
        v_p, e_v_p = self.linear_v_d(v_p), self.linear_v_d(e_v_p)
        v_d = self.cross_intention2(v_d, e_v_d)
        v_p = self.cross_intention2(v_p, e_v_p)
        f = self.cross_intention2(f, e_f)

        score = self.mlp_classifier(f)

        if mode == "train":
            return v_d, v_p, f, score
        elif mode == "eval":
            return v_d, v_p, score, att


class MolecularGCN(nn.Module):
    def __init__(self, in_feats, dim_embedding=128, padding=True, hidden_feats=None, activation=None):
        super(MolecularGCN2, self).__init__()

        # 初始特征转换层
        self.init_transform = nn.Linear(in_feats, dim_embedding, bias=False)

        if padding:
            with torch.no_grad():
                self.init_transform.weight[-1].fill_(0)

        # 基础GCN卷积层
        self.gnn = GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)

        # 计算输出特征维度
        self.output_feats = hidden_feats[-1]

        # 新增多跳邻居的卷积层
        self.multi_hop_convs = nn.ModuleList([
            GCN(in_feats=dim_embedding, hidden_feats=hidden_feats, activation=activation)
            for _ in range(2)  # 假设有两个额外的跳数层，可以根据需求调整
        ])

        # 非线性变换模块，用于语义对齐
        self.semantic_align = nn.Sequential(
            nn.Linear(self.output_feats, self.output_feats),
            nn.ReLU(),
            nn.Linear(self.output_feats, self.output_feats)
        )

    def forward(self, batch_graph):
        # 初始特征获取和转换
        node_feats = batch_graph.ndata.pop('h')
        node_feats = self.init_transform(node_feats)

        # 基础GCN卷积
        node_feats = self.gnn(batch_graph, node_feats)

        # 多跳邻居信息聚合
        multi_hop_feats = []
        for gcn_layer in self.multi_hop_convs:
            hop_feats = gcn_layer(batch_graph, node_feats)
            multi_hop_feats.append(hop_feats)

        # 将多跳邻居的特征聚合（例如，取平均或拼接）
        enhanced_feats = torch.stack([node_feats] + multi_hop_feats, dim=0).mean(dim=0)  # 使用平均聚合

        # 语义对齐
        enhanced_feats = self.semantic_align(enhanced_feats)

        # 调整维度以匹配 batch_size 和 output_feats
        batch_size = batch_graph.batch_size
        node_feats = node_feats.view(batch_size, -1, self.output_feats)
        enhanced_feats = enhanced_feats.view(batch_size, -1, self.output_feats)

        # 返回原始的节点特征和增强后的特征
        return node_feats, enhanced_feats


class ProteinACmix(nn.Module):
    def __init__(self, embedding_dim, num_filters, num_head, padding=True):
        super(ProteinACmix, self).__init__()
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]

        self.acmix1 = ACmix(in_planes=in_ch[0], out_planes=in_ch[1], head=num_head)
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.acmix2 = ACmix(in_planes=in_ch[1], out_planes=in_ch[2], head=num_head)
        self.bn2 = nn.BatchNorm1d(in_ch[2])

        self.acmix3 = ACmix(in_planes=in_ch[2], out_planes=in_ch[3], head=num_head)
        self.bn3 = nn.BatchNorm1d(in_ch[3])

        # 新增的卷积层，用于将通道数从 num_filters[-1] 转换为 128
        self.conv1 = nn.Conv1d(in_channels=in_ch[3], out_channels=128, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool1d(128)  # 使用自适应池化层将长度缩减到128

    def forward(self, v):
        v = self.embedding(v.long())
        v = v.transpose(2, 1)  # 64*128*1200

        v = self.bn1(F.relu(self.acmix1(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn2(F.relu(self.acmix2(v.unsqueeze(-2))).squeeze(-2))
        v = self.bn3(F.relu(self.acmix3(v.unsqueeze(-2))).squeeze(-2))

        # 新增卷积层处理，将形状从 torch.Size([64, 128, num_filters[-1]]) 转换为 torch.Size([64, 128, 128])
        v_modified = self.conv1(v)  # v 的形状变为 torch.Size([64, 128, 128])
        # 应用自适应平均池化将大小从1200缩减到128
        v_modified = self.pool(v_modified)  # 变为 torch.Size([64, 128, 128])

        v = v.view(v.size(0), v.size(2), -1)
        return v, v_modified  # 返回原始的 v 和修改后的 v_modified


class StructuralAttributeEnhancement(nn.Module):
    def __init__(self, hidden_dim, num_neighbors, restart_prob=0.15):
        super(StructuralAttributeEnhancement, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_neighbors = num_neighbors
        self.restart_prob = restart_prob

        # 用于捕捉邻居信息的线性层
        self.neighbor_aggregation = nn.Linear(hidden_dim, hidden_dim)
        self.label_embedding = nn.Embedding(num_neighbors, hidden_dim)  # 用于邻居节点标签的嵌入

    def random_walk(self, adjacency_matrix):
        # 带重启的随机游走策略
        transition_matrix = F.softmax(adjacency_matrix, dim=-1)
        walk_prob = torch.full((transition_matrix.size(0), transition_matrix.size(1)), self.restart_prob,
                               device=transition_matrix.device)

        for _ in range(100):  # 假设进行100步的随机游走
            walk_prob = (1 - self.restart_prob) * torch.matmul(walk_prob.unsqueeze(1), transition_matrix).squeeze(1)

        return walk_prob  # 返回最终的概率分布

    def forward(self, x):
        # x: torch.Size([64, 128, 128])

        batch_size, hidden_dim, seq_length = x.size()

        # Step 1: 构建全连接图
        adjacency_matrix = torch.bmm(x.transpose(1, 2), x)  # torch.Size([64, 128, 128])
        adjacency_matrix = F.softmax(adjacency_matrix, dim=-1)  # 归一化为概率分布

        # Step 2: 结构感知 - 使用带重启的随机游走获取结构相关性分数
        walk_probabilities = self.random_walk(adjacency_matrix)  # torch.Size([64, 128])

        # Step 3: 基于节点之间的结构接近度断开无效边
        mask = walk_probabilities > 0.1  # 假设阈值为0.1，保留有效边
        filtered_x = x * mask.unsqueeze(1)  # 过滤无效边

        # Step 4: 结构增强 - 聚合邻居信息，结合walk_probabilities
        # 使用walk_probabilities作为加权因子
        weighted_neighbors = filtered_x * walk_probabilities.unsqueeze(-1)  # 加权邻居特征
        aggregated_neighbors = self.neighbor_aggregation(weighted_neighbors)  # torch.Size([64, 128, 128])

        # Step 5: 学习邻居节点标签
        neighbor_labels = self.label_embedding(torch.arange(self.num_neighbors, device='cuda:5').repeat(batch_size,
                                                                                                        1))  # torch.Size([64, num_neighbors, hidden_dim])

        # Step 6: 计算相邻节点与目标节点的结构差异
        distances = torch.cdist(filtered_x.transpose(1, 2), neighbor_labels,
                                p=2)  # torch.Size([64, 128, num_neighbors])

        # Step 7: 聚合邻居节点的接近度分数
        proximity_scores = walk_probabilities.unsqueeze(-1) * (1 / (distances + 1e-6))  # 计算接近度分数与距离的关系

        # 使用 proximity_scores 对每个节点进行加权聚合 proximity_scores: torch.Size([64, 128, num_neighbors])
        # 计算每个节点的邻居加权特征
        enhanced_features = torch.sum(proximity_scores.unsqueeze(2) * aggregated_neighbors.unsqueeze(3),
                                      dim=2)  # torch.Size([64, 128, 128])
        enhanced_features = enhanced_features.view(enhanced_features.size(0), enhanced_features.size(2),
                                                   -1)  # enhanced_features torch.Size([64, 节点数量, 特征])

        return enhanced_features  # 返回增强后的张量


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, 'dim must be divisible by num_heads!'

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(dim, dim)

    def forward(self, x1, x2):
        # x1, x2: shape [64, 256]
        b, n1 = x1.size()
        b, n2 = x2.size()

        # 生成查询、键和值
        query = self.wq(x1)  # shape [64, 256]
        key = self.wk(x2)  # shape [64, 256]
        value = self.wv(x2)  # shape [64, 256]

        # Reshape for multi-head attention
        # Reshape for multi-head attention
        query = query.view(b, 1, self.head, self.head_dim).transpose(1, 2)  # [64, num_heads, 1, head_dim]
        key = key.view(b, 1, self.head, self.head_dim).transpose(1, 2)  # [64, num_heads, n2, head_dim]
        value = value.view(b, 1, self.head, self.head_dim).transpose(1, 2)  # [64, num_heads, n2, head_dim]

        # 计算注意力权重
        attn_scores = query @ key.transpose(-2, -1)  # [64, num_heads, 1, n2]
        attn_scores = self.softmax(attn_scores)

        # 计算加权值
        attn_output = attn_scores @ value  # [64, num_heads, 1, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(b, self.head * self.head_dim)  # [64, 1, 256]

        # 通过线性层得到最终输出
        out = self.out(attn_output)  # shape [64, 256]

        return out


class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):  # x.shpae[64, 256]
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x
