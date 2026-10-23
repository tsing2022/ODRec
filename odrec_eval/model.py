from pathlib import Path
import numpy as np
import torch as t
from torch import nn


def innerProduct(a, b):
    return (a * b).sum(-1)


class ODRec(nn.Module):

    def __init__(self, data_handler, config, feature_dir):
        super().__init__()
        self.config = config
        self.feature_dir = Path(feature_dir)
        self.user_num = data_handler.user_num
        self.item_num = data_handler.item_num
        self.embedding_size = config['model']['embedding_size']
        self.adj = data_handler.torch_adj
        self.layer_num = self.config['model']['layer_num']
        self.scorenum = 0
        if self.config['social'] == True:
            self.scorenum += 1
        self._load_offline_cognitive_features()
        self.channel_list = [int(channle_x) for channle_x in self.config['channel'].split(',')]
        if self.config['profile'] == True:
            self.low_dim = self.config['gpt']
            if self.config['relu'] == False:
                self.low_mlp = nn.Sequential(nn.Linear(self.uProfEmb_size, self.low_dim * 2), nn.Linear(self.low_dim * 2, self.low_dim))
            elif self.config['relu'] == True:
                self.low_mlp = nn.Sequential(nn.Linear(self.uProfEmb_size, self.low_dim * 2), nn.ReLU(), nn.Linear(self.low_dim * 2, self.low_dim))
        if self.config['MBTI'] == True:
            self.low_dim_2 = self.config['mbti']
            if self.config['relu'] == False:
                self.low_mlp_2 = nn.Sequential(nn.Linear(self.uANAEmb_size, self.low_dim_2 * 2), nn.Linear(self.low_dim_2 * 2, self.low_dim_2))
            elif self.config['relu'] == True:
                self.low_mlp_2 = nn.Sequential(nn.Linear(self.uANAEmb_size, self.low_dim_2 * 2), nn.ReLU(), nn.Linear(self.low_dim_2 * 2, self.low_dim_2))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(self.scorenum, self.scorenum * 16), nn.ReLU(), nn.Linear(self.scorenum * 16, self.scorenum * 8), nn.ReLU(), nn.Linear(self.scorenum * 8, self.scorenum * 4), nn.ReLU(), nn.Linear(self.scorenum * 4, self.scorenum * 2), nn.ReLU(), nn.Linear(self.scorenum * 2, self.scorenum), nn.Sigmoid())
        self.w = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.user_num, self.scorenum)))
        if self.config['w_1'] == 'kexuexi':
            self.w_1 = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.user_num, 1)))
        else:
            self.w_1 = int(self.config['w_1']) * 1.0 / 10
        self.user_embeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.user_num, self.embedding_size)))
        self.item_embeds = nn.Parameter(nn.init.xavier_uniform_(t.empty(self.item_num, self.embedding_size)))
        self.final_embeds = None
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _propagate(self, adj, embeds):
        return t.sparse.mm(adj, embeds)

    def forward(self, adj):
        if self.final_embeds is not None:
            return (self.final_embeds[:self.user_num], self.final_embeds[self.user_num:])
        embeds = t.cat([self.user_embeds, self.item_embeds], dim=0)
        embeds_list = [embeds]
        for i in range(self.layer_num):
            embeds = self._propagate(adj, embeds_list[-1])
            embeds_list.append(embeds)
        embeds = sum(embeds_list)
        self.final_embeds = embeds
        return (embeds[:self.user_num], embeds[self.user_num:])

    def _load_offline_cognitive_features(self):
        files = {'SCD': ('user_profile_pure_emb.npy', 'gpt_req_embedding.npy', 'converted_user_mbti_analyse_scd_gpt_emb.npy', 'converted_soul_mbti_analyse_emb.npy'), 'MGD': ('user_profile_emb.npy', 'gpt_req_emb.npy', 'converted_user_mbti_mgd_analyse_emb.npy', 'converted_soul_mbti_mgd_analyse_emb.npy')}
        dataset = self.config['data']['name']
        if dataset not in files:
            raise ValueError('ODRec supports dataset SCD or MGD')
        sp_self, sp_partner, pk_self, pk_partner = files[dataset]
        predir = str(self.feature_dir) + '/'

        def load(relative):
            return t.from_numpy(np.load(predir + relative, allow_pickle=False)).float().to(self.config['device'])
        if self.config['profile']:
            self.uProfEmb = load('contextual/' + sp_self)
            self.uProfEmb_size = self.uProfEmb.shape[1]
            self.uGPTEmb = load('contextual/' + sp_partner)
            self.scorenum += 1
        if self.config['MBTI']:
            self.uANAEmb = load('mental/' + pk_self)
            self.uANAEmb_size = self.uANAEmb.shape[1]
            self.sANAEmb = load('mental/' + pk_partner)
            self.scorenum += 1

    def compatibility_channel_names(self):
        return tuple((name for name, enabled in (('social', self.config['social']), ('profile', self.config['profile']), ('mbti', self.config['MBTI'])) if enabled))

    def project_sociological_personas(self):
        return (self.low_mlp(self.uProfEmb), self.low_mlp(self.uGPTEmb))

    def project_psychological_kernels(self):
        return (self.low_mlp_2(self.uANAEmb), self.low_mlp_2(self.sANAEmb))

    def retrieve_collaborative_memory(self, adj):
        return self.forward(adj)

    def encode_cognitive_state(self, adj=None):
        if adj is None:
            adj = self.adj
        state = {'memory': None, 'persona': None, 'kernel': None}
        if self.config['social']:
            state['memory'] = self.retrieve_collaborative_memory(adj)
        if self.config['profile']:
            state['persona'] = self.project_sociological_personas()
        if self.config['MBTI']:
            state['kernel'] = self.project_psychological_kernels()
        return state

    def unified_cognitive_representation(self, state, role='seeker'):
        if role not in ('seeker', 'candidate'):
            raise ValueError('role must be seeker or candidate')
        blocks = []
        if state['memory'] is not None:
            blocks.append(state['memory'][0 if role == 'seeker' else 1])
        for key, index in [('persona', 0), ('kernel', 0), ('persona', 1), ('kernel', 1)]:
            if state[key] is not None:
                blocks.append(state[key][index])
        if not blocks:
            raise ValueError('At least one compatibility component must be enabled')
        return t.cat(blocks, dim=-1)

    @staticmethod
    def _compatibility_dot(seeker, candidate, pairwise=False):
        if pairwise:
            return innerProduct(seeker, candidate).unsqueeze(-1)
        if candidate.dim() == 2:
            return t.matmul(seeker.unsqueeze(1), candidate.unsqueeze(2)).reshape((-1, 1))
        return t.matmul(seeker.unsqueeze(1), candidate.transpose(-1, -2)).reshape((-1, candidate.size(1)))

    def intuitive_matching(self, seeker_memory, candidate_memory, pairwise=False):
        return self._compatibility_dot(seeker_memory, candidate_memory, pairwise)

    def cognitive_alignment(self, seeker_partner, candidate_self, pairwise=False):
        return self._compatibility_dot(seeker_partner, candidate_self, pairwise)

    @staticmethod
    def build_score_tensor(intuitive=None, external=None, internal=None):
        channels = [score for score in (intuitive, external, internal) if score is not None]
        if not channels:
            raise ValueError('At least one score channel is required')
        if any((score.dim() != 2 or score.size() != channels[0].size() for score in channels)):
            raise ValueError('Compatibility channels must have matching [B,K] shapes')
        return t.cat([score.unsqueeze(-1) for score in channels], dim=-1)

    def compatibility_tensor(self, state, seekers, candidates, pairwise=False):
        intuitive = external = internal = None
        if state['memory'] is not None:
            source, target = state['memory']
            intuitive = self.intuitive_matching(source[seekers], target[candidates], pairwise)
        if state['persona'] is not None:
            candidate_self, seeker_partner = state['persona']
            external = self.cognitive_alignment(seeker_partner[seekers], candidate_self[candidates], pairwise)
        if state['kernel'] is not None:
            candidate_self, seeker_partner = state['kernel']
            internal = self.cognitive_alignment(seeker_partner[seekers], candidate_self[candidates], pairwise)
        X = self.build_score_tensor(intuitive, external, internal)
        return X

    def adaptive_channel_weights(self, pooled_scores, seekers):
        if self.config['w_1'] == 'kexuexi':
            mixed = t.sigmoid(self.w_1)[seekers] * self.fc(pooled_scores) + (1 - t.sigmoid(self.w_1)[seekers]) * t.sigmoid(self.w)[seekers]
        else:
            mixed = self.w_1 * self.fc(pooled_scores) + (1 - self.w_1) * t.sigmoid(self.w)[seekers]
        return mixed / mixed.sum(-1).unsqueeze(-1)

    def decision_arbiter(self, X, seekers, flag=0):
        x = X.permute(0, 2, 1).unsqueeze(-1)
        x = x[:, self.channel_list, :, :]
        b, c, h, w = x.size()
        pooled_scores = self.avg_pool(x).view(b, c)
        alpha = self.adaptive_channel_weights(pooled_scores, seekers).view(b, c, 1, 1)
        channel_names = self.compatibility_channel_names()
        selected_names = [channel_names[index] for index in self.channel_list]
        fused_channels = x * alpha.expand_as(x)
        scores = fused_channels.sum(dim=1).squeeze(-1)
        if flag == 1:
            contributions = []
            for name in ('social', 'profile', 'mbti'):
                value = fused_channels[:, selected_names.index(name), :].mean() if name in selected_names else fused_channels.sum() * 0
                contributions.append(value)
            return tuple(contributions) + (scores,)
        return scores

    def score_compatibility_blocks(self, seekers, positives, negatives, adj=None, pairwise=False, return_tensors=False):
        state = self.encode_cognitive_state(adj)
        X_pos = self.compatibility_tensor(state, seekers, positives, pairwise)
        X_neg = self.compatibility_tensor(state, seekers, negatives, pairwise)
        positive_scores = self.decision_arbiter(X_pos, seekers)
        negative_scores = self.decision_arbiter(X_neg, seekers)
        if return_tensors:
            return (positive_scores, negative_scores, {'X_pos': X_pos, 'X_neg': X_neg})
        return (positive_scores, negative_scores)

    def predict_candidate_block(self, seekers, candidates, adj=None, return_tensor=False):
        state = self.encode_cognitive_state(adj)
        X = self.compatibility_tensor(state, seekers, candidates, pairwise=False)
        scores = self.decision_arbiter(X, seekers)
        return (scores, X) if return_tensor else scores
