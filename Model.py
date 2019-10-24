from torch import nn
import torch
import numpy as np
from Agent import Agent
import random
import math
class Model():
    def __init__(self, FLAGS, vocab,embed,data_train=None):
        self.num_relations = FLAGS.num_relations
        self.num_units = FLAGS.num_units
        self.dim_embed_relation = FLAGS.dim_embed_relation
        self.max_edu_dist = FLAGS.max_edu_dist
        self.dim_feature_bi = FLAGS.dim_feature_bi
        self.use_structured = FLAGS.use_structured
        self.use_speaker_attn = FLAGS.use_speaker_attn
        self.use_shared_encoders = FLAGS.use_shared_encoders
        self.use_random_structured = FLAGS.use_random_structured
        self.use_traditional = FLAGS.use_traditional

        self.agent_bi = Agent( FLAGS, vocab,embed,  is_multi=False)
        self.agent_multi = Agent( FLAGS, vocab,embed,  is_multi=True)

        self.params_all = []

        self.params_all += self.agent_multi.parameters()
        self.params_all += self.agent_bi.parameters()

        self.optimizer = torch.optim.Adam(self.params_all,lr = FLAGS.learning_rate,betas=(0.9, 0.99))
        self.grad_unclipped = [param for param in self.params_all]
        self.grad_clipped = torch.nn.utils.clip_grad_norm_(self.grad_unclipped, 5.0)

    def get_hs(self,batch):
        self.max_num_edus = max([len(dialog["edus"]) for dialog in batch])
        self.edus, self.num_posts = [], []  # edus是每个dialog中的词，num posts是每个对话中edu的数目
        for dialog in batch:
            self.edus.append([])
            for edu in dialog["edus"]:
                self.edus[-1].append(edu["tokens"])
            self.num_posts.append(len(dialog["edus"]))
        resc_0,ress_0 = self.agent_bi.ns_encoder(self.edus)
        resc_1,ress_1 = self.agent_multi.ns_encoder(self.edus)
        self.sentences = []  # 存储所有的句子
        self.sentence_idx = []  # 存储句子的编号
        for dialog in batch:  # 获取句子及其编号
            idx = []
            for edu in dialog["edus"]:
                self.sentences.append(edu["tokens"])
                idx.append(len(self.sentences) - 1)
            self.sentence_idx.append(idx)
        self.hs_bi, self.hs_multi, self.hs_idp, self.hc_bi, self.hc_multi = [], [], [], [], []
        for i, dialog in enumerate(batch):
            for j in range(len(dialog["edus"])):
                self.hs_bi.append(ress_0[i][j])
                self.hs_multi.append(ress_1[i][j])
                self.hc_bi.append(resc_0[i][j])
                self.hc_multi.append(resc_1[i][j])


    def build_relation_list(self, batch):
        # relation list
        cnt_golden = 0
        self.relation_list = []
        self.relation_types = []
        self.parents = []
        self.parents_relation = []
        self.parents_hp = []
        self.parents_relation_hp = []
        for k, dialog in enumerate(batch):
            self.parents.append([[] for i in range(len(dialog["edus"]))])
            self.parents_relation.append([[] for i in range(len(dialog["edus"]))])
            self.parents_hp.append([[] for i in range(len(dialog["edus"]))])
            self.parents_relation_hp.append([[] for i in range(len(dialog["edus"]))])
            self.relation_types.append(np.zeros((len(dialog["edus"]), len(dialog["edus"])), dtype=np.int32))
            for relation in dialog["relations"]:
                self.relation_types[k][relation["x"]][relation["y"]] = relation["type"] + 1
                cnt_golden += 1
            for j in range(len(dialog["edus"])):
                r = []
                for i in range(len(dialog["edus"])):
                    if self.relation_types[k][i][j] > 0 and \
                        (i < j and j - i <= self.max_edu_dist):
                            r.append(i)
                self.relation_list.append(r)
        return cnt_golden

    def get_state(self, batch, hs, hc, hp, k, i, j):
        idx_i = self.sentence_idx[k][i]
        idx_j = self.sentence_idx[k][j]
        speaker_i = batch[k]["edus"][i]["speaker"]
        speaker_j = batch[k]["edus"][j]["speaker"]

        h = torch.cat([
            hc[idx_i],
            hs[idx_j],
        ], dim=-1)
        if self.use_structured:
            h = torch.cat([
                h,
                hp[idx_i][speaker_j],
                hc[idx_j],
            ], dim=-1)
        else:
            h = torch.cat([
                h,
                hs[idx_i],
                hc[idx_j],
            ], dim=-1)

        if self.use_traditional:
            h = torch.cat([
                h,
                torch.Tensor([
                    j - i,
                    speaker_i == speaker_j,
                    batch[k]["edus"][i]["turn"] == batch[k]["edus"][j]["turn"],
                    (i in self.parents[k][j]) or (j in self.parents[k][i])
                ])
            ], dim=-1)

        return h

    def new_edge(self, batch, k, i, j, r):
        # bp gradients of hp first before a new parent is added

        self.parents[k][j].append(i)
        self.parents_relation[k][j].append(r)

        if self.use_random_structured:
            i = random.randint(0, j - 1)
            r = random.randint(0, self.num_relations - 1)

        self.parents_hp[k][j].append(i)
        self.parents_relation_hp[k][j].append(r)

        idx_j = self.sentence_idx[k][j]
        if self.use_structured:
            if self.is_root[idx_j]:
                self.is_root[idx_j] = 0
                self.cntp[idx_j] = 1
            else:
                self.cntp[idx_j] += 1

            for l in range(self.cnt_speakers[k]):
                attn = bool(l == batch[k]["edus"][j]["speaker"])
                if not self.use_speaker_attn:
                    attn = 0
    def step(self, batch,is_train):
        cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi = 0, 0, 0, 0
        sum_loss_bi, cnt_loss_bi = torch.FloatTensor([0],),0
        total_loss = torch.FloatTensor([0])
        sum_loss_multi, cnt_loss_multi = torch.FloatTensor([0]),0
        self.get_hs(batch)
        if self.use_structured:
            pass#获取parent路径编码
        else:
            self.hp_bi, self.hp_multi = None, None
        cnt_golden = self.build_relation_list(batch)  # 获得关系列表，cnt_golden为返回的数目
        cur = [(1, 0)] * len(batch)
        unfinished = np.ones(len(batch), dtype=np.int32)#记录当前batch中的对话的处理情况

        for k, dialog in enumerate(batch):
            if len(dialog["edus"]) <= 1:
                unfinished[k] = False

        while (np.sum(unfinished) > 0):
            size = np.sum(unfinished)  # 获取未完成的数目
            golden = np.zeros(size, dtype=np.int32)
            idx = 0
            state = []
            lower = []
            for k, dialog in enumerate(batch):
                state_k = []
                if not unfinished[k]: continue
                j = cur[k][0]  # 存储了当前batch正在处理的edu的序号
                idx_j = self.sentence_idx[k][j]
                lower.append(0)
                for i in range(j):
                    if j - i <= self.max_edu_dist:
                        if (i in self.parents[k][j]): continue
                        state_k.append(self.get_state(
                            batch,
                            self.hs_bi,
                            self.hc_bi,
                            self.hp_bi,
                            k, i, j
                        ) ) # 获得2元组的向量表示

                lower[idx] = max(0,j-self.max_edu_dist)
                state.append(state_k)
                golden[idx] = 0
                flag = False
                for i in self.relation_list[idx_j]:
                    if (i in self.parents[k][j]): continue
                    golden[idx] = i
                    flag = True
                    break
                if not flag:
                    lower[idx] = 0
                idx += 1


            policy = self.agent_bi(state)
            action = self.sample_action(policy)
            if not is_train:
                print('action',action)
                print('golden',golden)
                print()

            idx = 0
            for k, dialog in enumerate(batch):
                if not unfinished[k]: continue
                # predicted a new relation
                if action[idx] != len(dialog["edus"]):
                    cnt_pred += 1
                    if self.relation_types[k][action[idx]][cur[k][0]] > 0:  # 依存预测正确
                        cnt_cor_bi += 1
                idx += 1

            loss = self.Loss_multi(golden,policy,lower,False)

            ##########################上面代码存在out of range的问题#############################
            cnt_loss_bi += 1
            sum_loss_bi += loss
            #predict labels
            idx = 0
            state_multi, golden_multi, idx_multi = [], [], []
            state_multi_train, golden_multi_train, idx_multi_train = [], [], []
            for k, dialog in enumerate(batch):
                if not unfinished[k]: continue
                j = cur[k][0]
                if action[idx] != len(dialog["edus"]):
                    i = action[idx]
                    if self.use_shared_encoders:
                        state_multi.append(self.get_state(
                            batch,
                            self.hs_bi,
                            self.hc_bi,
                            self.hp_bi,
                            k, i, j
                        ))
                    else:
                        state_multi.append(self.get_state(
                            batch,
                            self.hs_multi,
                            self.hc_multi,
                            self.hp_multi,
                            k, i, j
                        ))
                    idx_multi.append((k, i, j))
                for i in range(j):
                    if self.relation_types[k][i][j] > 0:
                        if i in self.parents[k][j]: continue
                        if self.use_shared_encoders:
                            state_multi_train.append(self.get_state(
                                batch,
                                self.hs_bi,
                                self.hc_bi,
                                self.hp_bi,
                                k, i, j)
                            )
                        else:
                            state_multi_train.append(self.get_state(
                                batch,
                                self.hs_multi,
                                self.hc_multi,
                                self.hp_multi,
                                k, i, j)
                            )
                        idx_multi_train.append((k, i, j))
                        golden_multi_train.append(self.relation_types[k][i][j] - 1)
                idx += 1
            if len(idx_multi) > 0:
                with torch.no_grad():
                    policy = self.agent_multi(state_multi)
                    labels = self.sample_action(policy)

            if len(idx_multi_train) > 0:
                policy = self.agent_multi(state_multi_train)
                loss = self.Loss_multi(golden_multi_train,policy,None,True)
                sum_loss_multi += loss
                cnt_loss_multi += 1

            idx, idx_multi = 0, 0
            for k, dialog in enumerate(batch):
                if not unfinished[k]: continue
                # predicted a new relation
                if action[idx] != len(dialog["edus"]):
                    if labels[idx_multi] == self.relation_types[k][action[idx]][cur[k][0]] - 1:
                        cnt_cor_multi += 1
                    idx_multi += 1
                idx += 1

            idx, idx_multi, idx_multi_train = 0, 0, 0

            for k, dialog in enumerate(batch):
                if not unfinished[k]: continue
                # valid prediction
                if action[idx] != len(dialog["edus"]):
                    r = labels[idx_multi]
                    if self.relation_types[k][action[idx]][cur[k][0]] > 0:
                        idx_multi_train += 1
                    idx_multi += 1
                    self.new_edge(batch, k, action[idx], cur[k][0], r)
                cur[k] = (cur[k][0] + 1, 0)
                if cur[k][0] >= len(dialog["edus"]):
                    unfinished[k] = False
                idx += 1
        #print(sum_loss_multi.detach())
        #print(sum_loss_bi.detach())
        total_loss += sum_loss_multi
        total_loss += sum_loss_bi

        self.optimizer.zero_grad()
        try:
            total_loss.backward()
        except:
            print('here')
        self.optimizer.step()

        relations_pred = []
        for k, dialog in enumerate(batch):
            relations_pred.append([])
            for i in range(len(dialog["edus"])):
                for j in range(len(self.parents[k][i])):
                    relations_pred[k].append((self.parents[k][i][j], i, self.parents_relation[k][i][j]))


        for dialog in batch:
            cnt = [0] * len(dialog["edus"])
            for r in dialog["relations"]:
                cnt[r["y"]] += 1
            for i in range(len(dialog["edus"])):
                if cnt[i] == 0:
                    cnt_golden += 1
            cnt_pred += 1
            if cnt[0] == 0:
                cnt_cor_bi += 1
                cnt_cor_multi += 1

        return [
            sum_loss_bi.detach() / cnt_loss_bi if cnt_loss_bi > 0 else 0,
            sum_loss_multi.detach() / cnt_loss_multi if cnt_loss_multi > 0 else 0,
            cnt_golden, cnt_pred, cnt_cor_bi, cnt_cor_multi,
            relations_pred,
        ]



    def Loss_multi(self,golden,policy,lower,is_multi):

        if is_multi:
            loss = []
            for i in range(len(policy)):
                loss.append(policy[i][golden[i]].view(-1))
            loss = torch.cat(loss,dim = 0)

            loss = torch.log(loss)
            loss = torch.mean(loss)
            return -loss
        else:
            try:

                loss = []
                for i in range(len(policy)):
                    loss.append(policy[i][golden[i]-lower[i]].view(-1))
                loss = torch.cat(loss, dim=0)
            except:
                print('hello')

            loss = torch.log(loss)
            loss = torch.mean(loss)
            return -loss

    def sample_action(self, policy):
        action = []
        for p in policy:
            action.append(torch.argmax(p.detach()))
        return action




