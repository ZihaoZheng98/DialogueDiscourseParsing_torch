from torch import nn
import torch
import numpy as np
from NonStructured_Encoder import NonStructured_Encoder
class Agent(nn.Module):
    def __init__(self, FLAGS, vocab,embed, is_multi):
        super(Agent, self).__init__()
        self.num_units = FLAGS.num_units
        self.num_relations = FLAGS.num_relations
        self.is_multi = is_multi
        self.dim_feature_bi = FLAGS.dim_feature_bi
        self.use_structured = FLAGS.use_structured
        self.use_speaker_attn = FLAGS.use_speaker_attn
        self.dim_state = 4 * self.num_units + (self.dim_feature_bi if FLAGS.use_traditional else 0)
        self.ns_encoder = NonStructured_Encoder(FLAGS.dim_embed_word,self.num_units,FLAGS,vocab,embed)
        self.l1 = nn.Linear(self.dim_state,2*self.num_units)
        if self.is_multi is True:
            self.l2 = nn.Linear(2*self.num_units,self.num_relations)
        else:
            self.l2 = nn.Linear(2 * self.num_units, 1)

    def forward(self, data):#输入为数组(batch,len,dim)，输出为数组(batch,dim)
        if self.is_multi is not True:
            policy = []
            for batch in data:
                one_dialog = []
                for d in  batch:
                    d = d.view(-1)
                    d = self.l2(self.l1(d))
                    one_dialog.append(d)

                one_dialog = torch.cat(one_dialog,dim = 0)
                one_dialog.view(-1)

                one_dialog = torch.softmax(one_dialog,dim = -1)
                policy.append(one_dialog)
            return policy
        else:
            policy = []
            for d in data:
                d = d.view(-1)

                d = self.l2(self.l1(d))
                d = torch.softmax(d,dim = -1)
                policy.append(d)
            return policy


