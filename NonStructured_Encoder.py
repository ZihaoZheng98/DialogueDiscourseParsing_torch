from torch import nn
import torch
import numpy as np
class NonStructured_Encoder(nn.Module):#得到每个edu的表示
    def __init__(self,embedding_dim,num_units,FLAGS,vocab,embed):
        super(NonStructured_Encoder, self).__init__()
        self.embed = nn.Embedding(embed.shape[0], embed.shape[1])
        self.embed.weight.data.copy_(torch.from_numpy(embed))
        if not FLAGS.train_embedding:
            for p in self.parameters():#将Embedding参数
                p.requires_grad = False
        self.encoder = nn.GRU(input_size=embedding_dim,hidden_size= num_units//2,dropout = FLAGS.keep_prob,bidirectional= True)
        self.encoder_cont = nn.GRU(input_size=num_units,hidden_size= num_units//2,dropout = FLAGS.keep_prob,bidirectional= True)
        self.num_units = num_units
        self.string2index = {}
        self.index2string = {}

        for i,string in enumerate(vocab):
            self.string2index[string] = i
            self.index2string[i] = string
    def text_to_embedding(self,text):
        list = []
        text.append('EOS')
        for word in text:
            if word in self.string2index:
                index = self.string2index[word]
            else:
                index = 0
            list.append(index)

        index_torch = torch.LongTensor(list)
        return self.embed(index_torch)
    def forward(self, data):#输出格式[(seq_len,dim)]
        hns = []
        gns = []
        for dialog in data:
            h_ns = []
            for text in dialog:#处理每个对话
                text = self.text_to_embedding(text).unsqueeze(1)
                _,h_n = self.encoder(text)
                h_ns.append(h_n.view(1,1,-1))
            #对h_n进行拼接
            h_ns = torch.cat(h_ns,dim = 0)
            output,h_n = self.encoder_cont(h_ns)#output为(seq_len,batch,dim)
            hns.append(output.squeeze(1))
            gns.append(h_ns.squeeze(1))
        return gns,hns










