import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

class EncoderImage(nn.Module):

    def __init__(self, img_dim, embed_size):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(img_dim*2, embed_size)


    def forward(self, images):
        return self.fc(images)

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImage, self).load_state_dict(new_state)

class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers, use_bi_gru=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
    
    """
    x:raw txt
    """
    def forward(self, x, lengths):
        """Handles variable size captions
        """
        
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)/2] + cap_emb[:,:,cap_emb.size(2)/2:])/2
        
        return cap_emb, cap_len #(batch_size,seq_length,embed_size)

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
    
def func_attention(query, context, opt):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)

    queryT = torch.transpose(query, 1, 2)
    
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*opt.lambda_softmax)
    attn_filter = (attn>(1.0/sourceL)).float()
    attn = attn*attn_filter
    attn = attn.view(batch_size, queryL, sourceL)
    attnT = torch.transpose(attn, 1, 2).contiguous()
    contextT = torch.transpose(context, 1, 2)
    weightedContext = torch.bmm(contextT, attnT)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext    


def build_relation(im):
    batch_size = im.shape[0]
    img_part_num = im.shape[1]
    embed_size = im.shape[2]
    img_relation_num = img_part_num*(img_part_num-1)
    relation_im = torch.zeros(batch_size,img_relation_num,embed_size*2)
    for i in range(batch_size):
        index = 0
        for m in range(img_part_num):
            for n in range(img_part_num):
                if m!=n:
                    relation_im[i,index,:] = torch.cat([im[i,m,:],im[i,n,:]])
                    index = index+1
    return relation_im
    
"""
im:(batch_size,36,embed_size)
s:(batch_size,36,embed_size)
"""
def compute_sim_score(im, s, cap_lens, opt):
    batch_size_im = im.shape[0]
    batch_size_txt = s.shape[0]
    img_part_num = im.shape[1]
    embed_size = im.shape[2]
    scores = torch.zeros(batch_size_txt,batch_size_im)
    im = im.float()
    s = s.float()
    for i in range(batch_size_txt):
        n_word = cap_lens[i]        
        txt_one = s[i, :n_word, :].unsqueeze(0) #(img_part_num,embed_size)       
        txt_one_expand = txt_one.repeat(batch_size_im, 1, 1)#(img_part_num,embed_size)->(batch_size_im,img_part_num,embed_size)
        #(batch_size_im,img_part_num,embed_size) cos (batch_size_im,img_part_num,embed_size)
        weiContext = func_attention(im, txt_one_expand, opt)
        score_i_all = cosine_similarity(im, weiContext, dim=2).view(batch_size_im,img_part_num)
        
        ####Average####
        #score_i_avg = score_i_filter.mean(dim=1, keepdim=False)
        
        ####KNN####
        score_i_sort,_ = score_i_all.sort()
        score_i_filter = score_i_sort[:,img_part_num/3:]
        score_i_avg = score_i_filter.mean(dim=1, keepdim=False)
        
        ####softmax####
        #score_i_all_softmax = nn.Softmax(dim=1)(score_i_all*4)
        #score_i_all_filter = (score_i_all_softmax>(1.0/img_part_num)).float()
        #score_i_all_softmax_filtered = score_i_all_softmax*score_i_all_filter
        #score_i_all_softmax_filtered_sum = score_i_all_softmax_filtered.sum(dim=1, keepdim=True)
        #score_i_all_filter_sum = score_i_all_filter.sum(dim=1, keepdim=True)
        #score_i_avg = torch.div(score_i_all_softmax_filtered_sum.float(),score_i_all_filter_sum.float()).view(batch_size_im)
                
        scores[i] = score_i_avg
    return scores

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        scores = compute_sim_score(im, s, s_l, self.opt)
        
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        #if torch.cuda.is_available():
            #I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()


class Local_Alignment(object):
    def __init__(self, opt):
    
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.img_dim, opt.embed_size)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim, opt.embed_size, opt.num_layers, use_bi_gru=opt.bi_gru)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0
        
    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = build_relation(images)
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward + attention
        img_emb = self.img_enc(images)

        # cap_emb (tensor), cap_lens (list) + attention
        cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, cap_lens

    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update('Le', loss.item(), img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(img_emb, cap_emb, cap_lens)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
