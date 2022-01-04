import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm, trange
import time
import wandb
import os
import argparse

# wandb.init(project="CBOW")

cpu_num = os.cpu_count() # 自动获取最大核心数目
# cpu_num = 4

os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#连续词袋模型
# 通过输入上下文词来预测中心词
# 初始化矩阵
torch.manual_seed(1)



def input():
    print("\n=================1.数据预处理阶段=======================")
    with open('../data/en.txt') as f:
        data = f.read()
    raw_text = data.split()                              # 以空格分割

    # print("raw_text=", raw_text)                             # raw_text数组存储分割结果

    vocab = set(raw_text)                                    # 删除重复元素/单词
    vocab_size = len(vocab)                                  # 这里的size是去重之后的词表大小
    word_to_idx = {word: i for i, word in enumerate(vocab)}  # 由单词索引下标
    idx_to_word = {i: word for i, word in enumerate(vocab)}  # 由下标索引单词

    data = []                                                # cbow那个词表，即{[w1,w2,w4,w5],"label"}这样形式
    for i in range(2, len(raw_text) - 2):  # 类似滑动窗口
        context = [raw_text[i - 2], raw_text[i - 1],
                   raw_text[i + 1], raw_text[i + 2]]
        target = raw_text[i]
        data.append((context, target))

    # print(data[:5])  # (['the', 'present', 'surplus', 'can'], 'food')
    return vocab_size, word_to_idx, idx_to_word, data

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

# 模型结构
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.proj = nn.Linear(embedding_dim, 128)
        self.output = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = sum(self.embeddings(inputs)).view(1, -1) # a=embedding(input)是去embedding.weight中取对应index的词向量！
        out = F.relu(self.proj(embeds))
        out = self.output(out)
        nll_prob = F.log_softmax(out, dim=-1)
        return nll_prob

def train():
    print("=================2.模型训练阶段=======================")
    for epoch in trange(epochs):
        total_loss = 0
        for context, target in tqdm(data):   #tqdm 进度条模块
            context_vector = make_context_vector(context, word_to_idx).cuda()  # 把训练集的上下文和标签都放进去
            target = torch.tensor([word_to_idx[target]]).cuda()
            # print("\ntarget_size", target.size())
            tmp = torch.tensor([0,0]).cuda()
            target1 = target + tmp
            # print("\ntarget1_size", target1.size())
            model.zero_grad()        # 梯度清零
            train_predict = model(context_vector).cuda()  # 开始前向传播
            loss = loss_function(train_predict, target1)
            loss.backward()       # 反向传播
            optimizer.step()     # 更新参
            total_loss += loss.item()
        losses.append(total_loss)    # 更新损失

    print("losses-=", losses)
    print(time.perf_counter()-start)
    W = model.embeddings.weight.cpu().detach().numpy()
    return W

def output(W):
    print("=================4.输出处理=======================")
    word_2_vec = {}  # 生成词嵌入字典，即{单词1:词向量1,单词2:词向量2...}的格式
    for word in word_to_idx.keys():
        word_2_vec[word] = W[word_to_idx[word], :]  # 词向量矩阵中某个词的索引所对应的那一列即为所该词的词向量

    # 将生成的字典写入到文件中
    with open("../output/en_wordvec.txt", 'w') as f:
        for key in word_to_idx.keys():
            f.write('\n')
            f.writelines('"' + str(key) + '":' + str(word_2_vec[key]))
        f.write('\n')

    print("词向量已保存")

def show(W):  # 将词向量降成二维之后，在二维平面绘图
    print("=================5.可视化阶段=======================")
    pca = PCA(n_components=2)  # 数据降维
    principalComponents = pca.fit_transform(W)
    word2ReduceDimensionVec = {}  # 降维后在生成一个词嵌入字典，即即{单词1:(维度一，维度二),单词2:(维度一，维度二)...}的格式

    for word in word_to_idx.keys():
        word2ReduceDimensionVec[word] = principalComponents[word_to_idx[word], :]

    plt.figure(figsize=(20, 20))  # 将词向量可视化
    count = 0
    for word, wordvec in word2ReduceDimensionVec.items():
        if count < 1000:  # 只画出1000个，太多显示效果很差
            plt.scatter(wordvec[0], wordvec[1])
            plt.annotate(word, (wordvec[0], wordvec[1]))
            count += 1
    plt.show()

def test():
    print("=================3.测试阶段=======================")

    context = ['present', 'food', 'can', 'specifically']

    context_vector = make_context_vector(context, word_to_idx).to(device)
    predict = model(context_vector).data.cpu().numpy()  # 预测的值
    # print('Raw text: {}\n'.format(' '.join(raw_text)))
    print('Test Context: {}'.format(context))
    max_idx = np.argmax(predict)  # 返回最大值索引
    print('Prediction: {}'.format(idx_to_word[max_idx]))  # 输出预测的值
    print("CBOW embedding'weight=", model.embeddings.weight)  # 获取词向量，这个Embedding就是我们需要的词向量，他只是一个模型的一个中间过程



gpus=[0, 1]

torch.cuda.set_device('cuda:{}'.format(gpus[0]))
if __name__ == '__main__':
    # -------------------0.参数设置--------------------#
    start = time.perf_counter()  #计算时间

    learning_rate = 0.001             # 学习率 超参数
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')     # 放cuda或者cpu里
    print("运行方式：", device)
    # print('CPU核心数', cpu_num)
    context_size = 2        # 上下文信息，上下文各2个词，中心词作为标签
    embedding_dim = 100      # 词向量维度，一般都是要100-300个之间
    epochs = 1                                     # 训练次数
    losses = []                           # 存储损失的集合
    loss_function = nn.NLLLoss()  # 最大似然 / log似然代价函数

    # -------------------1.输入与预处理模块---------------------#
    vocab_size, word_to_idx, idx_to_word, data = input()

    # -------------------2.训练模块---------------------------#

    model = CBOW(vocab_size, embedding_dim).to(device)
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", type=int)
    # args = parser.parse_args()
    # torch.distributed.init_process_group(backend='nccl')
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model.to(device), device_ids=gpus, output_device=gpus)
        # model = torch.nn.parallel.DistributedDataParallel(
        #     model,
        #     device_ids=gpus,
        #     output_device=gpus
        #     )
    optimizer = optim.SGD(model.parameters(), lr=0.001)         # 优化器
    W = train()
    # -------------------3.测试模块---------------------------#
    test()
    # -------------------4.输出处理模块------------------------#
    print("训练测试时间",time.perf_counter()-start)
    end1=time.perf_counter()
    output(W)
    print("输出处理时间", time.perf_counter() - end1)
    # -------------------5.可视化模块-------------------------#

    end2=time.perf_counter()
    show(W)
    print("可视化时间", time.perf_counter() - end2)
    end = time.perf_counter()
    print("运行耗时：", end - start)
    print(torch.cuda.get_device_name(0))
