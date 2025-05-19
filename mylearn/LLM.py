import os  # 操作系统接口
import requests  # HTTP请求库
import math  # 数学函数
import tiktoken  # OpenAI的tokenizer库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from torch.nn import functional as F  # PyTorch函数操作

batch_size = 4  # 每训练步的批量大小
context_length = 16  # 每个批次的token块长度
d_model = 64  # token嵌入的维度大小
num_blocks = 8  # Transformer块数量
num_heads = 4  # 多头注意力的头数
learning_rate = 1e-3  # 学习率
dropout = 0.1  # dropout率
max_iters = 5000  # 最大训练迭代次数
eval_interval = 50  # 评估间隔
eval_iters = 20  # 评估时使用的迭代次数
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 自动选择设备
TORCH_SEED = 1337  # 随机种子
torch.manual_seed(TORCH_SEED)  # 设置随机种子

# 如果数据文件不存在则下载
if not os.path.exists('data/sales_textbook.txt'):
    url = 'https://huggingface.co/datasets/goendalf666/sales-textbook_for_convincing_and_selling/raw/main/sales_textbook.txt'
    os.makedirs('data', exist_ok=True)  # 创建数据目录
    with open('data/sales_textbook.txt', 'w') as f:
        f.write(requests.get(url).text)  # 下载并保存文本

# 读取文本数据
with open('data/sales_textbook.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 使用tiktoken进行token化
encoding = tiktoken.get_encoding("cl100k_base")  # 加载编码器
tokenized_text = encoding.encode(text)  # 将文本编码为token
max_token_value = max(tokenized_text) + 1  # 计算最大token值
tokenized_text = torch.tensor(tokenized_text, dtype=torch.long, device=device)  # 转为tensor

# 划分训练集和验证集(90%训练,10%验证)
split_idx = int(len(tokenized_text) * 0.9)
train_data = tokenized_text[:split_idx]
val_data = tokenized_text[split_idx:]

# 前馈网络
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.dropout = dropout
        # 定义两层线性变换+ReLU激活
        self.ffn = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model * 4),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model * 4, out_features=self.d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.ffn(x)  # 顺序执行定义的前馈网络

# 注意力机制(Attention)
class Attention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.context_length = context_length
        self.dropout = dropout

        # 定义Q,K,V的线性变换层
        self.key_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.query_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)
        self.value_layer = nn.Linear(in_features=self.d_model, out_features=self.head_size, bias=False)

        # 注册下三角掩码(用于防止看到未来信息)
        self.register_buffer('tril', torch.tril(
            torch.ones((self.context_length, self.context_length))))
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        B, T, C = x.shape  # 获取输入形状: 批量大小,时间步数,特征维度

        # 计算Q,K,V
        q = self.query_layer(x)
        k = self.key_layer(x)
        v = self.value_layer(x)

        # 计算注意力权重(QK^T / sqrt(d_k))
        weights = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 应用掩码(将上三角部分设为负无穷)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # softmax归一化
        weights = F.softmax(input=weights, dim=-1)
        weights = self.dropout_layer(weights)  # 对注意力权重应用dropout

        # 计算加权和(权重*V)
        out = weights @ v
        return out

# 多头注意力(MultiHeadAttention)
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.d_model = d_model
        self.context_length = context_length
        self.dropout = dropout

        # 创建多个注意力头
        self.heads = nn.ModuleList([Attention(head_size=self.head_size) for _ in range(self.num_heads)])
        # 定义投影层和dropout
        self.projection_layer = nn.Linear(in_features=self.d_model, out_features=self.d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        # 计算每个头的输出并拼接
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 投影回原始维度
        out = self.projection_layer(out)
        out = self.dropout_layer(out)  # 应用dropout
        return out

#Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.head_size = d_model // num_heads  # 计算每个头的大小
        self.num_heads = num_heads
        self.dropout = dropout

        # 定义多头注意力层和前馈网络
        self.multi_head_attention_layer = MultiHeadAttention(head_size=self.head_size)
        self.feed_forward_layer = FeedForward()
        # 定义两个层归一化
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_model)

    def forward(self, x):
        # 残差连接+多头注意力(先层归一化)
        x = x + self.multi_head_attention_layer(self.layer_norm_1(x))
        # 残差连接+前馈网络(先层归一化)
        x = x + self.feed_forward_layer(self.layer_norm_2(x))
        return x

# 完整模型(TransformerLanguageModel)
class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = d_model
        self.context_length = context_length
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout = dropout
        self.max_token_value = max_token_value

        # token嵌入层
        self.token_embedding_lookup_table = nn.Embedding(
            num_embeddings=self.max_token_value + 1,
            embedding_dim=self.d_model)

        # 定义多个Transformer块+最后的层归一化
        self.transformer_blocks = nn.Sequential(*(
                [TransformerBlock(num_heads=self.num_heads) for _ in range(self.num_blocks)] +
                [nn.LayerNorm(self.d_model)]
        ))

        # 输出线性层(将隐藏状态映射回词表大小)
        self.language_model_out_linear_layer = nn.Linear(
            in_features=self.d_model,
            out_features=self.max_token_value)

    def forward(self, idx, targets=None):
        B, T = idx.shape  # 获取输入形状

        # 创建位置编码表(使用正弦/余弦函数)
        position_encoding_lookup_table = torch.zeros(self.context_length, self.d_model)
        position = torch.arange(0, self.context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)  # 偶数位置用sin
        position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)  # 奇数位置用cos

        # 根据输入长度截取位置编码
        position_embedding = position_encoding_lookup_table[:T, :].to(device)

        # token嵌入+位置编码
        x = self.token_embedding_lookup_table(idx) + position_embedding

        # 通过所有Transformer块
        x = self.transformer_blocks(x)

        # 计算logits(未归一化的输出)
        logits = self.language_model_out_linear_layer(x)

        # 计算损失(如果有目标)
        if targets is not None:
            B, T, C = logits.shape
            # 重塑为二维以计算交叉熵损失
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(input=logits_reshaped, target=targets_reshaped)
        else:
            loss = None
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # 自回归生成文本
        for _ in range(max_new_tokens):
            # 裁剪输入以适应上下文长度
            idx_crop = idx[:, -self.context_length:]
            # 获取预测
            logits, loss = self(idx_crop)
            # 只取最后一个时间步的logits
            logits_last_timestep = logits[:, -1, :]
            # softmax得到概率分布
            probs = F.softmax(input=logits_last_timestep, dim=-1)
            # 从分布中采样
            idx_next = torch.multinomial(input=probs, num_samples=1)
            # 将采样结果追加到输入中
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

#初始化模型
model = TransformerLanguageModel()
model = model.to(device)  # 将模型移动到指定设备

# 获取批数据
def get_batch(split: str):
    # 根据split选择数据集
    data = train_data if split == 'train' else val_data
    # 随机选择起始位置
    idxs = torch.randint(low=0, high=len(data) - context_length, size=(batch_size,))
    # 构建输入和标签(标签是输入的右移一位)
    x = torch.stack([data[idx:idx + context_length] for idx in idxs]).to(device)
    y = torch.stack([data[idx + 1:idx + context_length + 1] for idx in idxs]).to(device)
    return x, y

# 评估损失
@torch.no_grad()  # 禁用梯度计算
def estimate_loss():
    out = {}
    model.eval()  # 切换到评估模式
    for split in ['train', 'valid']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x_batch, y_batch = get_batch(split)
            logits, loss = model(x_batch, y_batch)
            losses[k] = loss.item()
        out[split] = losses.mean()  # 计算平均损失
    model.train()  # 切换回训练模式
    return out

# 训练循环
# 使用AdamW优化器
optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
tracked_losses = list()  # 记录损失

for step in range(max_iters):
    # 定期评估
    if step % eval_iters == 0 or step == max_iters - 1:
        losses = estimate_loss()
        tracked_losses.append(losses)
        print('Step:', step, 'Training Loss:', round(losses['train'].item(), 3), 'Validation Loss:',
              round(losses['valid'].item(), 3))

    # 获取批数据并训练
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)  # 清空梯度
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数

# 保存模型与生成文本
# 保存模型权重
torch.save(model.state_dict(), 'model-ckpt.pt')

# 生成文本
model.eval()  # 切换到评估模式
start = 'The salesperson'  # 起始文本
start_ids = encoding.encode(start)  # 编码为token
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])  # 转为tensor
y = model.generate(x, max_new_tokens=100)  # 生成100个新token
print('---------------')
print(encoding.decode(y[0].tolist()))  # 解码为文本并打印
print('---------------')