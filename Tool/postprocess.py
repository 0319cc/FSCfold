import torch
import math
import numpy as np
import torch.nn.functional as F


# 约束矩阵的生成函数，计算哪些碱基可以配对
def constraint_matrix_batch(x):
    """
    根据输入的RNA序列的One-Hot编码，生成一个约束矩阵，表示序列中哪些碱基可以合法配对。

    输入:
    x: 一个形状为 [batch_size, seq_length, 4] 的三维张量，表示批次中的RNA序列及其碱基(A, U, C, G)的One-Hot编码。

    返回:
    一个形状为 [batch_size, seq_length, seq_length] 的二维矩阵，表示可以配对的碱基对。
    """
    base_a = x[:, :, 0]  # A碱基
    base_u = x[:, :, 1]  # U碱基
    base_c = x[:, :, 2]  # C碱基
    base_g = x[:, :, 3]  # G碱基

    # 批次大小和序列长度
    batch = base_a.shape[0]
    length = base_a.shape[1]

    # 计算不同碱基对之间的配对关系
    au = torch.matmul(base_a.view(batch, length, 1), base_u.view(batch, 1, length))  # A与U配对
    au_ua = au + torch.transpose(au, -1, -2)  # 对称操作确保双向配对
    cg = torch.matmul(base_c.view(batch, length, 1), base_g.view(batch, 1, length))  # C与G配对
    cg_gc = cg + torch.transpose(cg, -1, -2)
    ug = torch.matmul(base_u.view(batch, length, 1), base_g.view(batch, 1, length))  # U与G配对
    ug_gu = ug + torch.transpose(ug, -1, -2)

    # 返回合并的配对矩阵，表示可以合法配对的碱基对
    return au_ua + cg_gc + ug_gu


# 计算a_hat的配对接触矩阵，结合配对矩阵m
def contact_a(a_hat, m):
    """
    计算接触矩阵 a，用于表示在配对矩阵 m 中的配对情况。

    输入:
    a_hat: 一个矩阵，表示预测的接触矩阵
    m: 一个约束矩阵，表示哪些碱基可以合法配对

    返回:
    a: 经过处理的接触矩阵。
    """
    a = a_hat * a_hat  # 将矩阵 a_hat 进行平方操作
    a = (a + torch.transpose(a, -1, -2)) / 2  # 使矩阵对称化
    a = a * m  # 结合约束矩阵 m 进行配对修正
    return a


# 简单的符号函数，返回输入x的符号
def sign(x):
    return (x > 0).type(x.dtype)  # 返回正数的标志


# logistic 函数实现的 soft_sign 函数
def soft_sign(x):
    """
    使用逻辑函数实现的软符号函数，用于平滑处理输入的符号。

    输入:
    x: 一个张量

    返回:
    一个经过soft_sign处理的张量
    """
    k = 1
    return 1.0 / (1.0 + torch.exp(-2 * k * x))  # logistic函数


# 后处理函数，用于对实用矩阵 u 进行处理
def postprocess(u, x, lr_min, lr_max, num_itr, rho=0.0, with_l1=False, s=math.log(9.0)):
    """
    后处理函数，根据输入的实用矩阵 u 和RNA序列 x，应用梯度下降和拉格朗日乘子法进行优化，生成最终的配对矩阵。

    参数:
    u: 实用矩阵 (utility matrix)，假设为对称矩阵
    x: RNA序列的One-Hot编码，形状为 [batch_size, seq_length, 4]
    lr_min: 最小化步骤的学习率
    lr_max: 最大化步骤的学习率（用于拉格朗日乘子）
    num_itr: 迭代次数
    rho: 稀疏性系数
    with_l1: 是否应用 L1 正则化
    s: 用于 soft_sign 函数中的阈值

    返回:
    a: 最终优化后的接触矩阵
    """
    # 生成约束矩阵 m，表示哪些碱基可以合法配对
    m = constraint_matrix_batch(x).float()

    # 对实用性矩阵 u 应用 soft_sign 函数，使其平滑
    u = soft_sign(u - s) * u

    # 初始化 a_hat 和拉格朗日乘子 lmbd
    a_hat = (torch.sigmoid(u)) * soft_sign(u - s).detach()  # 通过sigmoid函数对u进行平滑
    lmbd = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1).detach()  # 初始化拉格朗日乘子

    # 进行梯度下降的迭代过程
    for t in range(num_itr):
        # 计算 a_hat 的梯度，并更新 a_hat
        grad_a = (lmbd * soft_sign(torch.sum(contact_a(a_hat, m), dim=-1) - 1)).unsqueeze_(-1).expand(u.shape) - u / 2
        grad = a_hat * m * (grad_a + torch.transpose(grad_a, -1, -2))  # 更新梯度
        a_hat -= lr_min * grad  # 通过梯度下降更新a_hat
        lr_min = lr_min * 0.99  # 每次迭代后减小学习率

        # 如果使用L1正则化，应用L1更新
        if with_l1:
            a_hat = F.relu(torch.abs(a_hat) - rho * lr_min)

        # 更新拉格朗日乘子 lmbd
        lmbd_grad = F.relu(torch.sum(contact_a(a_hat, m), dim=-1) - 1)  # 计算lmbd的梯度
        lmbd += lr_max * lmbd_grad  # 更新lmbd
        lr_max = lr_max * 0.99  # 每次迭代后减小学习率

    # 最终的 a_hat 的对称化和接触矩阵计算
    a = a_hat * a_hat
    a = (a + torch.transpose(a, -1, -2)) / 2
    a = a * m  # 将最终的配对矩阵结合约束矩阵 m
    return a  # 返回优化后的接触矩阵
