import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR  # 导入 StepLR

from Module_Arthur import *

# 训练集
train_set = torchvision.datasets.CIFAR10(root='CIFAR_dataset', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 测试集
test_set = torchvision.datasets.CIFAR10(root='CIFAR_dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

# 训练集、测试集长度展示
train_set_size = len(train_set)
test_set_size = len(test_set)
print('训练集的长度为{}'.format(train_set_size))
print('测试集的长度为{}'.format(test_set_size))

# 利用 DataLoader 加载数据集
train_dataloader = DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_set, batch_size=64, shuffle=False)

# 创建神经网络
arthur = Arthur()
arthur = arthur.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(arthur.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)

# 添加 StepLR 学习率衰减
step_size = 10  # 每隔 10 个 epoch 衰减一次
gamma = 0.1  # 学习率衰减因子
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# 设置神经网络的一些参数
total_train_step = 0  # 记录训练次数
total_test_step = 0  # 记录测试次数
epoch = 30  # 训练轮数

# 添加 TensorBoard
writer = SummaryWriter('logs_train')

# 训练循环
for i in range(epoch):
    print('------第 {} 轮训练开始------'.format(i + 1))

    # 训练模式
    arthur.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()

        # 前向传播
        output = arthur(imgs)
        loss = loss_fn(output, targets)

        # 反向传播、优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 记录训练损失
        total_train_step += 1
        if total_train_step % 100 == 0:
            print('第 {} 次训练时, loss = {}'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试模式
    arthur.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()

            # 前向传播
            output = arthur(imgs)
            loss = loss_fn(output, targets)

            # 计算准确率
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy += accuracy
            total_test_loss += loss.item()

    # 记录测试损失和准确率
    print('整体测试集上的 loss：{}'.format(total_test_loss))
    print('整体测试集上的正确率：{}'.format(total_accuracy / test_set_size))
    writer.add_scalar('test_accuracy', total_accuracy / test_set_size, total_test_step)
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    total_test_step += 1

    # 更新学习率
    scheduler.step()  # 在每个 epoch 结束后更新学习率

    # 打印当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print('当前学习率：{}'.format(current_lr))

    # 保存模型
    if i % 10 == 0:
        torch.save(arthur, 'arthur_{}.pth'.format(i))
    print('模型已保存')

# 关闭 TensorBoard
writer.close()