import numpy as np
from torch import nn
import torch.optim as optim
import os
from torch_geometric.loader import DataLoader
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch.nn import Conv1d, Linear, Sequential, ReLU, BatchNorm1d
from torch_geometric.data import Data
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingLR
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
df = pd.read_csv("./output_data.csv", index_col=None, header=None)
data = df.values
reshaped_data = data.ravel(order='F')

reshaped_data = np.array(reshaped_data)
newdata = reshaped_data.reshape((365, 20, 288)).transpose((0, 2, 1))

insert_positions = [1, 2, 3, 6, 7, 9, 11, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 25, 28, 29]
expanded_x = np.zeros((365, 288, 30))
expanded_x[:, :, insert_positions] = newdata

edgeindex = pd.read_csv('./directed_edges.csv')
edgeindex = np.array(edgeindex)
edgeindex = edgeindex.transpose()

newy = pd.read_csv("./all_a_array_2.csv", index_col=None, header=None)
newy = newy.values
# newy = newy.ravel(order='F')
newy = newy.reshape((365 * 288, 6))
expanded_x = expanded_x.reshape((365 * 288, 30))

expanded_x = torch.from_numpy(expanded_x).float().to(device)
newy = torch.from_numpy(newy).float().to(device)
edgeindex = torch.from_numpy(edgeindex).long().to(device)

# 窗口大小和步长
window_size = 24
step_size = 1
# 重新构造数据
data_list = []
for start in range(0, 365 * 288 - window_size + 1, step_size):
    end = start + window_size
    x_window = expanded_x[start:end, :].view(window_size, 30).transpose(1, 0)  # 已经合并前两维，不需要再次 view
    y_window = newy[start:end, :]  # 修正为 view 而不是 shape
    data_window = Data(x=x_window, edge_index=edgeindex, y=y_window)
    data_list.append(data_window)
print(data_list[0])

print(len(data_list))


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


dataset = MyDataset(data_list)

# 划分数据集
dataset_size = len(dataset)
train_size = int(dataset_size * 0.8)
val_size = dataset_size - train_size
train_dataset = dataset[:train_size]
val_dataset = dataset[train_size:]
batch_size = 256
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
import os

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


def plot_attention_weights_with_edges(att_weights, edge_index, title, epoch, normalize=False, cmap='viridis',
                                      num_edges_to_plot=100, save_dir='./attention_weightstuopu'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Normalize attention weights if required
    if normalize:
        att_weights = (att_weights - att_weights.min()) / (att_weights.max() - att_weights.min() + 1e-8)

    num_edges = att_weights.shape[0]
    if num_edges > num_edges_to_plot:
        sampled_indices = np.random.choice(num_edges, num_edges_to_plot, replace=False)
    else:
        sampled_indices = np.arange(num_edges)

    sampled_att_weights = att_weights[sampled_indices]
    sampled_edge_index = edge_index[:, sampled_indices]

    plt.figure(figsize=(10, 8))

    # Flip the y-axis data so that the edge index's largest value is at the top
    sampled_att_weights = np.flipud(sampled_att_weights)

    plt.imshow(sampled_att_weights, cmap=cmap, interpolation='nearest', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('Head Index')
    plt.ylabel('Edge Index')

    # Set x-axis labels to show head indices without decimal points
    num_heads = att_weights.shape[1]
    plt.xticks(ticks=np.arange(num_heads), labels=np.arange(num_heads))

    # Draw vertical lines to separate different heads
    for head in range(num_heads):
        plt.axvline(x=head + 0.5, color='white', linestyle='--', linewidth=0.5)

    # Invert the y-axis to have the largest edge index at the top
    plt.gca().invert_yaxis()

    # Save the plot with epoch in the filename
    save_path = os.path.join(save_dir, f"{title.replace(' ', '_')}_epoch_{epoch}.svg")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved attention weights plot to {save_path}")


from torch.nn.utils import weight_norm


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation * (kernel_size - 1) // 2,
                      dilation=dilation))
        self.batch_norm = nn.LayerNorm(30)
        self.relu = nn.ReLU()
        # 添加一个1x1卷积用于维度匹配
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.relu(self.batch_norm(self.conv(x)))
        # 应用1x1卷积进行维度匹配（如果需要）
        residual = x if self.downsample is None else self.downsample(x)
        return out + residual


class TCNModule(nn.Module):
    def __init__(self, num_channels, kernel_size):
        super(TCNModule, self).__init__()
        layers = []
        dilations = [2 ** i for i in range(len(num_channels))]
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = dilations[i] if i < len(dilations) else 1  # 以防dilations列表短于num_channels列表
            in_channels = num_channels[i - 1] if i > 0 else num_channels[0]
            out_channels = num_channels[i]
            layers.append(TCNBlock(in_channels, out_channels, kernel_size, dilation))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class STGN(torch.nn.Module):
    def __init__(self, num_node_features, num_nodes, channel, head):
        super(STGN, self).__init__()
        self.conv1 = GATv2Conv(num_node_features, out_channels=channel, heads=head)
        self.conv2 = GATv2Conv(channel * head, 512, heads=head)
        # self.conv3 = GCNConv(1024,512)
        # TCN模块
        self.tcn = TCNModule([channel * head, 512, 256], kernel_size=3)  # 示例中的通道数可以根据需要调整
        # 全连接层
        self.fc1 = nn.Linear(256 * num_nodes, 512)  # 第一层全连接层
        self.fc2 = nn.Linear(512, 256)  # 第二层全连接层
        self.fc3 = nn.Linear(256, window_size * 6)  # 最后一层全连接层
        self.num_nodes = num_nodes
        self.all_channels = channel * head

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 图卷积处理
        x, (edge_index, att_weights_conv1) = self.conv1(x, edge_index, return_attention_weights=True)
        x = F.relu(x)
        # x, (edge_index, att_weights_conv2) = self.conv2(x, edge_index, return_attention_weights=True)
        # x = F.relu(x)
        # 重塑x以适应TCN模块
        # x的形状现在是(batch_size*length, num_features)
        # 需要将它变形为(batch_size, num_features, length)，因为TCN期望的输入形状是(batch_size, channels, length)
        x = x.view(-1, self.num_nodes, self.all_channels)  # 重新组织x的形状
        x = x.transpose(1, 2)  # 转置为(batch_size, channels, length)
        # 通过TCN模块
        x = self.tcn(x)
        # x的形状现在应该是(batch_size, num_channels[-1], num_nodes)
        # 全连接层处理
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc1(x))  # 第一层全连接层
        x = F.relu(self.fc2(x))  # 第二层全连接层
        x = self.fc3(x)  # 最后一层全连接层

        return x, att_weights_conv1, edge_index


# 定义包含可学习参数的STGN模型
class M_STGN(torch.nn.Module):
    def __init__(self, num_node_features, num_nodes, channel, head):
        super(M_STGN, self).__init__()
        self.gatcn = STGN(num_node_features, num_nodes, channel, head)
        # 添加两个可学习的参数
        self.sigma1 = nn.Parameter(torch.tensor(0.5))
        self.sigma2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, data):
        out, att_weights_conv1, edge_index = self.gatcn(data)
        return out, att_weights_conv1, edge_index


# Assuming the number of classes is the size of your y's second dimension
model = STGN(num_node_features=window_size, num_nodes=30, channel=32, head=2).to(device)

# Define loss function and optimizer
criterion = torch.nn.L1Loss()  # Use MSELoss for regression tasks
Fn_loss = torch.nn.MSELoss()


# 定义自定义损失函数
def custom_loss(outputs, targets, model):
    L1 = criterion(outputs, targets)
    # 计算特征和与标签和之间的差距的绝对值作为惩罚项
    feature_sum = outputs.sum(dim=1)
    target_sum = targets.sum(dim=1)
    L2 = torch.abs(feature_sum - target_sum).mean()
    return (1 / model.sigma1 ** 2) * L1 + (1 / model.sigma2 ** 2) * L2 + 2 * torch.log(model.sigma1) + 2 * torch.log(
        model.sigma2)


# 定义优化器为Adam
optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=1e-5)
# scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

best_val_loss = 0.13  # 初始化最佳验证损失为无穷大
best_epoch = 0  # 跟踪最佳epoch
save_path = './best_model'  # 设置保存最佳模型的路径


def save_model(epoch, model, optimizer, loss, save_path='./models'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_filename = os.path.join(save_path, f'best_model_epoch_{epoch}_loss_{loss:.4f}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, save_filename)
    print(f'Best model saved to {save_filename} at epoch {epoch} with loss: {loss:.4f}')


# Training loop

def train():
    model.train()
    total_loss = 0

    for step, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out, _, _ = model(data)  # Unpack the output
        y = data.y.view(-1, window_size * 6)
        loss = custom_loss(out, y, model)  # 使用自定义损失函数

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate():
    model.eval()
    total_loss = 0
    mse_total_loss = 0
    all_predictions = []
    all_targets = []
    att_weight = []
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out, _, _ = model(data)  # Unpack the output
            y = data.y.view(-1, window_size * 6)
            loss = custom_loss(out, y, model)  # 使用自定义损失函数
            mse_loss = Fn_loss(out, y)
            total_loss += loss.item()
            mse_total_loss += mse_loss.item()
            all_predictions.extend(out.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
    if epoch == 0 :
        with torch.no_grad():
            out, att_weights_conv1, edge_index = model(val_loader)
            plot_attention_weights_with_edges(att_weights_conv1.cpu().numpy(), edge_index.cpu().numpy(),
                                          title='Conv1 Attention Weights', epoch=epoch, num_edges_to_plot=108)
        # plot_attention_weights_with_edges(att_weights_conv2.cpu().numpy(), edge_index.cpu().numpy(),
        #                                   title='Conv2 Attention Weights', epoch=epoch, num_edges_to_plot=108)
    att_weight.append(att_weights_conv1.cpu().numpy())

    # 将提取的注意力权重转换为 NumPy 数组
    att_weight = np.array(att_weight)  # 去掉多余的维度

    num_samples = att_weight.shape[0]  # 样本数
    num_edges = att_weight.shape[1]  # 边的数量
    num_heads = att_weight.shape[2]  # 头的数量

    # 从 edge_index 确定自环
    edge_index = edge_index.cpu().numpy()
    is_self_loop = edge_index[0] == edge_index[1]
    # 打印 self-loop 检查信息
    print(f'edge_index shape: {edge_index.shape}')
    print(f'self loops: {is_self_loop}')
    # 创建 DataFrame 列表
    df_list = []
    for sample in range(num_samples):
        for head in range(num_heads):
            for edge in range(num_edges):
                df_list.append({
                    'Edge_Index': edge,
                    'Attention_Weight': att_weight[sample, edge, head],
                    'Head': head,
                    'Sample': sample,
                    'Is_Self_Loop': is_self_loop[edge]
                })

    # 转换为 DataFrame
    df_all = pd.DataFrame(df_list)

    # 保存到 CSV 文件
    csv_file_path = './attention_weights_with_self_loops.csv'
    df_all.to_csv(csv_file_path, index=False)

    print(f'注意力权重已经保存到 {csv_file_path}')


    return total_loss / len(val_loader), all_predictions, all_targets, mse_total_loss / len(val_loader)

# Training and Validation
train_losses = []  # 用于记录每个epoch的训练损失
val_losses = []  # 用于记录每个epoch的验证损失
counter = 0
patience = 1000
min_delta = 0.001
epoch = 15
for epoch in range(epoch):  # number of epochs
    train_loss = train()
    # start_time=time.time()
    val_loss, epoch_val_preds, epoch_val_targets, mse_loss = validate()
    # end_time=time.time()
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    # 打印当前的sigma1和sigma2值
    print(f"Current sigma1: {model.sigma1.item():.4f}, sigma2: {model.sigma2.item():.4f}")

    # if epoch <= 50:
    # scheduler.step()
    # current_lr = scheduler.get_last_lr()[0]
    if val_loss < best_val_loss:  # 检查这个epoch的验证损失是否是目前为止最低的
        counter = 0
        best_epoch = epoch  # 更新最佳epoch
        # 仅在当前epoch的验证损失为最低时保存模型
        save_model(epoch, model, optimizer, val_loss, save_path)
        best_val_loss = val_loss  # 更新最佳验证损失
        val_predictions = []
        val_targets = []
        val_predictions.extend(epoch_val_preds)
        val_targets.extend(epoch_val_targets)
        # 将预测值和真实值转换为DataFrame
        val_df = pd.DataFrame({
            'Val_Predictions': [item for sublist in val_predictions for item in sublist],
            'Val_Targets': [item for sublist in val_targets for item in sublist]
        })
        # 保存到CSV文件
        val_df.to_csv('./prediction.csv', index=False)
        print("验证预测值和真实值已保存到CSV文件。")
        print(f'Training complete. Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.4f}')
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping triggered. Stopping training at epoch {epoch}.')  # 早停机制
            break
    print(f"Epoch: {epoch}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, mse_loss:{mse_loss:.4f}")
    # print(f"time:{end_time-start_time:.4f}")
# final_model_path = './models/final_model.pth'
# torch.save(model.state_dict(), final_model_path)
# print(f'Final model parameters saved to {final_model_path}')

# 绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses[10:], label='Training Loss')
plt.plot(val_losses[10:], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 计算MSE
mse = mean_squared_error(val_targets, val_predictions)
print("Mean Squared Error (MSE):", mse)

# 计算MAE
mae = mean_absolute_error(val_targets, val_predictions)  # 0.7
print("Mean Absolute Error (MAE):", mae)

# 计算RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE):", rmse)

# 将预测值和真实值转换为DataFrame
val_df = pd.DataFrame({
    'Val_Predictions': [item for sublist in val_predictions for item in sublist],
    'Val_Targets': [item for sublist in val_targets for item in sublist]
})

# 保存到CSV文件
val_df.to_csv('./prediction.csv', index=False)
print("验证预测值和真实值已保存到CSV文件。")
