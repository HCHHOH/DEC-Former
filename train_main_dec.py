import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from time import time
import shutil

from model.DECFormer import make_model
from utils import get_adjacency_matrix_2direction, get_cluster_adjacency_matrix, get_semantic_adjacency_matrix, load_graphdata_normY_channel1, compute_val_loss, predict_and_save_results

# CUDA：默认0，用1号的话需要把下面这行注释掉
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0')
print("CUDA:", USE_CUDA, DEVICE, flush=True)

# data config
adj_filename = "data/PEMS08/PEMS08.csv"
graph_signal_matrix_filename = "data/PEMS08/PEMS08.npz"
cluster_res_filename = "data/PEMS08/cluster_wd_k6.csv"
num_of_vertices = 170
points_per_hour = 12
num_for_predict = 12
len_input = 12
dataset_name = "PEMS08"
dataset_exp = "PEMS08"
num_of_clusters = 6

#training config
model_name = "DEC-Former"
learning_rate = float(0.001)
start_epoch = 0
epochs = 80
fine_tune_epochs = 5 # fine_tune 微调
print('total training epoch, fine tune epoch:', epochs, ',' , fine_tune_epochs, flush=True)
batch_size = 8
direction = 2 # 1-有向图，2-无向图
encoder_input_size = 1
decoder_input_size = 1
dropout = float(0.0)
kernel_size = 3

num_of_weeks = 1 # 历史数据选择，以周为单位，num=3: [(0, 12), (2016, 2028), (4032, 4044)]
num_of_days = 1 # 历史数据范围，以天为单位
num_of_hours = 1
filename_npz = os.path.join(dataset_name + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '.npz'

num_layers = 4
d_model = 64
dd_model = 512
nb_head = 8
ScaledSAt = bool((1))  # whether use spatial self attention
SE = bool(1)  # whether use spatial embedding
smooth_layer_num = 0
TE = bool(1)
use_LayerNorm = True
residual_connection = True

geo_mx, distance_mx = get_adjacency_matrix_2direction(adj_filename, num_of_vertices) # 获得地理邻接矩阵
sem_mx = get_semantic_adjacency_matrix(cluster_res_filename, num_of_vertices, num_of_clusters) # 获得语义邻接矩阵
adj_mx = get_cluster_adjacency_matrix(geo_mx, cluster_res_filename, num_of_vertices, num_of_clusters) # 地理 + 语义邻接

folder_dir = 'MAE_%s_h%dd%dw%d_layer%d_head%d_dm%d_batch%d_kernel%d' % (model_name, num_of_hours, num_of_days, num_of_weeks, num_layers, nb_head, d_model, batch_size, 20)
print('folder_dir:', folder_dir, flush=True)
params_path = os.path.join('./experiments', dataset_exp, folder_dir)
print(params_path)

# all the input has been normalized into range [-1,1] by MaxMin normalization
train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, _max, _min = load_graphdata_normY_channel1(
    graph_signal_matrix_filename, num_of_hours,
    num_of_days, num_of_weeks, DEVICE, batch_size)

net = make_model(DEVICE, num_layers, encoder_input_size, decoder_input_size, d_model, adj_mx, geo_mx, sem_mx, nb_head, num_of_weeks,
                 num_of_days, num_of_hours, points_per_hour, num_for_predict, dropout=dropout, ScaledSAt=ScaledSAt, SE=SE, TE=TE, kernel_size=kernel_size, smooth_layer_num=smooth_layer_num, residual_connection=residual_connection, use_LayerNorm=use_LayerNorm)

print(net, flush=True)

def train_main():
    if (start_epoch == 0) and (not os.path.exists(params_path)):  # 从头开始训练，就要重新构建文件夹
        os.makedirs(params_path)
        print('create params directory %s' % (params_path), flush=True)
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path), flush=True)
    elif (start_epoch > 0) and (os.path.exists(params_path)):  # 从中间开始训练，就要保证原来的目录存在
        print('train from params directory %s' % (params_path), flush=True)
    else:
        raise SystemExit('Wrong type of model!')

    criterion = nn.L1Loss().to(DEVICE)  # 定义损失函数
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)  # 定义优化器，传入所有网络参数

    total_param = 0
    print('Net\'s state_dict:', flush=True)
    for param_tensor in net.state_dict():
        print(param_tensor, '\t', net.state_dict()[param_tensor].size(), flush=True)
        total_param += np.prod(net.state_dict()[param_tensor].size())
    print('Net\'s total params:', total_param, flush=True)

    print('Optimizer\'s state_dict:')
    for var_name in optimizer.state_dict():
        print(var_name, '\t', optimizer.state_dict()[var_name], flush=True)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf

    # train model
    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        net.load_state_dict(torch.load(params_filename))
        print('start epoch:', start_epoch, flush=True)
        print('load weight from: ', params_filename, flush=True)

    start_time = time()

    # first stage
    print('-' * 99)
    print('starting first stage training')
    for epoch in range(start_epoch, epochs):
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, epoch, process = 'first stage')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

        net.train()  # ensure dropout layers are in train mode
        train_start_time = time()

        for batch_index, batch_data in enumerate(train_loader):
            train_loader_length = len(train_loader)

            encoder_inputs, decoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F) transpose方法的作用是交换矩阵的两个维度
            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1) 表示在-1位置加一个维度
            labels = labels.unsqueeze(-1)

            optimizer.zero_grad()
            outputs = net(encoder_inputs, decoder_inputs, process = 'first stage')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()

            if batch_index % 500 == 0:
                print('training batch %s / %s, loss: %.2f' % (batch_index + 1, train_loader_length, training_loss))

            global_step += 1

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

    print('best epoch:', best_epoch, flush=True)
    print('apply the best val model on the test data set ...', flush=True)
    predict_main(best_epoch, test_loader, test_target_tensor, _max, _min, 'test', process = 'first stage')

    """
    # second stage
    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf
    print('-' * 99)
    print('starting second stage training')
    for epoch in range(start_epoch, epochs):
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, epoch, process = 'second stage')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

        net.train()  # ensure dropout layers are in train mode
        train_start_time = time()

        for batch_index, batch_data in enumerate(train_loader):
            train_loader_length = len(train_loader)

            encoder_inputs, decoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F) transpose方法的作用是交换矩阵的两个维度
            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1) 表示在-1位置加一个维度
            labels = labels.unsqueeze(-1)

            optimizer.zero_grad()
            outputs = net(encoder_inputs, decoder_inputs, process='second stage')
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()

            if batch_index % 500 == 0:
                print('training batch %s / %s, loss: %.2f' % (batch_index + 1, train_loader_length, training_loss))

            global_step += 1

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

    print('best epoch:', best_epoch, flush=True)
    print('apply the best val model on the test data set ...', flush=True)
    predict_main(best_epoch, test_loader, test_target_tensor, _max, _min, 'test', process = 'second stage')

    """
    # """
    # fine tune the model
    optimizer = optim.Adam(net.parameters(), lr=learning_rate * 0.1)
    print('fine tune the model ... ', flush=True)
    for epoch in range(epochs, epochs + fine_tune_epochs):
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        net.train()  # ensure dropout layers are in train mode
        train_start_time = time()

        for batch_index, batch_data in enumerate(train_loader):
            encoder_inputs, decoder_inputs, labels = batch_data
            encoder_inputs = encoder_inputs.transpose(-1, -2)  # (B, N, T, F)
            decoder_inputs = decoder_inputs.unsqueeze(-1)  # (B, N, T, 1)
            labels = labels.unsqueeze(-1)
            predict_length = labels.shape[2]  # T

            optimizer.zero_grad()

            encoder_output = net.encode(encoder_inputs)
            # decode
            decoder_start_inputs = decoder_inputs[:, :, :1, :]
            decoder_input_list = [decoder_start_inputs]

            for step in range(predict_length):
                decoder_inputs = torch.cat(decoder_input_list, dim=2)
                predict_output = net.decode(decoder_inputs, encoder_output)
                decoder_input_list = [decoder_start_inputs, predict_output]

            loss = criterion(predict_output, labels)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            global_step += 1

        print('epoch: %s, train time every whole data:%.2fs' % (epoch, time() - train_start_time), flush=True)
        print('epoch: %s, total time:%.2fs' % (epoch, time() - start_time), flush=True)

        # apply model on the validation data set
        val_loss = compute_val_loss(net, val_loader, criterion, epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(net.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename, flush=True)

    print('best epoch:', best_epoch, flush=True)
    print('apply the best val model on the test data set ...', flush=True)
    predict_main(best_epoch, test_loader, test_target_tensor, _max, _min, 'test')
    # """

def predict_main(epoch, data_loader, data_target_tensor, _max, _min, type, process = 'second stage'):
    '''
    在测试集上，测试指定epoch的效果
    :param epoch: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param _max: (1, 1, 3, 1)
    :param _min: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)
    print('load weight from:', params_filename, flush=True)
    net.load_state_dict(torch.load(params_filename))
    predict_and_save_results(net, data_loader, data_target_tensor, epoch, _max, _min, params_path, type, process)


if __name__ == "__main__":

    train_main()





















