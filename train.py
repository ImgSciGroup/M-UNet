
from torch.nn import functional as F
from loss import focal_loss, dice_loss
# from model.MCMnet import UNet
from matplotlib import pyplot as plt
import os
from args import  args
from model.MCMnet import UNet
from util.dataset import ISBI_Loader
from torch import optim
# from tensorboardX import SummaryWriter
import torchvision.transforms as Transforms
import torch.utils.data as data
import time
import glob
import cv2
from util.EvaluationNew import Evaluation
import numpy as np
import torch
import torch.nn as nn
from util.EvaluationNew import Index

def train_net(net, device, data_path, epochs=args.epoch, batch_size=args.batch_size, lr=0.0001, ModelName='FC_EF', is_Transfer= False):
    print('Conrently, Traning Model is :::::'+ModelName+':::::')
    if is_Transfer:
        print("Loading Transfer Learning Model.........")
        # BFENet.load_state_dict(torch.load('Pretrain_BFE_'+ModelName+'_model_epoch75_mIoU_89.657089.pth', map_location=device))
    else:
        print("No Using Transfer Learning Model.........")

    # 加载数据集
    isbi_dataset = ISBI_Loader(data_path=data_path, transform=Transforms.ToTensor())
    train_loader = data.DataLoader(dataset=isbi_dataset,
                                   batch_size=batch_size,
                                   shuffle=False)
    # 定义RMSprop算法
    # optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=1e-5, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    # scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70], gamma=0.9)
    scheduler1 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80, 85, 90], gamma=0.9)
    if not os.path.exists('txt/'+args.dataset):
        os.makedirs('txt/'+args.dataset)
    f_loss = open('txt/'+args.dataset+'/train_loss.txt', 'w')
    f_time = open('txt/'+args.dataset+'/train_time.txt', 'w')
    # 训练epochs次
    loss_list = []
    epoch_list = []
    total_loss=0
    x =args.loss_weight
    #
    start = time.time()
    # while (x <1):
    #      x = round(x * 100) / 100.0
    for epoch in range(1, epochs+1):
            net.train()
            # 训练模式
            # learning rate delay
            best_loss = float('inf')
            best_F1 = float('inf')
            # 按照batch_size开始训练
            total=0
            num = 0
            starttime = time.time()
            print('==========================epoch = '+str(epoch)+' ============:损失权重为'+str(x)+' ==========================')
            for image1, image2, label in train_loader:
                # caenet.train()
                optimizer.zero_grad()
                # print(label)
                # 将数据拷贝到device中
                image1 = image1.to(device=device)
                image2 = image2.to(device=device)
                label = label.to(device=device)
                pred = net(image1, image2)

                loss_ = focal_loss(label, pred)
                loss_1 = dice_loss(pred, label)
                loss_ = x*loss_1 +(1-x)* loss_
                if num == 0:
                    if epoch == 0:
                        f_loss.write('Note: epoch (num, edge_loss, focal_loss, BCE_loss, total_loss)\n')
                        f_loss.write('epoch = ' + str(epoch) + '\n')
                    else:
                        f_loss.write('epoch = ' + str(epoch) + '\n')

                print(str(epoch)+'/' + str(epochs)+':::::'+'lr='+str(optimizer.param_groups[0]['lr'])+':::::'+str(num)+'/'+str(int(len(isbi_dataset)/batch_size)))
                # print('Loss/train', total_loss.item())
                print('Loss/train', loss_.item())
                total=loss_.item()+total
                print('-----------------------------------------------------------------------')
                """
                if epoch % 10 == 0:
                    if total_loss < best_loss:
                        best_loss = total_loss
                        BFE_path = 'best_BFE_SPM_model_epoch' + str(epoch) + '.pth'
                        BCD_path = 'best_BCD_SPM_model_epoch' + str(epoch) + '.pth'
                        torch.save(BFENet.state_dict(), BFE_path)
                        torch.save(net.state_dict(), BCD_path)
                """

                # 更新参数
                loss_.backward()
                optimizer.step()
                num += 1
            # learning rate delay
            scheduler1.step()
            epoch_list.append(epoch)
            loss_list.append(total/6101)
            f_loss.write(str(total / num) + '\n')


    endtime = time.time()
    if not os.path.exists('./model_pth/'+args.dataset):
        os.makedirs('./model_pth/'+args.dataset)
    torch.save(net.state_dict(), './model_pth/'+args.dataset+'/'+ str(x) + '.pth')
    f_loss.write(str(endtime - start) + '\n')

def val(net1, device, epoc):
    net1.eval()
    tests1_path = glob.glob('./data_Italy/test/image1/*.jpg')
    tests2_path = glob.glob('./data_Italy/test/image2/*.jpg')
    label_path = glob.glob('./data_Italy/test/label/*.jpg')
    trans = Transforms.Compose([Transforms.ToTensor()])
    TPSum = 0
    TNSum = 0
    FPSum = 0
    FNSum = 0
    C_Sum_or = 0
    UC_Sum_or = 0
    num = 0
    val_acc = open('val_acc.txt', 'a')
    val_acc.write('===============================' + 'epoch=' + str(epoc) + '==============================\n')
    for tests1_path, tests2_path, label_path in zip(tests1_path, tests2_path, label_path):
        num += 1
        # 读取图片
        test1_img = cv2.imread(tests1_path)
        test2_img = cv2.imread(tests2_path)
        label_img = cv2.imread(label_path)
        label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
        test1_img = trans(test1_img)
        test2_img = trans(test2_img)
        test1_img = test1_img.unsqueeze(0)
        test2_img = test2_img.unsqueeze(0)
        test1_img = test1_img.to(device=device, dtype=torch.float32)
        test2_img = test2_img.to(device=device, dtype=torch.float32)
        # 使用网络参数，输出预测结果
        list = []
        out = net1(test1_img, test2_img)

        # 提取结果
        pred = np.array(out.data.cuda()[0])[0]
        # 处理结果
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        monfusion_matrix = Evaluation(label=label_img, pred=pred)
        TP, TN, FP, FN, c_num_or, uc_num_or = monfusion_matrix.ConfusionMatrix()
        TPSum += TP
        TNSum += TN
        FPSum += FP
        FNSum += FN
        C_Sum_or += c_num_or
        UC_Sum_or += uc_num_or

        if num > 400 and num % 10 == 0:
            Indicators = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
            IoU, c_IoU, uc_IoU = Indicators.IOU_indicator()
            OA, Precision, Recall, F1 = Indicators.ObjectExtract_indicators()
            print("OA=", str(float('%4f' % OA)), "^^^^^", "mIoU=", str(float('%4f' % IoU)), "^^^^^", "c_mIoU=", str(float('%4f' % c_IoU)), "^^^^^", "uc_mIoU=", str(float('%4f' % uc_IoU)), "^^^^^", "Precision=",
                  str(float('%4f' % Precision)), "^^^^^", "Recall=", str(float('%4f' % Recall)), "^^^^^", "mF1=", str(float('%4f' % F1)))
            val_acc.write('mIou = ' + str(float('%2f' % IoU)) + ',' + 'c_mIoU = ' +
                          str(float('%2f' % (c_IoU))) + ',' +
                          'uc_mIoU = ' + str(float('%2f' % (uc_IoU))) + ',' +
                          'F1 = ' + str(float('%2f' % (F1))) + '\n')
    Indicators2 = Index(TPSum, TNSum, FPSum, FNSum, C_Sum_or, UC_Sum_or)
    OA, Precision, Recall, F1 = Indicators2.ObjectExtract_indicators()
    IoU, c_IoU, uc_IoU = Indicators2.IOU_indicator()
    return OA, IoU

if __name__ == '__main__':
    # 选择设备，有cuda用cuda，没有就用cuda
    device ='cuda'# torch.device('cuda' if torch.cuda.is_available() else 'cuda')
    # 加载网络，图片单通道3，分类1(目标)
    net = UNet(n_channels=6,n_classes=1)
    # 待测试的代码
    # 将网络拷贝到device中
    # net=BiSRNet()
    net.to(device=device)
    # 指定训练集地址，开始训练
    data_path="./data/"+args.dataset+"/train"
    start_time = time.time()
    train_net(net, device, data_path)
    end_time = time.time()

    print('method1所用时间：', end_time - start_time)
