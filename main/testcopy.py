import os
import math
import argparse
from tkinter import MITER
from matplotlib.pyplot import axis

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torch.utils.data import DataLoader,random_split
import scipy.io as sio
from meet.models.vit import meet_base_patch16 as create_model
#from my_dataset import MyDataSet
#from timesformer.models.vit import TimeSformer
# from vit_model import vit_base_patch16_224_in21k as create_model
from utils import train_one_epoch, evaluate
from eegUtils import *

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

seed = 123
np.random.seed(seed)
torch.manual_seed(seed) #CPU随机种子确定
torch.cuda.manual_seed(seed) #GPU随机种子确定
torch.cuda.manual_seed_all(seed) #所有的GPU设置种子

# CHB-MIT 测试实验
def CHB_MIT_test_main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")
    
    # 创建任务对应的weights文件夹
    if os.path.exists("./weights/CHB-MIT/" + args.task) is False:
        os.makedirs("./weights/CHB-MIT/" + args.task)

    # 存储summary结果
    tb_writer = SummaryWriter("./Summary/CHB-MIT/" + args.task + "/")

    #Test_data = sio.loadmat("./dataset/CHB-MIT/convertImage/chb" + str(10) + "_1time-step_32@32.mat")
    Test_data = sio.loadmat(r"C:\Users\xiang.shen\Desktop\MEET-main/dataset/CHB-MIT/convertImage/chbtest.mat")
    Test_Images = Test_data["img"]  # [132,6,5,32,32]
    Test_Images = np.transpose(Test_Images, (0,2,1,3,4))
    Test_Label = (Test_data['label']).astype(int)
    Test_Label = np.reshape(Test_Label, (-1,1))[:,0]

    # Test Set
    Test_EEG = EEGImagesDataset(label=Test_Label, image=Test_Images)
    lengths = [int(len(Test_EEG)*1.0), int(len(Test_EEG)*0)]
    if sum(lengths) != len(Test_EEG):
            lengths[0] = lengths[0] + 1
    # Test, _ = random_split(Test_EEG, lengths)
    Test = Test_EEG

    batch_size = args.batch_size
    Testloader = DataLoader(Test, batch_size=batch_size,shuffle=False)
    
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    '''
    model = TimeSformer(img_size=32,
            patch_size=8,  
            num_classes=2, 
            num_frames=1, 
            attention_type='divided_space_time',  
            pretrained_model=''
    ).to(device)
    '''
    # model = create_model(num_classes=4, has_logits=False).to(device)
    model = create_model(num_classes=2).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 删除不需要的权重
        del_keys = ['head.weight', 'head.bias'] if False \
            else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        # for k in del_keys:
        #     del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head, pre_logits外，其他权重全部冻结
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.
    acc_log = []
    for epoch in range(args.epochs):
        # validate
        val_loss, val_acc, prediction = evaluate(model=model,
                                    data_loader=Testloader,
                                    device=device,
                                    epoch=epoch)

        tags = ["val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], val_loss, epoch)
        tb_writer.add_scalar(tags[1], val_acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        acc_log.append([val_loss, val_acc, optimizer.param_groups[0]["lr"]])

        # 保存最好的权重参数
        '''
        if best_acc < val_acc:
            torch.save(model.state_dict(), r"C:/Users/xiang.shen/Desktop/MEET-main/weights/CHB/" + args.task + "/best_model.pth")
            best_acc = val_acc
        '''

        # 结果输出到mat文件中
        result_file = os.path.join(r"C:\Users\xiang.shen\Desktop\MEET-main/Results/CHB/", 'Result_%s.mat'%args.task)
        Acc_Log = np.array(acc_log)
        sio.savemat(result_file, {'Acclog': Acc_Log.astype(np.double)})
        label_file = os.path.join(r"C:\Users\xiang.shen\Desktop\MEET-main/Results/CHB/", 'Label_%s.t'%args.task)
        torch.save(prediction, label_file)

        print(prediction[0])
        return prediction[0].numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.001)      # 原：0.001
    parser.add_argument('--lrf', type=float, default=0.01)

    # 执行任务类型
    # parser.add_argument('--task', default="WM_S1-15_1time_1000epochs_12depth_16heads_16patch_hhh")
    parser.add_argument('--task', default="CHB_S1")

    # 数据集所在根目录
    # http://download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default=r"E:\Data\Datasets\CV\flower_photos")
    parser.add_argument('--model-name', default='', help='create model name')

    # 预训练权重路径，如果不想载入就设置为空字符
    # parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
    #                     help='initial weights path')
    parser.add_argument('--weights', type=str, default=r"C:\Users\xiang.shen\Desktop\MEET-main\weights\CHB\CHB_S1\best_model.pth",
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)     # 原：冻结=True
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    CHB_MIT_test_main(opt)
