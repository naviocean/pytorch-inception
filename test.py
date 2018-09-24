from __future__ import print_function
import os
import os.path
import time
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
from meter import AverageMeter
from logger import Logger
from dataset import MyDataset
from models.inceptionv4 import inceptionv4
from utils import check_gpu, accuracy
from visualize import Visualizer
from torchvision.transforms import *
# os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1"

class Testing(object):
    def __init__(self, name_list, num_classes=400, **kwargs):
        self.__dict__.update(kwargs)

        self.num_classes = num_classes
        self.name_list = name_list

        # Set best precision = 0
        self.best_prec1 = 0
        # init start epoch = 0
        self.start_epoch = 0

        self.checkDataFolder()

        self.loading_model()

        self.test_loader = self.loading_data()

        # run
        self.process()

    def checkDataFolder(self):
        try:
            os.stat('./' + self.model_type + '_' + self.data_set)
        except:
            os.mkdir('./' + self.model_type + '_' + self.data_set)
        self.data_folder = './' + self.model_type + '_' + self.data_set

    # Loading P3D model
    def loading_model(self):

        print('Loading %s model' % (self.model_type))
        pretrained = None
        if self.pretrained:
            pretrained = 'imagenet'

        if self.model_type == 'inceptionv4':
            self.model = inceptionv4(num_classes=1000, pretrained=pretrained)
            if self.pretrained:
                num_ftrs = self.model.last_linear.in_features
                self.model.last_linear = nn.Linear(num_ftrs, self.num_classes)
                # free all layers:
                for i, param in self.model.named_parameters():
                    param.requires_grad = False
                # unfreeze last layers:
                ct = []
                for name, child in self.model.features.named_children():
                    if "4" in ct:
                        for param in child.parameters():
                            param.requires_grad = True
                    ct.append(name)


            else:
                num_ftrs = self.model.last_linear.in_features
                self.model.last_linear = nn.Linear(num_ftrs, self.num_classes)
        else:
            print('no model')
            exit()


        # Check gpu and run parallel
        if check_gpu() > 0:
            self.model = torch.nn.DataParallel(self.model).cuda()
            # self.model.cuda()

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss()
        if check_gpu() > 0:
            self.criterion = nn.CrossEntropyLoss().cuda()

        params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.optimizer = optim.SGD(params=params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)

        file = os.path.join(self.data_folder, 'model_best.pth.tar')
        if os.path.isfile(file):
            print("=> loading checkpoint '{}'".format('model_best.pth.tar'))

            checkpoint = torch.load(file)
            self.start_epoch = checkpoint['epoch']
            self.best_prec1 = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model")
        else:
            print("=> no model best found at ")
            exit()

        cudnn.benchmark = True

    # Loading data
    def loading_data(self):
        # normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        normalize = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        if self.tencrop:
            val_transformations = Compose([
                Resize(341),
                TenCrop(299),
                Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])),
                Lambda(
                    lambda normal: torch.stack([normalize(nor) for nor in normal]))

            ])
        else:
            val_transformations = Compose([
                Resize(341),
                CenterCrop(299),
                ToTensor(),
                normalize
            ])




        test_dataset = MyDataset(
            self.data,
            name_list=self.name_list,
            data_folder="test",
            version="1",
            transform=val_transformations,
        )

        test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            pin_memory=False)

        return test_loader

    # Test
    def process(self):
        acc = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        losses = AverageMeter()
        log_file = os.path.join(self.data_folder, 'test.log')
        logger = Logger('test', log_file)
        # switch to evaluate mode
        self.model.eval()

        start_time = time.clock()
        print("Begin testing")
        predicted, probs = [], []
        for i, (images, labels) in enumerate(self.test_loader):

            if check_gpu() > 0:
                images = images.cuda(async=True)
                labels = labels.cuda(async=True)
            images = torch.autograd.Variable(images)
            labels = torch.autograd.Variable(labels)

            if self.tencrop:
                # Due to ten-cropping, input batch is a 5D Tensor
                batch_size, number_of_crops, number_of_channels, height, width = images.size()

                # Fuse batch size and crops
                images = images.view(-1, number_of_channels, height, width)

                # Compute model output
                output_batch_crops = self.model(images)

                # Average predictions for each set of crops
                output_batch = output_batch_crops.view(batch_size, number_of_crops, -1).mean(1)
                label_repeated = labels.repeat(10, 1).transpose(1, 0).contiguous().view(-1, 1).squeeze()
                loss = self.criterion(output_batch_crops, label_repeated)
            else:
                output_batch = self.model(images)
                loss = self.criterion(output_batch, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output_batch.data, labels, topk=(1, 5))
        #     print(prec1, prec5)
            losses.update(loss.item(), images.size(0))
            acc.update(prec1.item(), images.size(0))
            top1.update(prec1.item(), images.size(0))
            top5.update(prec5.item(), images.size(0))

            if i % self.print_freq == 0:
                print('TestVal: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(self.test_loader), loss=losses, top1=top1, top5=top5))

        print(
            ' * Accuracy {acc.avg:.3f}  Acc@5 {top5.avg:.3f} Loss {loss.avg:.3f}'.format(acc=acc, top5=top5,
                                                                                         loss=losses))

        end_time = time.clock()
        print("Total testing time %.2gs" % (end_time - start_time))
        logger.info("Total testing time %.2gs" % (end_time - start_time))
        logger.info(
            ' * Accuracy {acc.avg:.3f} Loss {loss.avg:.3f}'.format(acc=acc, top5=top5,
                                                                                         loss=losses))




