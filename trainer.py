import torch
from torch import optim
import torch.nn as nn
from models import FCN, unet, pspnet, dfn
from datasets.voc import to_rgb
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import time
from datetime import timedelta
import visdom
import cv2

try:
    import nsml
    from nsml import Visdom
    USE_NSML = True
    print('NSML imported')
except ImportError:
    print('Cannot Import NSML. Use local GPU')
    USE_NSML = False

cudnn.benchmark = True # For fast speed

class Trainer:
    def __init__(self, train_data_loader, val_data_loader, config):
        self.cfg = config
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.build_model()

    def denorm(self, x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

    def reset_grad(self):
        self.optim.zero_grad()

    def save_model(self, n_iter):
        torch.save(self.model.state_dict(), 'CvcModel.pth')
        """checkpiont = {
            'n_iters' : n_iter + 1,
            'm_state_dict' : self.model.state_dict(),
            'optim' : self.optim.state_dict()
            }
        torch.save({
                    'n_iters' : n_iter + 1,
                    'm_state_dict' : self.model.state_dict(),
                    'optim' : self.optim.state_dict()
                    }, 'cvc_net_10.pkl')"""


    def pixel_acc(self):
        """ Calculate accuracy of pixel predictions """
        pass

    def mean_IU(self):
        """ Calculate mean Intersection over Union
        x1 = Reframe[0]
        y1 = Reframe[1]
        width1 = Reframe[2]-Reframe[0]
        height1 = Reframe[3]-Reframe[1]
       
        x2 = GTframe[0]
        y2 = GTframe[1]
        width1 = GTframe[2]-GTframe[0]
        height1 = GTframe[3]-GTframe[1]

        endx = max(x1+width1,x2+width2)
        startx = min(x1,x2)
        width = width1+width2-(endx-startx)

        endy = max(y1+height1,ye+height2)
        starty = min(y1,y2)
        height = height+height2-(endy-starty)

        if width <= 0 or height <= 0:
            ratio = 0
        else:
            Area = width*height
            Area1 = width1*height1
            Area2 = width2*height2
            ratio = Area*1./(Area1+Area2-Area)
        return ratio,Reframe,GTframe"""
        pass

    def build_model(self):
        if self.cfg.model == 'unet':
            self.model = unet.UNet(num_classes=2, in_dim=3, conv_dim=64)
        elif self.cfg.model == 'fcn8':
            self.model = fcn.FCN8(num_classes=2)
        elif self.cfg.model == 'pspnet_avg':
            self.model = pspnet.PSPNet(num_classes=2, pool_type='avg')
        elif self.cfg.model == 'pspnet_max':
            self.model = pspnet.PSPNet(num_classes=2, pool_type='max')
        elif self.cfg.model == 'dfnet':
            self.model = dfn.SmoothNet(num_classes=2,
                                       h_image_size=self.cfg.h_image_size,
                                       w_image_size=self.cfg.w_image_size)
        self.optim = optim.Adam(self.model.parameters(),
                                lr=self.cfg.lr,
                                betas=[self.cfg.beta1, self.cfg.beta2])
        # Poly learning rate policy
        lr_lambda = lambda n_iter: (1 - n_iter/self.cfg.n_iters)**self.cfg.lr_exp
        self.scheduler = LambdaLR(self.optim, lr_lambda=lr_lambda)
        #weight = torch.tensor([0.04,0.07])
        #criterion = nn.CrossEntropyLoss(weight=weight)
        self.c_loss = nn.CrossEntropyLoss().to(self.device)
        self.softmax = nn.Softmax(dim=1).to(self.device) # channel-wise softmax

        self.n_gpu = torch.cuda.device_count()
        if self.n_gpu > 1:
            print('Use data parallel model(# gpu: {})'.format(self.n_gpu))
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        if USE_NSML:
            self.viz = Visdom(visdom=visdom)

    def train_val(self):
        # Compute epoch's step size
        iters_per_epoch = len(self.train_data_loader.dataset) // self.cfg.train_batch_size
        if len(self.train_data_loader.dataset) % self.cfg.train_batch_size != 0:
            iters_per_epoch += 1
        epoch = 1

        torch.cuda.synchronize()  # parallel mode
        self.model.train()

        train_start_time = time.time()
        data_iter = iter(self.train_data_loader)
        for n_iter in range(self.cfg.n_iters):
            self.scheduler.step()
            try:
                input, target = next(data_iter)
            except:
                data_iter = iter(self.train_data_loader)
                input, target = next(data_iter)

            input_var = input.clone().to(self.device)
            target_var = target.to(self.device)
            #output = self.model(input_var)

            #output = output.view(output.size(0), output.size(1), -1)
            #print('output=', output)
            #target_var = target_var.view(target_var.size(0), -1)
            #loss = self.c_loss(output, target_var)

            #loss = criterion(output, target_var)
            self.reset_grad()
            output = self.model(input_var)
            loss = self.c_loss(output, target_var)
            loss.backward()
            self.optim.step()
            # print('Done')

            output_label = torch.argmax(output, dim=1)
            #print('output_label=',output_label[1,:,:])
            #input shape [4,256,256]
            acc = 0.0
            #total = 4*256*256
            #print('output_label[0,0,0]=',output_label[0,0,0])
            #print('target_var[0,0,0]=',target_var[0,0,0])
            #shape 4,256,156
            """for k in range(4):
                    for i in range(256):
                        for j in range(256):
                            if output_label[k,i,j] == target_var[k,i,j]:
                                acc+=1
            acc = acc/total"""
            #print('The meanIoU is', acc, '%')


            if (n_iter + 1) % self.cfg.log_step == 0:
                seconds = time.time() - train_start_time
                elapsed = str(timedelta(seconds=seconds))
                print('Iteration : [{iter}/{iters}]\t'
                      'Time : {time}\t'
                      'Loss : {loss:.4f}\t'
                      'Mean : {meanIoU}\t'.format(
                      iter=n_iter+1, iters=self.cfg.n_iters,
                      time=elapsed, loss=loss.item(), meanIoU=acc))
                try:
                    nsml.report(
                            train__loss=loss.item(),
                            step=n_iter+1)
                except ImportError:
                    pass

            #ori_pic = self.denorm(input_var[0:4])
            #ori_pic = input_var.cpu().numpy()
            #ori_pic = (ori_pic*255).astype('uint8')
            #ori_pic = ori_pic.transpose(0, 2, 3, 1)
            ori_pic = transforms.ToPILImage()(input_var).convert('RGB')
            gt_mask = to_rgb(target_var[0:4])
            model_mask = to_rgb(output_label[0:4].cpu())
            for k in range (4):
                ori_pic.save('./pict/ori_pic'+str(n_iter)+'_'+str(k)+'.jpg')
                #cv2.imwrite('./pict/ori_pic'+str(n_iter)+'_'+str(k)+'.jpg',ori_pic[k])
                cv2.imwrite('./pict/gt_mask'+str(n_iter)+'_'+str(k)+'.jpg',gt_mask[k])
                cv2.imwrite('./pict/model_mask'+str(n_iter)+'_'+str(k)+'.jpg',model_mask[k])

            if (n_iter + 1) % iters_per_epoch == 0:
                self.validate(epoch)
                epoch += 1
        self.save_model(n_iter)
        print("End training")

    def validate(self, epoch):

        self.model.eval()
        val_start_time = time.time()
        data_iter = iter(self.val_data_loader)
        max_iter = len(self.val_data_loader)
        # for n_iter in range(max_iter): #FIXME
        n_iter =  0
        input, target = next(data_iter)

        input_var = input.clone().to(self.device)
        target_var = target.to(self.device)

        output = self.model(input_var)
        #_output = output.clone()
        # output = output.view(output.size(0), output.size(1), -1)
        # target_var = target_var.view(target_var.size(0), -1)
        loss = self.c_loss(output, target_var)

        #output_label = torch.argmax(_output, dim=1)

        if (n_iter + 1) % self.cfg.log_step == 0:
            seconds = time.time() - val_start_time
            elapsed = str(timedelta(seconds=seconds))
            print('### Validation\t'
                  'Iteration : [{iter}/{iters}]\t'
                  'Time : {time:}\t'
                  'Loss : {loss:.4f}\t'.format(
                  iter=n_iter+1, iters=max_iter,
                  time=elapsed, loss=loss.item()))
            try:
                nsml.report(
                        val__loss=loss.item(),
                        step=epoch)
            except ImportError:
                pass

        """if USE_NSML:
            ori_pic = self.denorm(input_var[0:4])
            self.viz.images(ori_pic, opts=dict(title='Original_' + str(epoch)))
            gt_mask = to_rgb(target_var[0:4])
            self.viz.images(gt_mask, opts=dict(title='GT_mask_' + str(epoch)))
            model_mask = to_rgb(output_label[0:4].cpu())
            self.viz.images(model_mask, opts=dict(title='Model_mask_' + str(epoch)))"""
       
