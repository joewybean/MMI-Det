# YOLOv5 YOLO-specific modules
from utils.torch_utils import time_synchronized, fuse_conv_and_bn, model_info, scale_img, initialize_weights, \
    select_device, copy_attr
from utils.general import make_divisible, check_file, set_logging
from utils.autoanchor import check_anchor_order
from models.experimental import *
from models.common import *
from torchsummary import summary
import argparse
import logging
import sys
from copy import deepcopy
from pathlib import Path

from thop import profile
from thop import clever_format

# to run '$ python *.py' files in subdirectories
sys.path.append(Path(__file__).parent.parent.absolute().__str__())
logger = logging.getLogger(__name__)


try:
    import thop  # for FLOPS computation
except ImportError:
    thop = None


class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(
            self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1)
                               for x in ch)  # output conv

    def forward(self, x):
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(
                0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 +
                               self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * \
                    self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()



class Model(nn.Module):

    # model, input channels, number of classes
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):
        super(Model, self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict

        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.safe_load(f)  # model dict
            # print("YAML")
            # print(self.yaml)

        ################################
        self.contrastive_loss_func = RecContrastiveLoss(margin=1.0)

        ################################
        # Define model
        self.Enhance = AdaptiveModule3(
            in_channels=int(ch), out_channels=int(ch))
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels
        if nc and nc != self.yaml['nc']:
            logger.info(
                f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        if anchors:
            logger.info(
                f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        self.model, self.save = parse_model(
            deepcopy(self.yaml), ch=[ch])  # model, savelist
        # print(self.model)
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names
        # logger.info([x.shape for x in self.forward(torch.zeros(1, ch, 64, 64))])

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        # print(m)
        GPT_layer = self.model[6]  # the first GPT
        if isinstance(GPT_layer, GPT1):
            print("GPT1_layer is exit !!!")
        if isinstance(GPT_layer, GPT1_fourier):
            print("GPT1_fourier is exit !!!")
        if isinstance(m, Detect):
            s = 256  # 2x min stride
            # print("1, ch, s, s", 1, ch, s, s)
            # m.stride = torch.tensor([s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s), torch.zeros(1, ch, s, s))])  # forward
            m.stride = torch.Tensor([8.0, 16.0, 32.0])
            # print("m.stride", m.stride)
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()  # only run once
            # logger.info('Strides: %s' % m.stride.tolist())

        # Init weights, biases
        initialize_weights(self)
        self.info()
        logger.info('')

    def forward(self, x, x2, augment=False, profile=False):
        if augment:
            img_size = x.shape[-2:]  # height, width
            s = [1, 0.83, 0.67]  # scales
            f = [None, 3, None]  # flips (2-ud, 3-lr)
            y = []  # outputs
            for si, fi in zip(s, f):
                xi = scale_img(x.flip(fi) if fi else x, si,
                               gs=int(self.stride.max()))
                yi = self.forward_once(xi)[0]  # forward
                # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
                yi[..., :4] /= si  # de-scale
                if fi == 2:
                    yi[..., 1] = img_size[0] - yi[..., 1]  # de-flip ud
                elif fi == 3:
                    yi[..., 0] = img_size[1] - yi[..., 0]  # de-flip lr
                y.append(yi)
            return torch.cat(y, 1), None  # augmented inference, train
        else:
            # single-scale inference, train
            return self.forward_once(x, x2, profile)

    def forward_once(self, x, x2, profile=False):
        """

        :param x:          RGB Inputs
        :param x2:         IR  Inputs
        :param profile:
        :return:
        """
        a1 = torch.tensor(
            0.1, device=x.device)  # ContrastiveValue torch.zeros(0, device=x.device) # 1.0 # torch.zeros(0, device=x.device) #  1.0 0.0 0.1 1.0
        # SSIMloss torch.zeros(0, device=x.device) # torch.zeros(0, device=x.device) # 2.0 0.0 0.2
        a2 = torch.tensor(1.0, device=x.device)

        # PTLoss 1.0 torch.zeros(0, device=x.device) 0.06
        a3 = torch.tensor(0.06, device=x.device)
        # compute_EntropyLoss(self, rgb_features, ir_features) 0.03
        a4 = torch.tensor(0.03, device=x.device)

        # self.ContrastiveValue = self.compute_contrastive_loss(x, x2) #
        self.ContrastiveValue = torch.zeros(0, device=x.device)
        self.SSIMloss = torch.zeros(0, device=x.device)  # 0.0
        self.PTLoss = torch.zeros(0, device=x.device)
        self.Combine_loss = torch.zeros(0, device=x.device)
        self.Entropy_loss = torch.zeros(0, device=x.device)

        x = self.Enhance(x)  # CEM
        y, dt = [], []  # outputs
        i = 0
        for m in self.model:
            # print("Moudle", i)
            if m.f != -1:  # if not from previous layer
                if m.f != -4:
                    # print(m)
                    x = y[m.f] if isinstance(m.f, int) else [
                        x if j == -1 else y[j] for j in m.f]  # from earlier layers

            if profile:
                o = thop.profile(m, inputs=(x,), verbose=False)[
                    0] / 1E9 * 2 if thop else 0  # FLOPS
                t = time_synchronized()
                for _ in range(10):
                    _ = m(x)
                dt.append((time_synchronized() - t) * 100)
                if m == self.model[0]:
                    logger.info(
                        f"{'time (ms)':>10s} {'GFLOPS':>10s} {'params':>10s}  {'module'}")
                logger.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')

            if isinstance(m, GPT1):  # 通过模型前  GPT1
                GPT1_input1 = x[0]
                GPT1_input2 = x[1]
                self.ContrastiveValue = self.compute_contrastive_loss(
                    GPT1_input1, GPT1_input2)
                # print("GPT1 OUTPUT SHAPE:", x.shape)]
            if isinstance(m, GPT1_fourier):  # 通过模型前 GPT1_fourier GPT1 ####
                GPT1_input1 = x[0]
                GPT1_input2 = x[1]
                self.ContrastiveValue = self.compute_contrastive_loss(
                    GPT1_input1, GPT1_input2)
                # print("GPT1 OUTPUT SHAPE:", x.shape)]
            if m.f == -4:
                x = m(x2)
            else:
                if isinstance(m, GPT1):  # 通过模型后 GPT1_fourier GPT1
                    x[0], x[1], PTLoss = m(x)  # run
                    self.PTLoss = torch.tensor(PTLoss,  device=x[0].device)
                elif isinstance(m, GPT1_fourier):  # 通过模型后 GPT1_fourier GPT1
                    x[0], x[1], PTLoss = m(x)  # run
                    self.PTLoss = torch.tensor(PTLoss,  device=x[0].device)

                else:
                    x = m(x)  # run
            if isinstance(m, GPT1):  # 通过模型后 GPT1_fourier GPT1
                GPT1_output1 = x[0]
                GPT1_output2 = x[1]
                # GPT_output_avg =  (GPT1_output1 + GPT1_output2) / 2.0
                GPT_output_avg = torch.mean(torch.stack(
                    [GPT1_output1, GPT1_output2]), dim=0)
                # print("GPT1 OUTPUT SHAPE:", x.shape)
                # self.SSIMloss = self.compute_fusing_loss(GPT1_input1, GPT1_input2, GPT_output_avg, GPT_output_avg) # SSIM_Loss

                # print("ir_fea.shape:{}".format(ir_fea.shape))  # ir_fea.shape:torch.Size([4, 128, 8, 8])
            if isinstance(m, GPT1_fourier):  # 通过模型后 GPT1_fourier GPT1
                GPT1_output1 = x[0]
                GPT1_output2 = x[1]
                # GPT_output_avg =  (GPT1_output1 + GPT1_output2) / 2.0
                GPT_output_avg = torch.mean(torch.stack(
                    [GPT1_output1, GPT1_output2]), dim=0)
                # print("GPT1 OUTPUT SHAPE:", x.shape)
                self.SSIMloss = self.compute_fusing_loss2(
                    GPT1_input1, GPT1_input2, GPT_output_avg, GPT_output_avg)  # SSIM_Loss
                print("SSIM Loss:", self.SSIMloss)
                self.Entropy_loss = self.compute_EntropyLoss(
                    GPT1_input1, GPT1_input2, GPT_output_avg)

                # print("ir_fea.shape:{}".format(ir_fea.shape))  # ir_fea.shape:torch.Size([4, 128, 8, 8])
            # print("A * Loss: a1:{}, Contrast loss:{}, a2:{}, SSIM loss:{}, a3:{}, PT loss:{}, a4:{}, Entropy_loss:{}". format(a1 , self.ContrastiveValue, a2 , self.SSIMloss, a3 , self.PTLoss, a4 , self.Entropy_loss))
            y.append(x if m.i in self.save else None)  # save output
            # print(len(y))
            i += 1
        # self.Contast_SSIM_loss = self.ContrastiveValue + belta * self.SSIMloss
        Combine_loss = a1 * self.ContrastiveValue + a2 * \
            self.SSIMloss + a3 * self.PTLoss + a4 * self.Entropy_loss

        Combine_loss = self.SSIMloss

        self.Combine_loss = torch.tensor(Combine_loss, device=x[0].device)
        print("Inner Combine_loss:", self.Combine_loss)
        ###
        if profile:
            logger.info('%.1fms total' % sum(dt))
        return x, self.Combine_loss
        # if profile:
        #     logger.info('%.1fms total' % sum(dt))
        # return x

    # initialize biases into Detect(), cf is class frequency

    def _initialize_biases(self, cf=None):
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)
                                      ) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names',
                  'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def ContrastiveLoss(self, embeddings1, embeddings2, labels, margin=1.0):
        # margin = margin
        distances = F.normalize(embeddings1 - embeddings2, dim=1)
        # print("distance: ", distances)
        # mean_dis = torch.mean(distances)
        squared_dis = torch.square(distances)
        mean_dis = torch.mean(squared_dis)
        # print("mean distance:", mean_dis)
        # print("distance shape:", distances.shape)
        # print("labels shape:", labels.shape)

        # loss = (1 - torch.mean(labels)) * torch.pow(mean_dis, 2) + torch.mean(labels) * (torch.clamp(margin - torch.pow(mean_dis, 2), min=0.0))  # 负样本对的损失
        loss = (1 - torch.mean(labels)) * torch.exp(mean_dis) + \
            torch.mean(labels) * (torch.exp(mean_dis) - margin)  # 负样本对的损失

        # print("inter loss: ", loss)
        return loss

    def compute_contrastive_loss(self, rgb_features, ir_features):
        # # 获取RGB和IR图像的预测结果和目标
        # pred_rgb = rgb_features # 假设这是RGB图像的预测结果
        # pred_ir = ir_features # 假设这是IR图像的预测结果

        # 提取正样本对
        pros_rgb = rgb_features[0:-1:1]
        pros_ir = ir_features[0:-1:1]
        # 提取负样本对1
        neg_rgb = rgb_features[0:-1:1]  # RGB样本

        # 倒置多维数组
        # inverted_ir_features = list(zip(*ir_features))
        neg_ir = ir_features[1::1]  # IR样本

        # 提取负样本对2
        neg_rgb2 = rgb_features[1::1]  # RGB样本2

        neg_ir2 = ir_features[0:-1:1]  # IR样本2

        # 获取目标的索引
        # target_idx = targets[:, 0]  # 假设目标索引位于第一列

        # # 提取正样本对
        # pos_rgb = pred_rgb[target_idx]
        # pos_ir = pred_ir[target_idx]

        # # 提取负样本对
        # neg_rgb = pred_rgb[~target_idx]
        # neg_ir = pred_ir[~target_idx]

        # 计算对比学习损失
        pos_labels = torch.zeros(
            (pros_rgb.shape[0], pros_rgb.shape[2], pros_rgb.shape[3]), device=pros_rgb.device)
        neg_labels = torch.ones(
            (neg_rgb.shape[0], neg_rgb.shape[2], neg_rgb.shape[3]), device=neg_rgb.device)
        # print("label shape:", pos_labels.shape)
        # print("pros_rgb shape:", pros_rgb.shape)

        pos_loss = self.ContrastiveLoss(pros_rgb, pros_ir, pos_labels)
        neg_loss = self.ContrastiveLoss(neg_rgb, neg_ir, neg_labels)

        neg_loss2 = self.ContrastiveLoss(neg_rgb2, neg_ir2, neg_labels)

        # 取正样本和负样本损失的平均
        contrastive_loss = (pos_loss * 2 + neg_loss + neg_loss2) / 4.0
        # contrastive_loss = - torch.log( pos_loss / (pos_loss + neg_loss))

        return contrastive_loss

    def compute_EntropyLoss(self, original_rgb_features, original_ir_features, fused_image):
        # Reshape the fused image to match RGB feature dimensions
        original_rgb_features_float = original_rgb_features.float()
        original_ir_features_float = original_ir_features.float()
        fused_image_float = fused_image.float()

        # Reshape the fused image to match RGB feature dimensions
        # fused_image_reshaped = fused_image.view(original_rgb_features.shape)

        entropy_rgb = self.calculate_entropy(original_rgb_features_float)
        entropy_ir = self.calculate_entropy(original_ir_features_float)
        entropy_fused = self.calculate_entropy(fused_image_float)

        # Calculate the difference in entropy
        entropy_diff = (entropy_rgb + entropy_ir) - entropy_fused

        return entropy_diff

    def calculate_entropy(self, image):
        hist = torch.histc(image, bins=256, min=0, max=1)
        hist /= hist.sum()
        non_zero_elements = hist[hist > 0]
        entropy = -torch.sum(non_zero_elements * torch.log2(non_zero_elements))
        return entropy

    def compute_fusing_loss(self, rgb_features, ir_features, fuse_RGBfeature, fuse_IRfeature):
        # # 获取RGB和IR图像的预测结果和目标
        # pred_rgb = rgb_features # 假设这是RGB图像的预测结果
        # pred_ir = ir_features # 假设这是IR图像的预测结果

        loss_rgb = self.ssim_loss(rgb_features, fuse_RGBfeature)
        loss_ir = self.ssim_loss(ir_features, fuse_IRfeature)

        # Combine the losses
        ssim_fuse_loss = (loss_rgb + loss_ir) / 2.0
        # print("ssim fusing loss:", ssim_fuse_loss)
        return ssim_fuse_loss

    def compute_fusing_loss2(self, rgb_features, ir_features, fuse_RGBfeature, fuse_IRfeature):
        loss_rgb = self.ssim_loss(rgb_features, fuse_RGBfeature)
        loss_ir = self.ssim_loss(ir_features, fuse_IRfeature)

        # Weighted fusion loss
        alpha_rgb = 0.5  # Weight for RGB loss
        alpha_ir = 0.5   # Weight for IR loss
        weighted_ssim_fuse_loss = alpha_rgb * loss_rgb + alpha_ir * loss_ir

        # Contrast consistency term
        contrast_loss = torch.mean(
            torch.abs(torch.std(fuse_RGBfeature) - torch.std(fuse_IRfeature)))

        total_loss = weighted_ssim_fuse_loss + contrast_loss

        return total_loss

    def ssim_loss(self, img1, img2):
        # Calculate mean of img1 and img2
        # print("img1:", img1)
        # print("img2:", img2)

        # mu1 = img1.mean()
        # mu2 = img2.mean()
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)

        # Calculate variance of img1 and img2
        var1 = torch.mean(((img1 - mu1) ** 2))
        var2 = torch.mean(((img2 - mu2) ** 2))

        # Calculate covariance of img1 and img2
        cov12 = torch.mean((img1 - mu1) * (img2 - mu2))

        # Set constants for SSIM calculation
        c1 = 0.01 ** 2  # Small constant to avoid division by zero
        c2 = 0.03 ** 2  # Small constant to avoid division by zero

        # Calculate SSIM
        ssim = (2 * mu1 * mu2 + c1) * (2 * cov12 + c2) / \
            ((mu1 ** 2 + mu2 ** 2 + c1) * (var1 + var2 + c2))

        return 1 - ssim

    # initialize biases into Detect(), cf is class frequency
    def _initialize_biases(self, cf=None):
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)
                                      ) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _print_biases(self):
        m = self.model[-1]  # Detect() module
        for mi in m.m:  # from
            b = mi.bias.detach().view(m.na, -1).T  # conv.bias(255) to (3,85)
            logger.info(
                ('%6g Conv2d.bias:' + '%10.3g' * 6) % (mi.weight.shape[1], *b[:5].mean(1).tolist(), b[5:].mean()))

    # def _print_weights(self):
    #     for m in self.model.modules():
    #         if type(m) is Bottleneck:
    #             logger.info('%10.3g' % (m.w.detach().sigmoid() * 2))  # shortcut weights

    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        logger.info('Fusing layers... ')
        for m in self.model.modules():
            if type(m) is Conv and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.fuseforward  # update forward
        self.info()
        return self

    def nms(self, mode=True):  # add or remove NMS module
        present = type(self.model[-1]) is NMS  # last layer is NMS
        if mode and not present:
            logger.info('Adding NMS... ')
            m = NMS()  # module
            m.f = -1  # from
            m.i = self.model[-1].i + 1  # index
            self.model.add_module(name='%s' % m.i, module=m)  # add
            self.eval()
        elif not mode and present:
            logger.info('Removing NMS... ')
            self.model = self.model[:-1]  # remove
        return self

    def autoshape(self):  # add autoShape module
        logger.info('Adding autoShape... ')
        m = autoShape(self)  # wrap model
        copy_attr(m, self, include=('yaml', 'nc', 'hyp', 'names',
                  'stride'), exclude=())  # copy attributes
        return m

    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)


def parse_model(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' %
                ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors,
                                              list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # print("ch", ch)
    # from, number, module, args
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:

            if m is Focus:
                c1, c2 = 3, args[0]
                # print("focus c2", c2)
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)
                args = [c1, c2, *args[1:]]
            else:
                c1, c2 = ch[f], args[0]
                if c2 != no:  # if not output
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in [BottleneckCSP, C3, C3TR]:
                    args.insert(2, n)  # number of repeats
                    n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Add:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            args = [c2]
        elif m is Add2:
            # print("ch[f]", f, ch[f[0]])
            c2 = ch[f[0]]
            # print("Add2 arg", args[0])
            args = [c2, args[1]]
        elif m is GPT:
            c2 = ch[f[0]]
            args = [c2]

        elif m is GPT1:
            c2 = args[0]
            args = [c2]
        elif m is GPT1_fourier:
            c2 = args[0]
            args = [c2]

        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]
                           ) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        # attach index, 'from' index, type, number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' %
                    (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(
            f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        # if i == 4:
        #     ch = []
        ch.append(c2)
    # print(layers)
    return nn.Sequential(*layers), sorted(save)


def parse_model_rgb_ir(d, ch):  # model_dict, input_channels(3)
    logger.info('\n%3s%18s%3s%10s  %-40s%-30s' %
                ('', 'from', 'n', 'params', 'module', 'arguments'))
    anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
    na = (len(anchors[0]) // 2) if isinstance(anchors,
                                              list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # from, number, module, args
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
        m = eval(m) if isinstance(m, str) else m  # eval strings
        for j, a in enumerate(args):
            try:
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings
            except:
                pass

        n = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in [Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, DWConv, MixConv2d, Focus, CrossConv, BottleneckCSP,
                 C3, C3TR]:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, 8)

            args = [c1, c2, *args[1:]]
            if m in [BottleneckCSP, C3, C3TR]:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum([ch[x] for x in f])
        elif m is Detect:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*[m(*args) for _ in range(n)]
                           ) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum([x.numel() for x in m_.parameters()])  # number params
        # attach index, 'from' index, type, number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np
        logger.info('%3s%18s%3s%10.0f  %-40s%-30s' %
                    (i, f, n, np, t, args))  # print
        save.extend(x % i for x in ([f] if isinstance(
            f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)

    layers_rgb = layers[:4].copy()
    layer_ir = layers[:4].copy()
    rgb_stream = nn.Sequential(*layers_rgb)
    ir_stream = nn.Sequential(*layer_ir)

    # 以concat为界，分割模型
    my_layer = []
    for i in range(4, len(layers)):
        my_layer.append([layers[i]].copy())

    # print("My Layer")
    # print(len(my_layer))
    # for i in range(len(my_layer)):
    #     print(my_layer[i])
    # layer_4 = layers[4].copy()
    # layer_5 = layers[5].copy()
    # layers_rest = layers[4:].copy()
    # rest_net = nn.Sequential(*layers_rest)
    # print(rest_net)
    # print(" REST Net")
    # print(rest_net)

    # return nn.Sequential(*layers), sorted(save)
    return model, sorted(save)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./models/transformer/yolov5l_fusion_transformer_M3FD_fuse3_fourier.yaml',
                        help='model.yaml')  # yolov5l_fusion_transformer_M3FD_fuse3_fourier yolov5l_fusion_transformer_M3FD
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)
    print(device)

    model = Model(opt.cfg).to(device)
    input_rgb = torch.Tensor(8, 3, 640, 640).to(device)
    input_ir = torch.Tensor(8, 3, 640, 640).to(device)

    output, _ = model(input_rgb, input_ir)
    print("YOLO")
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    # print(output)
    # 使用 `thop.profile` 函数计算 FLOPS 和 Gparams
    macs, params = profile(model, inputs=(input_rgb, input_ir))

    # 使用 `thop.clever_format` 函数将结果格式化成易读的字符串
    flops, params = clever_format([macs, params], "%.3f")

    print(f"Gparams: {params}")
    print(f"FLOPS: {flops}")

    # # Create model
    # model =TwoStreamModel(opt.cfg).to(device)
    # print(model)
    # input_rgb = torch.Tensor(8, 3, 640, 640).to(device)
    # input_ir = torch.Tensor(8, 3, 640, 640).to(device)
    # output = model.model(input_rgb, input_ir)
    # print("YOLO Fusion")
    # print(output[0].shape)
    # print(output[1].shape)
    # print(output[2].shape)
    # print(output.shape)

    # print(model)
    # model.train()
    # torch.save(model, "yolov5s.pth")

    # Profile
    # img = torch.rand(8 if torch.cuda.is_available() else 1, 3, 320, 320).to(device)
    # y = model(img, profile=True)

    # Tensorboard (not working https://github.com/ultralytics/yolov5/issues/2898)
    # from torch.utils.tensorboard import SummaryWriter
    # tb_writer = SummaryWriter('.')
    # logger.info("Run 'tensorboard --logdir=models' to view tensorboard at http://localhost:6006/")
    # tb_writer.add_graph(torch.jit.trace(model, img, strict=False), [])  # add model graph
    # tb_writer.add_image('test', img[0], dataformats='CWH')  # add model to tensorboard
