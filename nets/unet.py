import torch
import torch.nn as nn

from nets.resnet import resnet101
from nets.vgg import VGG16



class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        #self.up     = nn.MaxUnpool2d((2,2),stride=(2,2))
        self.relu   = nn.ReLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, inputs2], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


'''class DUpsampling(nn.Module):
    def __init__(self, inplanes, scale, num_class=21, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, num_class * scale * scale, kernel_size=1, padding = pad,bias=False)
        self.scale = scale
    
    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1) 

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, W * self.scale, int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((N, W * self.scale, H * self.scale, int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)
        
        return x'''


'''class DUC(nn.Module):
    def __init__(self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(inplanes, planes, kernel_size=3,
                              padding=1)
        self.bn = nn.BatchNorm2d(planes)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x'''

class D_conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation, padding=dilation)
        super().__init__(conv2d)


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1,):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class Unet(nn.Module):
    
    def __init__(self, num_classes = 3, pretrained = False, backbone = 'vgg'):
        super(Unet, self).__init__()
        if backbone == 'vgg':
            self.vgg    = VGG16(pretrained = pretrained)
            in_filters  = [192, 384, 768, 1024]
        elif backbone == "resnet101":
            self.resnet = resnet101(pretrained = pretrained)
            in_filters  = [192, 512, 1024, 3072]
        else:
            raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
        out_filters = [64, 128, 256, 512]

        #self.num_classes = 3

        self.up = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.dropout = nn.Dropout(0.5)

        
        self.encoder_layer5 =  nn.TransformerEncoderLayer(d_model=32, nhead=8)
        self.transformer_encoder5 = nn.TransformerEncoder(self.encoder_layer5, num_layers=6)
        
        
        self.d_conv1 = D_conv(512, 512,kernel_size=3,dilation=1)
        self.d_conv2 = D_conv(512, 512,kernel_size=3,dilation=2)
        self.d_conv5 = D_conv(512, 512,kernel_size=3,dilation=5)
        self.d_conv1 = D_conv(512, 512,kernel_size=3,dilation=1)
        self.d_conv2 = D_conv(512, 512,kernel_size=3,dilation=2)
        self.d_conv5 = D_conv(512, 512,kernel_size=3,dilation=5)
        

        '''self.dupsample = DUpsampling(64, 4, num_class=3)'''

        '''self.duc1 = DUC(512, 2048)
        self.duc2 = DUC(512, 1024)
        self.duc3 = DUC(256, 1024)
        self.duc4 = DUC(128, 512)
        self.duc5 = DUC(64, 256)'''

        '''#self.transformer = nn.Conv2d(320, 128, kernel_size=1)
        self.out5 = self._classifier(32)'''



        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])

        if backbone == 'resnet101':
            self.up_conv = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor = 2), 
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
                nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
                nn.ReLU(),
            )
        else:
            self.up_conv = None

        #self.final = nn.Conv2d(out_filters[0], num_classes, 1)

        self.final = SegmentationHead(out_filters[0], num_classes, kernel_size=3)
        self.backbone = backbone


    '''def _classifier(self, inplanes):
        if inplanes == 32:
            return nn.Sequential(
                nn.Conv2d(inplanes, self.num_classes, 1),
                nn.Conv2d(self.num_classes, self.num_classes,
                          kernel_size=3, padding=1)
            )
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes/2, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes/2, momentum=.95),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes/2, self.num_classes, 1),
        )'''


    def forward(self, inputs):
        if self.backbone == "vgg":
            [feat1, feat2, feat3, feat4, feat5] = self.vgg.forward(inputs)
        elif self.backbone == "resnet101":
            [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)


        
        
        '''feat1 = feat1.squeeze()
        feat1 = self.transformer_encoder1(feat1)
        feat1 = feat1.unsqueeze(0)'''
        '''feat5 = feat5.squeeze()
        #print(feat5.shape)
        feat5 = self.transformer_encoder5(feat5)
        feat5 = feat5.unsqueeze(0)
        #print("trans",trans_feat.shape)'''
        
        
        
        dil_feat = self.d_conv1(feat5)
        dil_feat = self.d_conv2(dil_feat)
        dil_feat = self.d_conv5(dil_feat)
        dil_feat = self.d_conv1(dil_feat)
        dil_feat = self.d_conv2(dil_feat)
        dil_feat = self.d_conv5(dil_feat)

        dil_feat = self.dropout(dil_feat)


        #print("0",dil_feat.shape)

        #dil_feat = dil_feat.squeeze()
        #a = torch.rand(256,512,512).cuda()
        #trans_feat1 = self.transformer_decoder(dil_feat, a)
        #trans_feat1 = trans_feat1.unsqueeze(0)



        feat = self.up(dil_feat)
        #print("dil",dil_feat.shape)
        up04 = self.up_concat4(feat4, feat)

        up04 = self.up(up04)
        #print("4",up04.shape)
        up03 = self.up_concat3(feat3, up04)
        #print('3',up3.shape)
        #feat2 = trans_feat

        up03 = self.up(up03)
        #print('3',up03.shape)
        up02 = self.up_concat2(feat2, up03)
        #print("2",up2.shape)

        up02 = self.up(up02)

        up01 = self.up_concat1(feat1, up02)
        #print('1',up1.shape)

        '''up4 = feat4 + self.duc1(feat5)
        up3 = feat3 + self.duc2(up4)
        up2 = feat2 + self.duc3(up3)
        up1 = feat1 + self.duc4(up2)
        up1 = self.duc5(up1)
        out = self.out5(up1)
        #print("out",out.shape)'''
        




        if self.up_conv != None:
            up1 = self.up_conv(up1)
        
        
        final = self.final(up01)
        #print('out',out.shape)
        #print('final',final.shape)



        return final

    def freeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = False
        elif self.backbone == "resnet101":
            for param in self.resnet.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        if self.backbone == "vgg":
            for param in self.vgg.parameters():
                param.requires_grad = True
        elif self.backbone == "resnet101":
            for param in self.resnet.parameters():
                param.requires_grad = True
