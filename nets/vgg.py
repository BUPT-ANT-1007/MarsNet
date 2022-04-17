import torch
import torch.nn as nn
#from torchvision.models.utils import load_state_dict_from_url
#from nets.Transformer import TransformerModel

class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        #self.transformer = TransformerModel()
        self._initialize_weights()
        self.encoder_layer5 =  nn.TransformerEncoderLayer(d_model=32, nhead=4)
        self.transformer_encoder5 = nn.TransformerEncoder(self.encoder_layer5, num_layers=6)


    def forward(self, x):
        # x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        #print(x.shape)
        feat1 = self.features[  :4 ](x)
        #print('1',feat1.shape)
        '''feat1 = feat1.squeeze()
        feat1 = self.transformer_encoder1(feat1)
        feat1 = feat1.unsqueeze(0)'''

        feat2 = self.features[4 :9 ](feat1)
        #print('2',feat2.shape)
        '''feat2 = feat2.squeeze()
        feat2 = self.transformer_encoder2(feat2)
        feat2 = feat2.unsqueeze(0)'''

        feat3 = self.features[9 :16](feat2)
        '''feat3 = feat3.squeeze()
        feat3 = self.transformer_encoder3(feat3)
        feat3 = feat3.unsqueeze(0)'''
        #print('3',feat3.shape)
        feat4 = self.features[16:23](feat3)
        '''feat4 = feat4.squeeze()
        feat4 = self.transformer_encoder4(feat4)
        feat4 = feat4.unsqueeze(0)'''
        #print('4',feat4.shape)
        feat5 = self.features[23:-1](feat4)


        trans_feat = feat5
        trans_feat = trans_feat.squeeze()
        trans_feat= self.transformer_encoder5(trans_feat)
        trans_feat = trans_feat.unsqueeze(0)
        feat5 = feat5 + trans_feat
        #print('5',feat5.shape)
        #print(feat.shape,"0")
        
        #print(trans_feat.shape)

        #trans_feat = torch.tensor(trans_feat, dtype=torch.float32)
        #print(trans_feat.shape)

        #print(trans_feat.shape)
        return [feat1, feat2, feat3, feat4, feat5]#, trans_feat]
    

    '''def trans_layers(self,feat,d_models):
        encoder = self.encoder_layer(d_models,8)
        transformer_encoder = self.transformer_encoder(encoder,1)
        feat = feat.squeeze()
        trans_feat0 = transformer_encoder(feat)
        trans_feat = trans_feat0.unsqueeze(0)
        
        return trans_feat'''



    def _initialize_weights(self):
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False, in_channels = 3):
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
# 512,512,3 -> 512,512,64 -> 256,256,640 -> 256,256,128 -> 128,128,128 -> 128,128,256 -> 64,64,2560
# 64,64,512 -> 32,32,512 -> 32,32,512
cfgs = {
    'D': [64, 64, 'M', 128, 128,'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


def VGG16(pretrained, in_channels = 3, **kwargs):
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = in_channels), **kwargs)
    if pretrained:
        #state_dict = load_state_dict_from_url("https://download.pytorch.org/models/vgg16-397923af.pth", model_dir="./model_data")
        model.load_state_dict(torch.load('/home/lvweikun/vgg16-397923af.pth'), strict=False)
    
    del model.avgpool
    del model.classifier
    return model

if __name__ == "__main__":
    model = VGG(make_layers(cfgs["D"], batch_norm = False, in_channels = 3))
    a = model.features
    print(a)
    