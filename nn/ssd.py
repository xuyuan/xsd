import torch
from torch.nn import Flatten, Dropout, Linear, AdaptiveMaxPool2d, BCEWithLogitsLoss, Sequential
from detnet.nn.ssd import SingleShotDetector, SSDLoss
from .attention import DANetHead
from .oc_net import BaseOC


class Loss(SSDLoss):
    def __init__(self, priors, variance, num_classes, class_activation, args):
        super(Loss, self).__init__(priors, variance, num_classes, class_activation, args)
        self.classifier_loss = BCEWithLogitsLoss()

    def forward(self, outputs, targets):
        losses = super(Loss, self).forward(outputs, targets)

        priors_pos = targets.get('priors_pos', None)  # [N, L]
        num_pos = priors_pos.long().sum(1, keepdim=True)  # [N, 1]
        classifier_target = (num_pos > 0).float()
        losses['classifier'] = self.classifier_loss(outputs[4], classifier_target)

        return losses


class SingleShotDetectorWithClassifier(SingleShotDetector):
    def __init__(self, version, classnames, freeze_pretrained=0, frozen_bn=False, pretrained=False, attention=False,
                 oc_net=False, cls_add=False):

        basenet = {'efficientdet-d1': 'efficientnet-b1',
                   'efficientdet-d2': 'efficientnet-b2',
                   'efficientdet-d3': 'efficientnet-b3',
                   'efficientdet-d4': 'efficientnet-b4',
                   'efficientdet-d5': 'efficientnet-b5',}
        basenet = basenet[version]

        super(SingleShotDetectorWithClassifier, self).__init__(classnames=classnames, basenet=basenet, version=version,
                                                               pretrained=pretrained, frozen_bn=frozen_bn)
        classifier_in_channels = self.backbone[-1].out_channels

        classifier = [DANetHead(classifier_in_channels, classifier_in_channels) if attention else None,
                      BaseOC(classifier_in_channels, classifier_in_channels, classifier_in_channels, classifier_in_channels) if oc_net else None,
                      AdaptiveMaxPool2d(1),
                      Flatten(),
                      Dropout(p=0.2, inplace=True),
                      Linear(classifier_in_channels, 1)
                      ]
        classifier = [m for m in classifier if m]
        self.classifier = Sequential(*classifier)
        self.set_pretrained_frozen(freeze_pretrained)
        self.cls_add = cls_add

    def unpretrained_forward(self, x, activations, image_size):
        """untrained network"""
        if activations:
            x = activations[-1]

        for i in range(self.num_of_pretrained_layers, len(self.backbone)):
            layer = self.backbone[i]
            x = layer(x)
            activations.append(x)

        cls = self.classifier(x)

        if self.pfn:
            activations = self.pfn(activations)

        if self.feature_layer_index is None:
            # take last layers
            features = activations[-self.priorbox.n_layer:]
        else:
            # remap feature layers
            features = [activations[i] for i in self.feature_layer_index]

        if self.global_features:
            gf = self.layers_global_features(x)
            for k, x in enumerate(features):
                gfx = gf.expand(-1, -1, x.size(2), x.size(3))
                features[k] = torch.cat((x, gfx), dim=1)

        if self.mask_output:
            mask = self.mask_head(features[0])
        else:
            mask = None

        if self.elevation_output:
            elevation = self.elevation_head(features[0])
        else:
            elevation = None

        # apply multibox head to source layers
        for k, layer in enumerate(self.feature_layers):
            features[k] = layer(features[k])

        if self.dropout > 0 and self.training:
            features = [torch.nn.functional.dropout2d(i, p=self.dropout, training=self.training) for i in features]

        loc = self.loc(features)
        conf = self.conf(features)

        #self.feature_maps = conf  # for receptivefield.pytorch

        if not self.training and not self.trace_mode:
            self.priorbox((image_size, loc))

        num_points_per_box = self.priorbox.num_points_per_box()
        if self.implementation == 0:
            # LxN(AC)HW -> LxNHW(AC)
            loc = [l.permute(0, 2, 3, 1).contiguous() for l in loc]
            conf = [c.permute(0, 2, 3, 1).contiguous() for c in conf]

            # --> LxN(HWA)C --> N(LHWA)C
            loc = torch.cat([o.view(o.size(0), -1, num_points_per_box) for o in loc], 1)
            conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)

            # --> NC(LHWA)
            loc = loc.transpose(1, 2)
            conf = conf.transpose(1, 2)
        else:
            # LxN(CA)HW --> LxNC(AHW) --> NC(LAHW)
            loc = torch.cat([o.view(o.size(0), num_points_per_box, -1) for o in loc], dim=-1)
            conf = torch.cat([o.view(o.size(0), self.num_classes, -1) for o in conf], dim=-1)

        if self.cls_add:
            conf += cls.unsqueeze(2)
        else:
            conf *= torch.sigmoid(cls.unsqueeze(2))

        output = [loc, conf]
        if not self.trace_mode:
            output += [mask, elevation]
        else:
            output += [o for o in [mask, elevation] if o is not None]

        output.append(cls)
        return output

    def criterion(self, args):
        class_activation = self.cfg.get('class_activation', 'softmax')
        return Loss(self.priorbox.priors, self.cfg['variance'], self.num_classes, class_activation, args)