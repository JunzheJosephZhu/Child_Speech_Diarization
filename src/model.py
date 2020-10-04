import torch
import json
import encoder
import backbone
import classifier
import os

def load(model, root, path):
    pkg = torch.load(os.path.join(root, path))
    model.load_state_dict(pkg["state_dict"])
    print("loaded pretrained %s at %s"%(type(model), path))

def freeze(model):
    for parameter in model.parameters():
        parameter.requires_grad = False
    print("froze %s" % type(model))

class Model(torch.nn.Module):
    def __init__(self, config, root):
        super(Model, self).__init__()
        # initialize encoder
        encoder_config = config["encoder"]
        self.encoder = getattr(encoder, encoder_config["type"])(**encoder_config["args"])
        if encoder_config["load"]:
            load(self.encoder, root, encoder_config["pretrained_path"])
        if encoder_config["freeze"]:
            freeze(self.encoder)
        # initialize backbone
        backbone_config = config['backbone']
        self.backbone = getattr(backbone, backbone_config["type"])(**backbone_config["args"])
        if backbone_config["load"]:
            load(self.backbone, root, backbone_config["pretrained_path"])
        if backbone_config["freeze"]:
            freeze(self.backbone)
        # initialize classifier
        classifier_config = config['classifier']
        self.classifier = getattr(classifier, classifier_config["type"])(**classifier_config["args"])

    def forward(self, waveform):
        features = self.encoder(waveform)
        output = self.backbone(features)
        return output

if __name__ == "__main__":
    import os
    root = "/ws/ifp-10_3/hasegawa/junzhez2/MaxMin_Pytorch"
    config_filename = "configs/SAE_RNN.json"
    with open(os.path.join(root, config_filename)) as file:
        config = json.load(file)
    model = Model(config["model"], root)
    waveform = torch.randn(1, 80 * 4096)
    print(model(waveform).shape)