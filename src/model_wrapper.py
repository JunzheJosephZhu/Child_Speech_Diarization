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
    def __init__(self, config):
        super(Model, self).__init__()
        root = config["root"]

        # initialize encoder
        encoder_config = config["model"]["encoder"]
        self.encoder = getattr(encoder, encoder_config["type"])(**encoder_config["args"])

        # initialize backbone
        backbone_config = config["model"]["backbone"]
        self.backbone = getattr(backbone, backbone_config["type"])(**backbone_config["args"])

        # initialize classifier
        classifier_config = config["model"]["classifier"]
        self.classifier = getattr(classifier, classifier_config["type"])(**classifier_config["args"])

        # use kaiming initialization because xavier doesn't work on wave-u-net
        # for parameter in self.parameters():
        #     if len(parameter.size()) > 1:
        #         torch.nn.init.kaiming_normal_(parameter)

        # load & freeze modules
        if encoder_config["load"]:
            load(self.encoder, root, encoder_config["pretrained_path"])
        if encoder_config["freeze"]:
            freeze(self.encoder)

        if backbone_config["load"]:
            load(self.backbone, root, backbone_config["pretrained_path"])
        if backbone_config["freeze"]:
            freeze(self.backbone)
            
    def forward(self, waveform, mask):
        features = self.encoder(waveform)
        embedding = self.backbone(features, mask)
        output = self.classifier(embedding, mask)
        return output

if __name__ == "__main__":
    import os
    torch.manual_seed(0)
    root = "/home/joseph/Desktop/MaxMin_Pytorch"
    config_filename = "configs/AE_RNN.json"
    with open(os.path.join(root, config_filename)) as file:
        config = json.load(file)
        config["root"] = root
    model = Model(config)

    torch.manual_seed(2)
    waveform = torch.randn(1, 80 * 4096)
    print(model(waveform))