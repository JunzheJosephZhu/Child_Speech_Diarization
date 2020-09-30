import torch
import json
import encoder
import backbone

class Model(torch.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        encoder_config = config["encoder"]
        self.encoder = getattr(encoder, encoder_config["type"])(**encoder_config["args"])
        backbone_config = config['backbone']
        self.backbone = getattr(backbone, backbone_config["type"])(**backbone_config["args"])
        if config["freeze_encoder"]:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

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
        encoder_config = config['model']['encoder']
        if encoder_config["type"] == "SAE":
            encoder_config['args']['pretrained_path'] = os.path.join(root, encoder_config['args']['pretrained_path'])
    model = Model(config["model"])
    waveform = torch.randn(1, 80 * 4096)
    print(model(waveform).shape)