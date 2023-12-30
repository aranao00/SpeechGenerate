import torch
import torch.nn as nn

class Encoder(nn.Module):#음성 번역기에서 사용한 인코더를 그대로 사용한다.
    pass

class Generate(nn.Module):
    def __init__(self):
        super(Generate, self).__init__()
        self.dlay=nn.TransformerDecoderLayer()
        self.dec=nn.TransformerDecoder()

class Synthesizer(nn.Module):#음성 번역기에서 사용한 합성기 그대로 사용한다.
    pass


def train_generate(src, tgt):
    encoder=Encoder()
    synthesizer=Synthesizer()
    model=Generate()
    optim=torch.optim.AdamW()
    lossfn=nn.MSELoss()
    
    enc_seq=encoder(src)
    pred_seq=model(enc_seq)
    pred_spectrogram=synthesizer(pred_seq)

    loss=lossfn(pred_spectrogram, tgt)
    optim.zero_grad()
    loss.backward()
    optim.step()