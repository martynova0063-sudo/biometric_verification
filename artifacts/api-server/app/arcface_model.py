import torch
import torch.nn as nn
import torchvision.models as models

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m

    def forward(self, input, label):
        cosine = torch.nn.functional.linear(
            torch.nn.functional.normalize(input),
            torch.nn.functional.normalize(self.weight)
        )
        return cosine, self.s, self.m

class ResNet50ArcFace(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, pretrained=True):
        super().__init__()
        weights = (
            models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        )
        resnet = models.resnet50(weights=weights)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.embedding = nn.Sequential(
            nn.Linear(2048, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self.arcface = ArcMarginProduct(embedding_dim, num_classes)

    def forward(self, x, label=None):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        emb = self.embedding(x)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)

        if label is not None:
            logits, s, m = self.arcface(emb, label)
            return emb, logits, s, m
        return emb
