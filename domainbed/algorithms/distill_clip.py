import clip
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel,ViTForImageClassification

from domainbed import networks
from domainbed.algorithms import Algorithm
from domainbed.algorithms.miro import (
    MeanEncoder,
    VarianceEncoder,
    get_shapes,
    ForwardModel,
    get_optimizer,
)
from domainbed.networks.ur_networks import URFeaturizer
from domainbed.lib import misc
from tqdm import tqdm
from itertools import chain

dims = {
    "RN50": 1024,
    "RN101": 512,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
}

class CustomViT(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        try:
            self.vit = ViTForImageClassification.from_pretrained(
                model_path,
                ignore_mismatched_sizes=True,
                output_hidden_states=True,  # 确保返回隐藏状态
            )
            print(f"ViT_Deepfake_Detection model loaded successfully from {model_path}.")
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            self.vit = None  # 设置为 None 以便后续处理

        self.dropout = nn.Dropout(p=0.1)  # 假设原代码中有 Dropout 层

    def forward(self, x):
        if self.vit is None:
            raise RuntimeError("Model is not loaded. Check initialization.")
        outputs = self.vit(x)  # 返回完整的输出对象
        hidden_states = outputs.hidden_states  # 获取所有隐藏状态
        features = hidden_states[-1][:, 0, :]  # 提取最后一层的 CLS Token 特征 [batch_size, 768]
        return self.dropout(features)
class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)


class LabelSmoothingCrossEntropy(nn.Module):

    def __init__(self, label_smoothing=0.1, num_classes=2, reduction='mean'):
        super().__init__()
        self.smoothing = label_smoothing
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, pred, target):
        log_pred = F.log_softmax(pred, dim=-1)


        smooth_target = (1 - self.smoothing) * F.one_hot(target, self.num_classes) \
                        + self.smoothing / (self.num_classes - 1) * (1 - F.one_hot(target, self.num_classes))


        loss = - (smooth_target * log_pred).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class TransformerModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, hidden_dim):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=8,
                dim_feedforward=2048
            ),
            num_layers=2,
        )

    def forward(self, x):
        # 假设输入是 [batch_size, seq_len, embed_dim]
        x = x.permute(1, 0, 2)  # Transformer 需要 [seq_len, batch_size, embed_dim]
        x = self.transformer(x)
        return x.permute(1, 0, 2)


class CLIP:
    def __init__(self, hparams):
        self.device = "cuda"
        self.hparams = hparams
        self.clip_model = clip.load(self.hparams["clip_backbone"], device=self.device)[0].float()
        self.clip_model.eval()
        self.clip_model.to(self.device)
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # 添加 Transformer 模块
        embed_dim = dims[self.hparams["clip_backbone"]]
        # self.transformer = TransformerModel(
        #     embed_dim=embed_dim,
        #     num_heads=8,  # 设置注意力头的数量
        #     num_layers=3,  # Transformer 层的数量
        #     hidden_dim=2048,  # FFN 的隐藏层维度
        # ).to(self.device)

    # def get_img_feat(self, x):
    #     """获取经过 Transformer 处理后的图像特征"""
    #     # CLIP 图像编码
    #     clip_img_feat = self.clip_model.encode_image(x)
    #     clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
    #
    #     # 添加维度并传入 Transformer
    #     clip_img_feat = clip_img_feat.unsqueeze(1)  # [batch_size, 1, embed_dim]
    #     transformer_feat = self.transformer(clip_img_feat)
    #     return transformer_feat.squeeze(1)
    def get_img_feat(self, x):
        """Get normalized image embeddings"""

        clip_img_feat = self.clip_model.encode_image(x)

        clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        return clip_img_feat

    def get_txt_feat(self, labels):
        """Get normalized text embeddings"""

        clip_txt_feat = []
        for i in range(labels.size(0)):
            feat = self.zeroshot_weights[labels[i].item()]
            clip_txt_feat.append(feat)
        clip_txt_feat = torch.stack(clip_txt_feat, dim=0)
        return clip_txt_feat

    def get_zeroshot_classifier(self, classnames, templates):
        logit_scale = self.clip_model.logit_scale

        print("Getting zeroshot weights.")
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [
                    template.format(class_name=classname) for template in templates
                ]

                # Embeddings for each class
                texts = clip.tokenize(texts).to(self.device)  # tokenize
                embeddings = self.clip_model.encode_text(texts)

                # embed with text encoder
                embeddings /= embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.mean(dim=0, keepdim=True)
                embeddings /= embeddings.norm()
                zeroshot_weights.append(embeddings)

            # Computing zero-shot weights
            zeroshot_weights = torch.stack(zeroshot_weights, dim=0).to(self.device)
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 2)
            zeroshot_weights *= logit_scale.exp()
            zeroshot_weights = zeroshot_weights.squeeze().float()
            zeroshot_weights = torch.transpose(zeroshot_weights, 0, 1)

        classification_head = ClassificationHead(
            normalize=True, weights=zeroshot_weights
        )
        self.zeroshot_weights = zeroshot_weights
        return classification_head, zeroshot_weights


class DistillCLIP(Algorithm):
    def __init__(
        self, input_shape, num_classes, num_domains, hparams, clip_model, tgt_dom=0
    ):
        super(DistillCLIP, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )

        self.hparams = hparams
        self.lmd = hparams["lmd"]
        self.cls_num = num_classes
        self.dom_num = num_domains + 1
        self.prompt_style = hparams["prompt_style"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.featurizer = networks.Featurizer(input_shape, self.hparams)

        # 加载 DeepFakeDetector 模型
        self.dfd_model = CustomViT("/media/vipsl04/Harddisk/Deep-Fake-Detector-v2-Model").to(self.device)
        for param in self.dfd_model.parameters():
            param.requires_grad = False

        # 加载 CLIP 零样本分类器
        classnames = [name.replace("_", " ") for name in hparams["class_names"]]
        if self.prompt_style == 1:
            templates = ["{class_name}"]
        elif self.prompt_style == 2:
            templates = ["a photo of a {class_name}"]
        elif self.prompt_style == 3:
            templates = [
                "an art of a {class_name}",
                "a clipart of a {class_name}",
                "a photo of a {class_name} product",
                "a photo of a {class_name}",
            ]
        elif self.prompt_style == 4:
            templates = [
                "an art photo of a {size} {{class_name}}",
                "a clipart photo of a {size} {{class_name}}",
                "a product photo of a {size} {{class_name}}",
                "a real photo of a {size} {{class_name}}",
            ]
        elif self.prompt_style == 5:
            templates = [
                "an art of a {class_name}",
                "a clipart of a {class_name}",
                "a photo of a {class_name} product",
                "a photo of a {class_name}",
            ]
            del templates[tgt_dom]
        elif self.prompt_style == 6:
            classnames = [str(i) for i in range(len(classnames))]
            templates = ["a photo of a {class_name}"]

        self.clip_cls, self.zeroshot_weights = clip_model.get_zeroshot_classifier(
            classnames, templates
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # CLIP 特征
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
        clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

        # DeepFakeDetector 特征
        with torch.no_grad():
            dfd_feats = self.dfd_model(all_x).last_hidden_state.mean(dim=1)

        # 特征提取器和投影层
        img_feat = self.featurizer(all_x)
        proj_feat = self.proj_lyr(img_feat)
        proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # 计算损失
        clip_loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)
        dfd_loss = F.mse_loss(img_feat, dfd_feats)  # 与 DeepFakeDetector 特征对齐
        total_loss = clip_loss + dfd_loss

        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {"loss": total_loss.item(), "clip_loss": clip_loss.item(), "dfd_loss": dfd_loss.item()}


class DFC_STAGE1(DistillCLIP):
    """
    - ImNet frozen w/ CLIP loss
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE1, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 1 - ImNet frozen w/ CLIP loss\n")

        ##########################
        # CHANGES TO MAKE FOR ID
        ##########################
        """
        1. set embed_dim=512
        """
        ##########################

        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.proj_lyr.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE11(DistillCLIP):
    """
    - ImNet frozen w/ CLIP loss, act
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE11, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 1 - ImNet frozen w/ CLIP loss, ACT\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.proj_lyr.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = F.gelu(self.proj_lyr(img_feat))
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(F.gelu(x))

    def forward(self, x):
        return self.predict(x)


class DFC_CLIP_INIT(DistillCLIP):
    """
    - ImNet frozen w/ CLIP loss, no projection (CLIP Init)
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_CLIP_INIT, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nImNet frozen w/ CLIP loss, no projection (CLIP Init)\n")

        ##########################
        # CHANGES TO MAKE FOR ID
        ##########################
        """
        1. set embed_dim=512
        """
        ##########################

        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        train_params = chain(
            self.featurizer.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            img_feat_norm = img_feat / img_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, img_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE2(DistillCLIP):
    """
    Stage 2 - Proj lyr frozen w/ CLIP loss
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE2, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )

        ##########################
        # CHANGES TO MAKE FOR ID
        ##########################
        """
        1. set embed_dim=512
        """
        ##########################

        print("\n\nStage 2 - Proj lyr frozen w/ CLIP loss\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        # # CLIP features
        # clip_img_feat = self.clip_model.encode_image(all_x)
        # clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        # clip_txt_feat = []
        # for i in range(all_y.size(0)):
        #     feat = self.zeroshot_weights[all_y[i].item()]
        #     clip_txt_feat.append(feat)
        # clip_txt_feat = torch.stack(clip_txt_feat, dim=0)

        # img_feat = self.featurizer(all_x)
        # proj_feat = self.proj_lyr(img_feat)
        # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # # Losses
        # kd_loss_1 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_img_feat))
        # kd_loss_2 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_txt_feat))
        # loss = self.lmd * kd_loss_1 + (1 - self.lmd) * kd_loss_2

        # # [#] Backward pass

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return {"loss": loss.item()}

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE21(DistillCLIP):
    """
    Stage 2 - Proj lyr frozen w/ CLIP loss, ACT
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE21, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 2 - Proj lyr frozen w/ CLIP loss, ACT\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # [#] cutmix forward pass
        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = F.gelu(self.proj_lyr(img_feat))
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = F.gelu(self.proj_lyr(x))
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE22(DistillCLIP):
    """
    Stage 2 - Proj lyr frozen w/ CLIP loss, FULL
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE22, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 2 - Proj lyr frozen w/ CLIP loss, FULL\n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.classifier.parameters(),
            self.proj_lyr.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # loss comp.
            loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE31(DistillCLIP):
    """
    Stage 3 - classifier only
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE31, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - classifier only \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.classifier.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        # # CLIP features
        # clip_img_feat = self.clip_model.encode_image(all_x)
        # clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        # clip_txt_feat = []
        # for i in range(all_y.size(0)):
        #     feat = self.zeroshot_weights[all_y[i].item()]
        #     clip_txt_feat.append(feat)
        # clip_txt_feat = torch.stack(clip_txt_feat, dim=0)

        # img_feat = self.featurizer(all_x)
        # proj_feat = self.proj_lyr(img_feat)
        # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # # Losses
        # kd_loss_1 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_img_feat))
        # kd_loss_2 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_txt_feat))
        # loss = self.lmd * kd_loss_1 + (1 - self.lmd) * kd_loss_2

        # # [#] Backward pass

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return {"loss": loss.item()}

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            # clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            # clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # # forward pass
            # img_feat = self.featurizer(all_x)
            # proj_feat = self.proj_lyr(img_feat)
            # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # # loss comp.
            # loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE32(DistillCLIP):
    """
    Stage 3 - Full network
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE32, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - classifier only \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.proj_lyr.parameters(),
            self.classifier.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # # [#] Forward pass

        # # CLIP features
        # clip_img_feat = self.clip_model.encode_image(all_x)
        # clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        # clip_txt_feat = []
        # for i in range(all_y.size(0)):
        #     feat = self.zeroshot_weights[all_y[i].item()]
        #     clip_txt_feat.append(feat)
        # clip_txt_feat = torch.stack(clip_txt_feat, dim=0)

        # img_feat = self.featurizer(all_x)
        # proj_feat = self.proj_lyr(img_feat)
        # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # # Losses
        # kd_loss_1 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_img_feat))
        # kd_loss_2 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_txt_feat))
        # loss = self.lmd * kd_loss_1 + (1 - self.lmd) * kd_loss_2

        # # [#] Backward pass

        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        # return {"loss": loss.item()}

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            # clip features
            # clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            # clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

            # # forward pass
            # img_feat = self.featurizer(all_x)
            # proj_feat = self.proj_lyr(img_feat)
            # proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # # loss comp.
            # loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE33(DistillCLIP):
    """
    Stage 3 - Classifier only, new cls
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE33, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - Classifier only, new cls \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.classifier.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE34(DistillCLIP):
    """
    Stage 3 - full network, new cls
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE34, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - full network, new cls \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.proj_lyr.parameters(),
            self.classifier.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE35(DistillCLIP):
    """
    Stage 3 - classifier only, new cls, no proj
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE35, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - classifier only, new cls, no proj \n")
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.new_cls = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.new_cls.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        r = np.random.rand(1)
        if self.hparams["beta"] > 0 and r < self.hparams["cutmix_prob"]:
            # print("Cutmix")
            # generate mixed sample
            beta = self.hparams["beta"]
            lam = np.random.beta(beta, beta)
            rand_index = torch.randperm(all_x.size()[0]).cuda()
            target_a = all_y
            target_b = all_y[rand_index]
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(all_x.size(), lam)
            all_x[:, :, bbx1:bbx2, bby1:bby2] = all_x[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]

            # adjust lambda to exactly match pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (all_x.size()[-1] * all_x.size()[-2])
            )

            # forward pass
            img_feat = self.featurizer(all_x)
            proj_feat = self.proj_lyr(img_feat)
            proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

            # clip features
            clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
            clip_txt_feat_a = kwargs["clip_model"].get_txt_feat(target_a)
            clip_txt_feat_b = kwargs["clip_model"].get_txt_feat(target_b)

            # loss comp.
            loss_a = self.dfc_loss(
                clip_img_feat, clip_txt_feat_a, proj_feat_norm, self.lmd
            )
            loss_b = self.dfc_loss(
                clip_img_feat, clip_txt_feat_b, proj_feat_norm, self.lmd
            )
            loss = lam * loss_a + (1.0 - lam) * loss_b

        # [#] normal forward pass
        else:
            logits = self.predict(all_x)
            loss = F.cross_entropy(logits, all_y)

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        return self.new_cls(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE36(DistillCLIP):
    """
    Stage 3 - Stage 2 init, KLD distillation
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    @staticmethod
    def rand_bbox(size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE36, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nStage 3 - Stage 2 init, KLD distillation \n")
        self.tmp = hparams["tmp"]
        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.new_cls = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        train_params = chain(
            self.featurizer.parameters(),
            self.new_cls.parameters(),
        )
        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # [#] forward pass and loss
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
        clip_logits = self.clip_cls(clip_img_feat) / self.tmp

        # forward pass of student
        img_feat = self.featurizer(all_x)
        logits = self.new_cls(img_feat) / self.tmp

        # cos sim loss
        ce_loss = F.cross_entropy(logits, all_y)
        sim_loss = F.kl_div(logits.softmax(dim=-1), clip_logits.softmax(dim=-1))
        loss = ce_loss + (self.lmd * sim_loss)

        # [#] backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        return self.new_cls(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE2_MIRO(DistillCLIP):
    """
    - Stage 2 training on backbone + MIRO
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(DFC_STAGE2_MIRO, self).__init__(
            input_shape, num_classes, num_domains, hparams
        )
        print("\n\n Stage 2 training on backbone \n")
        self.embed_dim = 512
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)
        self.pre_featurizer = URFeaturizer(
            input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
        )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.ld = hparams.ld

        # train_params = chain(
        #     self.featurizer.parameters(),
        # )
        # for param in self.pre_featurizer.parameters():
        #     param.requires_grad = False

        # build mean/var encoders
        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes])
        self.var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes])
        # optimizer
        parameters = [
            {"params": self.featurizer.parameters()},
            {
                "params": self.mean_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
            {
                "params": self.var_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

        # self.optimizer = torch.optim.Adam(
        #     train_params,
        #     lr=self.hparams["lr"],
        #     weight_decay=self.hparams["weight_decay"],
        # )

    def update(self, x, y, **kwargs):
        # all_x = torch.cat([x for x, y in minibatches])
        # all_y = torch.cat([y for x, y in minibatches])
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # [#] Forward pass

        # CLIP features
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
        clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

        # clip_img_feat = self.clip_model.encode_image(all_x)
        # clip_img_feat /= clip_img_feat.norm(dim=-1, keepdim=True)
        # clip_txt_feat = []
        # for i in range(all_y.size(0)):
        #     feat = self.zeroshot_weights[all_y[i].item()]
        #     clip_txt_feat.append(feat)
        # clip_txt_feat = torch.stack(clip_txt_feat, dim=0)

        img_feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        proj_feat = self.proj_lyr(img_feat)
        proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # CLIP cos sim loss
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_img_feat))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj_feat_norm, clip_txt_feat))
        loss = self.lmd * kd_loss_1 + (1 - self.lmd) * kd_loss_2

        # MIRO loss
        with torch.no_grad():
            _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.0
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            # mutual information regularization
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.0

        loss += reg_loss * self.ld

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE3_old(Algorithm):
    """Mutual-Information Regularization with Oracle"""

    def __init__(self, input_shape, num_classes, num_domains, hparams, **kwargs):
        super().__init__(input_shape, num_classes, num_domains, hparams)
        self.pre_featurizer = URFeaturizer(
            input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
        )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.ld = hparams.ld

        # build mean/var encoders
        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes])
        self.var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes])

        # optimizer
        parameters = [
            {"params": self.network.parameters()},
            {
                "params": self.mean_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
            {
                "params": self.var_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)
        feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        logit = self.classifier(feat)
        loss = F.cross_entropy(logit, all_y)

        # MIRO
        with torch.no_grad():
            _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.0
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            # mutual information regularization
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.0

        loss += reg_loss * self.ld

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "reg_loss": reg_loss.item()}

    def predict(self, x):
        return self.network(x)

    def get_forward_model(self):
        forward_model = ForwardModel(self.network)
        return forward_model


class DFC_STAGE3_MIRO(DistillCLIP):
    """
    - Stage 3 training on backbone + MIRO
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE3_MIRO, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model=clip_model
        )
        print("\n\n Stage 3 w/ MIRO ...\n")
        self.embed_dim = 512

        # [#] stage 3 - clip cls
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)

        self.pre_featurizer = URFeaturizer(
            input_shape, self.hparams, freeze="all", feat_layers=hparams.feat_layers
        )
        self.featurizer = URFeaturizer(
            input_shape, self.hparams, feat_layers=hparams.feat_layers
        )
        # self.network = nn.Sequential(self.featurizer, self.proj_lyr, self.classifier)
        self.ld = hparams.ld

        # train_params = chain(
        #     self.featurizer.parameters(),
        # )
        # for param in self.pre_featurizer.parameters():
        #     param.requires_grad = False

        # build mean/var encoders
        shapes = get_shapes(self.pre_featurizer, self.input_shape)
        self.mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes])
        self.var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes])
        # optimizer
        # {"params": self.classifier.parameters()},
        # parameters = [
        #     {"params": self.classifier.parameters()},
        #     {
        #         "params": self.mean_encoders.parameters(),
        #         "lr": hparams.lr * hparams.lr_mult,
        #     },
        #     {
        #         "params": self.var_encoders.parameters(),
        #         "lr": hparams.lr * hparams.lr_mult,
        #     },
        # ]
        parameters = [
            {"params": self.featurizer.parameters()},
            {"params": self.proj_lyr.parameters()},
            {"params": self.classifier.parameters()},
            {
                "params": self.mean_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
            {
                "params": self.var_encoders.parameters(),
                "lr": hparams.lr * hparams.lr_mult,
            },
        ]
        self.optimizer = get_optimizer(
            hparams["optimizer"],
            parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        # all_x = torch.cat([x for x, y in minibatches])
        # all_y = torch.cat([y for x, y in minibatches])
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # [#] MIRO forward pass

        # forward pass of RN50
        feat, inter_feats = self.featurizer(all_x, ret_feats=True)
        proj_feat = self.proj_lyr(feat)
        logit = self.classifier(proj_feat)
        loss = F.cross_entropy(logit, all_y)

        # MIRO loss
        with torch.no_grad():
            _, pre_feats = self.pre_featurizer(all_x, ret_feats=True)

        reg_loss = 0.0
        for f, pre_f, mean_enc, var_enc in misc.zip_strict(
            inter_feats, pre_feats, self.mean_encoders, self.var_encoders
        ):
            # mutual information regularization
            mean = mean_enc(f)
            var = var_enc(f)
            vlb = (mean - pre_f).pow(2).div(var) + var.log()
            reg_loss += vlb.mean() / 2.0

        loss += reg_loss * self.ld

        # [#] Backward pass

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)
class DFC_STAGE_COMBINED1(DistillCLIP):
    """
    Combined Stage - Featurizer and Proj_lyr jointly trained w/ CLIP loss
    """

    @staticmethod
    def dfc_loss(img, txt, proj, lmd):
        kd_loss_1 = -torch.mean(F.cosine_similarity(proj, img))
        kd_loss_2 = -torch.mean(F.cosine_similarity(proj, txt))
        kd_loss = 0.25 * kd_loss_1 +  0.25 * kd_loss_2
        return kd_loss

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None):
        super(DFC_STAGE_COMBINED, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nCombined Stage - Featurizer and Proj_lyr jointly trained w/ CLIP loss\n")

        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)

        # Combine parameters from both featurizer and proj_lyr
        train_params = chain(
            self.featurizer.parameters(),
            self.proj_lyr.parameters(),
        )

        self.optimizer = torch.optim.Adam(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # Forward pass
        # CLIP features
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
        clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

        # Featurizer and projection layer
        img_feat = self.featurizer(all_x)
        proj_feat = self.proj_lyr(img_feat)
        proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # Compute loss
        loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def predict(self, x):
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)


class DFC_STAGE_COMBINED(DistillCLIP):
    """
    Combined Stage with DeepFakeDetector Alignment
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams, clip_model=None, dfd_model=None):
        super(DFC_STAGE_COMBINED, self).__init__(
            input_shape, num_classes, num_domains, hparams, clip_model
        )
        print("\n\nCombined Stage with DeepFakeDetector Alignment with Multi-Task Learning\n")

        self.embed_dim = dims[hparams["clip_backbone"]]
        self.classifier = self.clip_cls
        self.proj_lyr = nn.Linear(self.featurizer.n_outputs, self.embed_dim)

        # 添加多任务学习参数（直接定义而非通过hparams）
        self.cls_weight = 0.5  # 分类损失权重
        self.dfd_weight = 2.0 # DFD对齐损失权重
        self.temperature = 0.07  # 温度参数（固定值）

        # DeepFakeDetector 模型（ViT）
        self.dfd_model = CustomViT("/media/vipsl04/Harddisk/Deep-Fake-Detector-v2-Model").to(self.device)
        for param in self.dfd_model.parameters():  # 冻结 DeepFakeDetector 的参数
            param.requires_grad = False

        # 优化器（保持原样）
        train_params = chain(
            self.featurizer.parameters(),
            self.proj_lyr.parameters(),
        )
        self.optimizer = torch.optim.AdamW(
            train_params,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def dfc_loss(self, img, txt, proj, lmd):
        """CLIP 蒸馏损失（添加温度参数）"""
        img_sim = F.cosine_similarity(proj, img) / self.temperature
        txt_sim = F.cosine_similarity(proj, txt) / self.temperature
        kd_loss_1 = -torch.mean(img_sim)
        kd_loss_2 = -torch.mean(txt_sim)
        kd_loss = lmd * kd_loss_1 + (1 - lmd) * kd_loss_2
        return kd_loss

    def dfd_loss(self, feats, dfd_feats):
        """DeepFakeDetector 特征对齐损失"""
        return 1 - torch.mean(F.cosine_similarity(feats, dfd_feats))

    def update(self, x, y, **kwargs):
        all_x = torch.cat(x)
        all_y = torch.cat(y)

        # CLIP 特征
        clip_img_feat = kwargs["clip_model"].get_img_feat(all_x)
        clip_txt_feat = kwargs["clip_model"].get_txt_feat(all_y)

        # DeepFakeDetector 特征
        if self.dfd_model is not None:
            with torch.no_grad():
                dfd_feats = self.dfd_model(all_x)
                dfd_feats = nn.Linear(768, 512).to(self.device)(dfd_feats)
                dfd_feats = dfd_feats / dfd_feats.norm(dim=-1, keepdim=True)
        else:
            dfd_feats = None

        # 特征提取器和投影层
        img_feat = self.featurizer(all_x)
        proj_feat = self.proj_lyr(img_feat)
        proj_feat_norm = proj_feat / proj_feat.norm(dim=-1, keepdim=True)

        # 计算各项损失
        clip_loss = self.dfc_loss(clip_img_feat, clip_txt_feat, proj_feat_norm, self.lmd)

        # 添加分类损失
        cls_logits = self.classifier(proj_feat_norm)
        cls_loss = F.cross_entropy(cls_logits, all_y)

        # DFD对齐损失
        if dfd_feats is not None:
            dfd_loss = - torch.mean(F.cosine_similarity(proj_feat_norm, dfd_feats))
        else:
            dfd_loss = 0.0

        # 多任务损失组合
        total_loss = clip_loss + self.cls_weight * cls_loss + self.dfd_weight * dfd_loss

        # 反向传播和优化
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "clip_loss": clip_loss.item(),
            "cls_loss": cls_loss.item(),  # 新增分类损失监控
            "dfd_loss": dfd_loss.item()
        }

    def predict(self, x):
        """推理（保持不变）"""
        x = self.featurizer(x)
        x = self.proj_lyr(x)
        return self.classifier(x)

    def forward(self, x):
        return self.predict(x)