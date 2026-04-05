import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import ocnn
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保项目根目录在 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from config.config import cfg
import src.models.ocnn_model_ref.my_ocnn as ocnn_unet


class AcousticFieldHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, vertex_features):
        return self.layers(vertex_features)


class MyPipeline(pl.LightningModule):
    def __init__(self, learning_rate=None):
        super().__init__()
        self.learning_rate = learning_rate if learning_rate is not None else getattr(cfg, "LEARNING_RATE", 1e-3)
        self.hidden_dim = getattr(cfg, "HIDDEN_DIM", 256)
        self.output_dim = getattr(cfg, "OUTPUT_DIM", 256)
        self.input_feature = ocnn.modules.InputFeature("NPD", nempty=cfg.OCTREE_NEMPTY)
        self.backbone_network = ocnn_unet.UNet(in_channels=7, out_channels=self.hidden_dim, nempty=cfg.OCTREE_NEMPTY)
        self.acoustic_head = AcousticFieldHead(self.hidden_dim, self.output_dim)

    def build_targets(self, batch_data):
        targets = torch.cat([mel.float().to(self.device).max(dim=-1).values for mel in batch_data["mel_spectrogram"]], dim=0)
        return F.adaptive_avg_pool1d(targets.unsqueeze(1), self.output_dim).squeeze(1)

    def build_prediction_report(self, targets, output, loss, stage):
        gt = targets.detach().cpu()
        pred = output.detach().cpu()
        diff = (pred - gt).abs()
        sample_idx = int(diff.mean(dim=1).argmax().item())
        gt_sample = gt[sample_idx]
        pred_sample = pred[sample_idx]
        diff_sample = diff[sample_idx]
        sample_count = min(8, gt.size(0))
        gt_panel = gt[:sample_count]
        pred_panel = pred[:sample_count]
        mae = diff.mean().item()
        rmse = torch.sqrt(((pred - gt) ** 2).mean()).item()
        corr = torch.corrcoef(torch.stack([gt_sample, pred_sample]))[0, 1].item() if gt_sample.numel() > 1 else 0.0
        worst_dims = torch.topk(diff_sample, k=min(8, diff_sample.numel())).indices.tolist()

        fig = plt.figure(figsize=(16, 10), dpi=160)
        gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1.6, 1.6])

        ax_text = fig.add_subplot(gs[0, :])
        ax_text.axis("off")
        text = "\n".join([
            f"epoch={self.current_epoch}  {stage}_loss={float(loss.item()):.6f}  samples={gt.size(0)}  dims={gt.size(1)}",
            f"global_mae={mae:.6f}  global_rmse={rmse:.6f}  sample_idx={sample_idx}  sample_corr={corr:.6f}",
            f"gt(mean/std/min/max)=({gt_sample.mean():.4f}, {gt_sample.std():.4f}, {gt_sample.min():.4f}, {gt_sample.max():.4f})",
            f"pred(mean/std/min/max)=({pred_sample.mean():.4f}, {pred_sample.std():.4f}, {pred_sample.min():.4f}, {pred_sample.max():.4f})",
            f"worst_dims={worst_dims}",
        ])
        ax_text.text(0.01, 0.98, text, va="top", ha="left", family="monospace", fontsize=11)

        ax_line = fig.add_subplot(gs[1, 0])
        ax_line.plot(gt_sample.numpy(), label="GT", linewidth=2)
        ax_line.plot(pred_sample.numpy(), label="Pred", linewidth=2)
        ax_line.set_title("GT vs Pred")
        ax_line.legend()

        ax_diff = fig.add_subplot(gs[1, 1])
        ax_diff.bar(range(diff_sample.numel()), diff_sample.numpy(), color="tab:red")
        ax_diff.set_title("Absolute Error")

        ax_scatter = fig.add_subplot(gs[1, 2])
        ax_scatter.scatter(gt_sample.numpy(), pred_sample.numpy(), s=18, alpha=0.8)
        lo = min(gt_sample.min().item(), pred_sample.min().item())
        hi = max(gt_sample.max().item(), pred_sample.max().item())
        ax_scatter.plot([lo, hi], [lo, hi], color="black", linewidth=1)
        ax_scatter.set_title("GT-Pred Scatter")
        ax_scatter.set_xlabel("GT")
        ax_scatter.set_ylabel("Pred")

        ax_gt = fig.add_subplot(gs[2, 0])
        im_gt = ax_gt.imshow(gt_panel.numpy(), aspect="auto", cmap="viridis")
        ax_gt.set_title("GT Heatmap")
        fig.colorbar(im_gt, ax=ax_gt, fraction=0.046, pad=0.04)

        ax_pred = fig.add_subplot(gs[2, 1])
        im_pred = ax_pred.imshow(pred_panel.numpy(), aspect="auto", cmap="viridis")
        ax_pred.set_title("Pred Heatmap")
        fig.colorbar(im_pred, ax=ax_pred, fraction=0.046, pad=0.04)

        ax_err = fig.add_subplot(gs[2, 2])
        im_err = ax_err.imshow((pred_panel - gt_panel).abs().numpy(), aspect="auto", cmap="magma")
        ax_err.set_title("AbsDiff Heatmap")
        fig.colorbar(im_err, ax=ax_err, fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.canvas.draw()
        image = torch.from_numpy(np.asarray(fig.canvas.buffer_rgba()).copy()[..., :3]).permute(2, 0, 1)
        plt.close(fig)
        return image

    def forward(self, batch_data):
        positions = [pos.to(self.device) for pos in batch_data["gnn_vertices"]]
        octree = batch_data["octree"].to(self.device)
        data = self.input_feature(octree)
        offsets = torch.tensor([0] + [pos.size(0) for pos in positions[:-1]], dtype=torch.long, device=self.device).cumsum(0)
        hit_face_indices = torch.cat([
            gnn_face_index.to(self.device).long() + offset
            for gnn_face_index, offset in zip(batch_data["gnn_face_index"], offsets)
        ], dim=0)
        hit_barycentric = torch.cat([weights.to(self.device) for weights in batch_data["gnn_barycentric"]], dim=0)
        targets = self.build_targets(batch_data)
        query_batch_index = torch.cat(
            [torch.full((pos.size(0),), idx, dtype=torch.long, device=self.device) for idx, pos in enumerate(positions)],
            dim=0,
        )
        query_pts = torch.cat(
            [torch.cat([pos, batch_idx[:, None].float()], dim=1) for pos, batch_idx in zip(positions, query_batch_index.split([pos.size(0) for pos in positions]))],
            dim=0,
        )
        vertex_features = self.backbone_network(data=data, octree=octree, depth=octree.depth, query_pts=query_pts)
        vertex_embeddings = self.acoustic_head(vertex_features)
        output = (vertex_embeddings[hit_face_indices] * hit_barycentric.unsqueeze(-1)).sum(dim=1)
        loss = F.smooth_l1_loss(output, targets)
        return loss, output

    def training_step(self, batch, batch_idx):
        loss, output = self(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        if batch_idx == 0 and getattr(self.logger, "experiment", None) is not None:
            targets = self.build_targets(batch)
            report = self.build_prediction_report(targets, output, loss, stage="train")
            self.logger.experiment.add_image("train/gt_pred_absdiff", report, self.current_epoch)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, output = self(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        if batch_idx == 0 and getattr(self.logger, "experiment", None) is not None:
            targets = self.build_targets(batch)
            report = self.build_prediction_report(targets, output, loss, stage="val")
            self.logger.experiment.add_image("val/gt_pred_absdiff", report, self.current_epoch)
        return loss

    def test_step(self, batch, batch_idx):
        loss, _ = self(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch["num_impacts"].sum().item())
        return loss

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=getattr(cfg, "WEIGHT_DECAY", 0.0),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
