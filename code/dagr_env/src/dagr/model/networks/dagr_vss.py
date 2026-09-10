import torch
import math

import torch.nn.functional as F

from torch_geometric.data import Data
from yolox.models import YOLOX, YOLOXHead, IOUloss

from dagr.model.networks.net_vss import Net
from dagr.model.layers.spline_conv import SplineConvToDense
from dagr.model.layers.conv import ConvBlock
from dagr.model.layers.ssm_hyper import hyperedge_regularization
from dagr.model.utils import shallow_copy, init_subnetwork, voxel_size_to_params, postprocess_network_output, convert_to_evaluation_format, init_grid_and_stride, convert_to_training_format


class DAGR(YOLOX):
    def __init__(self, args, height, width):
        self.conf_threshold = 0.001
        self.nms_threshold = 0.65

        self.height = height
        self.width = width

        backbone = Net(args, height=height, width=width)
        head = GNNHead(num_classes=backbone.num_classes,
                       in_channels=backbone.out_channels,
                       in_channels_cnn=backbone.out_channels_cnn,
                       strides=backbone.strides,
                       pretrain_cnn=args.pretrain_cnn,
                       args=args)

        super().__init__(backbone=backbone, head=head)

        self.ssm_hyper_entropy_weight = float(getattr(args, "ssm_hyper_entropy_weight", 0.0))
        self.ssm_hyper_sharpness_weight = float(getattr(args, "ssm_hyper_sharpness_weight", 0.0))
        self.ssm_hyper_consistency_weight = float(getattr(args, "ssm_hyper_consistency_weight", 0.0))

        if "img_net_checkpoint" in args:
            state_dict = torch.load(args.img_net_checkpoint)
            init_subnetwork(self, state_dict['ema'], "backbone.net.", freeze=True)
            init_subnetwork(self, state_dict['ema'], "head.cnn_head.")

    def cache_luts(self, width, height, radius):
        M = 2 * float(int(radius * width + 2) / width)
        r = int(radius * width + 1)

        # stage 1
        self.backbone.conv_block1.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=r)
        self.backbone.conv_block1.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=r)

        if getattr(self.backbone, "use_vss_backbone", False) and self.backbone.graph_stages == 1:
            return

        # stage 2
        rx, ry, M = voxel_size_to_params(self.backbone.pool1, height, width)
        self.backbone.layer2.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer2.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        if getattr(self.backbone, "use_vss_backbone", False) and self.backbone.graph_stages == 2:
            return

        # stage 3
        rx, ry, M = voxel_size_to_params(self.backbone.pool2, height, width)
        self.backbone.layer3.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer3.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        if getattr(self.backbone, "use_vss_backbone", False):
            return

        # stage 4
        rx, ry, M = voxel_size_to_params(self.backbone.pool3, height, width)
        self.backbone.layer4.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer4.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        # graph head scale 1
        self.head.stem1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.cls_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.reg_conv1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.cls_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.reg_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.head.obj_pred1.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        # stage 5
        rx, ry, M = voxel_size_to_params(self.backbone.pool4, height, width)
        self.backbone.layer5.conv_block1.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
        self.backbone.layer5.conv_block2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

        if self.head.num_scales > 1:
            self.head.stem2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.cls_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_conv2.conv.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.cls_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.reg_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)
            self.head.obj_pred2.init_lut(height=height, width=width, Mx=M, rx=rx, ry=ry)

    def forward(self, x: Data, reset=True, return_targets=True, filtering=True):
        if not hasattr(self.head, "output_sizes"):
            self.head.output_sizes = self.backbone.get_output_sizes()

        if self.training:
            targets = convert_to_training_format(x.bbox, x.bbox_batch, x.num_graphs)

            if self.backbone.use_image:
                targets0 = convert_to_training_format(x.bbox0, x.bbox0_batch, x.num_graphs)
                targets = (targets, targets0)

            # gt_target inputs need to be [l cx cy w h] in pixels
            outputs = YOLOX.forward(self, x, targets)

            reg_loss, reg_logs = self._compute_ssm_hyper_regularization()
            if reg_loss is not None:
                outputs["total_loss"] = outputs["total_loss"] + reg_loss
                outputs.update(reg_logs)

            return outputs

        x.reset = reset

        outputs = YOLOX.forward(self, x)

        detections = postprocess_network_output(outputs, self.backbone.num_classes, self.conf_threshold, self.nms_threshold, filtering=filtering,
                                                height=self.height, width=self.width)

        ret = [detections]

        if return_targets and hasattr(x, 'bbox'):
            targets = convert_to_evaluation_format(x)
            ret.append(targets)

        return ret

    def _compute_ssm_hyper_regularization(self):
        H = getattr(self.backbone, "_last_H", None)
        # The consistency term pulls nodes sharing a hyperedge toward similar
        # features of WHATEVER drove H (probe source); fall back to the SSM
        # state for checkpoints/modules that do not expose `_last_h_src`.
        h_src = getattr(self.backbone, "_last_h_src", None)
        if h_src is None:
            h_src = getattr(self.backbone, "_last_h_ssm", None)
        if H is None or H.numel() == 0:
            return None, {}
        return hyperedge_regularization(
            h_src,
            H,
            entropy_weight=self.ssm_hyper_entropy_weight,
            sharpness_weight=self.ssm_hyper_sharpness_weight,
            consistency_weight=self.ssm_hyper_consistency_weight,
        )


class CNNHead(YOLOXHead):
    def forward(self, xin):
        outputs = dict(cls_output=[], reg_output=[], obj_output=[])

        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            reg_feat = reg_conv(reg_x)

            outputs["cls_output"].append(self.cls_preds[k](cls_feat))
            outputs["reg_output"].append(self.reg_preds[k](reg_feat))
            outputs["obj_output"].append(self.obj_preds[k](reg_feat))

        return outputs


class GNNHead(YOLOXHead):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        in_channels_cnn=[256, 512, 1024],
        act="silu",
        depthwise=False,
        pretrain_cnn=False,
        args=None
    ):
        head_width = float(args.yolo_stem_width)
        yolox_in_channels = in_channels
        if bool(getattr(args, "use_vss_backbone", False)) and head_width != 1.0:
            # YOLOX scales both hidden width and declared input channels. VSS
            # already emits fixed-width tensors, so compensate the declaration
            # to keep the actual stem inputs equal to ``in_channels``.
            yolox_in_channels = [math.ceil(channel / head_width) for channel in in_channels]
        YOLOXHead.__init__(
            self, num_classes, head_width, strides, yolox_in_channels, act, depthwise
        )

        self.pretrain_cnn = pretrain_cnn
        self.num_scales = args.num_scales
        self.use_image = args.use_image
        self.batch_size = args.batch_size
        self.no_events = args.no_events

        self.use_vss_backbone = bool(getattr(args, "use_vss_backbone", False))

        self.in_channels = in_channels
        self.n_anchors = 1
        self.num_classes = num_classes

        if not self.use_vss_backbone:
            n_reg = max(in_channels)
            self.stem1 = ConvBlock(in_channels=in_channels[0], out_channels=n_reg, args=args)
            self.cls_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.cls_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
            self.reg_conv1 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
            self.reg_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
            self.obj_pred1 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

            if self.num_scales > 1:
                self.stem2 = ConvBlock(in_channels=in_channels[1], out_channels=n_reg, args=args)
                self.cls_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
                self.cls_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors * self.num_classes, bias=True, args=args)
                self.reg_conv2 = ConvBlock(in_channels=n_reg, out_channels=n_reg, args=args)
                self.reg_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=4, bias=True, args=args)
                self.obj_pred2 = SplineConvToDense(in_channels=n_reg, out_channels=self.n_anchors, bias=True, args=args)

        if self.use_image:
            self.cnn_head = CNNHead(num_classes=num_classes, strides=strides, in_channels=in_channels_cnn)

        self.use_l1 = False
        self.l1_loss = torch.nn.L1Loss(reduction="none")
        self.bcewithlog_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        self.grid_cache = None
        self.stride_cache = None
        self.cache = []

    def process_feature(self, x, stem, cls_conv, reg_conv, cls_pred, reg_pred, obj_pred, batch_size, cache):
        x = stem(x)

        cls_feat = cls_conv(shallow_copy(x))
        reg_feat = reg_conv(x)

        # we need to provide the batchsize, since sometimes it cannot be foudn from the data, especially when nodes=0
        cls_output = cls_pred(cls_feat, batch_size=batch_size)
        reg_output = reg_pred(shallow_copy(reg_feat), batch_size=batch_size)
        obj_output = obj_pred(reg_feat, batch_size=batch_size)

        return cls_output, reg_output, obj_output

    def process_dense_feature(self, x, scale_idx: int):
        """Process dense feature map with the inherited YOLOX dense head."""
        x = self.stems[scale_idx](x)
        cls_feat = self.cls_convs[scale_idx](x)
        reg_feat = self.reg_convs[scale_idx](x)

        cls_output = self.cls_preds[scale_idx](cls_feat)
        reg_output = self.reg_preds[scale_idx](reg_feat)
        obj_output = self.obj_preds[scale_idx](reg_feat)
        return cls_output, reg_output, obj_output

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        is_dense_backbone = isinstance(xin[0], torch.Tensor)

        if self.use_image:
            out_cnn = self.cnn_head(self.filter(xin), labels, imgs)
        else:
            out_cnn = None

        if self.use_image:
            batch_size = len(out_cnn["cls_output"][0])
        else:
            batch_size = xin[0].shape[0] if is_dense_backbone else self.batch_size

        for k in range(self.num_scales):
            if is_dense_backbone:
                cls_output, reg_output, obj_output = self.process_dense_feature(xin[k], k)
            else:
                if k == 0:
                    cls_output, reg_output, obj_output = self.process_feature(
                        xin[k],
                        self.stem1,
                        self.cls_conv1,
                        self.reg_conv1,
                        self.cls_pred1,
                        self.reg_pred1,
                        self.obj_pred1,
                        batch_size=batch_size,
                        cache=self.cache
                    )
                else:
                    cls_output, reg_output, obj_output = self.process_feature(
                        xin[k],
                        self.stem2,
                        self.cls_conv2,
                        self.reg_conv2,
                        self.cls_pred2,
                        self.reg_pred2,
                        self.obj_pred2,
                        batch_size=batch_size,
                        cache=self.cache
                    )

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, self.strides[k],
                    xin[k].type() if is_dense_backbone else xin[k].x.type(),
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1], device=grid.device).fill_(self.strides[k])
                )

                if self.use_l1:
                    reg_output = reg_output.view(
                        batch_size, 1, 4, reg_output.shape[-2], reg_output.shape[-1]
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            output = torch.cat(outputs, 1)
            dtype = xin[0].dtype if is_dense_backbone else xin[0].x.dtype

            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                output,
                origin_preds,
                dtype=dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)

            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=outputs.type())
            else:
                return outputs

    def collect_outputs(self, cls_output, reg_output, obj_output, k, stride_this_level, ret=None):
        if self.training:
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, output.type())
            ret['x_shifts'].append(grid[:, :, 0])
            ret['y_shifts'].append(grid[:, :, 1])
            ret['expanded_strides'].append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(output))
        else:
            output = torch.cat(
                [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
            )

        ret['outputs'].append(output)

    def decode_outputs(self, outputs, dtype):
        if self.grid_cache is None:
            self.grid_cache, self.stride_cache = init_grid_and_stride(self.hw, self.strides, dtype)

        outputs[..., :2] = (outputs[..., :2] + self.grid_cache) * self.stride_cache
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * self.stride_cache
        return outputs
