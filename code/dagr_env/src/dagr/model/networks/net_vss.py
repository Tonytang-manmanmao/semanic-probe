import torch
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.data import Data

from dagr.model.layers.ev_tgn import EV_TGN
from dagr.model.layers.pooling import Pooling
from dagr.model.layers.conv import Layer
from dagr.model.layers.components import Cartesian
from dagr.model.networks.net_img import HookModule
from dagr.model.utils import shallow_copy
from torchvision.models import resnet18, resnet34, resnet50
from dagr.model.layers.vss_dense import graph_to_dense_mean, VSSDenseStage
from dagr.model.layers.ssm_hyper import NodeSSM, AdaptiveHyperedge, HyperConv


def sampling_skip(data, image_feat):
    image_feat_at_nodes = sample_features(data, image_feat)
    return torch.cat((data.x, image_feat_at_nodes), dim=1)


def compute_pooling_at_each_layer(pooling_dim_at_output, num_layers):
    py, px = map(int, pooling_dim_at_output.split("x"))
    pooling_base = torch.tensor([1.0 / px, 1.0 / py, 1.0 / 1])
    poolings = []
    for i in range(num_layers):
        pooling = pooling_base / 2 ** (3 - i)
        pooling[-1] = 1
        poolings.append(pooling)
    poolings = torch.stack(poolings)
    return poolings


class Net(torch.nn.Module):
    def __init__(self, args, height, width):
        super().__init__()

        channels = [
            1,
            int(args.base_width * 32),
            int(args.after_pool_width * 64),
            int(args.net_stem_width * 128),
            int(args.net_stem_width * 128),
            int(args.net_stem_width * 128),
        ]

        self.out_channels_cnn = []
        if args.use_image:
            img_net = eval(args.img_net)
            self.out_channels_cnn = [256, 256]
            self.net = HookModule(
                img_net(pretrained=True),
                input_channels=3,
                height=height,
                width=width,
                feature_layers=["conv1", "layer1", "layer2", "layer3", "layer4"],
                output_layers=["layer3", "layer4"],
                feature_channels=channels[1:],
                output_channels=self.out_channels_cnn,
            )

        self.use_image = args.use_image
        self.num_scales = args.num_scales

        # ------------------------------------------------------------
        # Hybrid Graph -> Dense(VSS) backbone
        # Strictly fair version:
        #   graph_stages=k => dense starts from pool{k} scale
        # ------------------------------------------------------------
        self.use_vss_backbone = bool(getattr(args, "use_vss_backbone", False))
        self.graph_stages = int(getattr(args, "graph_stages", 2))
        self.vss_stages = int(getattr(args, "vss_stages", 2))

        if self.use_vss_backbone:
            assert self.graph_stages in [1, 2, 3], \
                f"graph_stages must be one of [1,2,3], got {self.graph_stages}"
            assert self.vss_stages >= 2, \
                f"vss_stages must be >= 2 for multi-scale detection, got {self.vss_stages}"

        output_channels = channels[1:]
        self.vss_dim = int(getattr(args, "vss_dim", output_channels[-1]))

        # ---- VSS stage depths ----
        self.vss_depths = getattr(args, "vss_depths", [1, 1])
        if isinstance(self.vss_depths, int):
            self.vss_depths = [self.vss_depths] * self.vss_stages
        elif not isinstance(self.vss_depths, (list, tuple)):
            self.vss_depths = [1] * self.vss_stages
        else:
            self.vss_depths = list(self.vss_depths)

        if len(self.vss_depths) < self.vss_stages:
            self.vss_depths = self.vss_depths + [self.vss_depths[-1]] * (self.vss_stages - len(self.vss_depths))
        elif len(self.vss_depths) > self.vss_stages:
            self.vss_depths = self.vss_depths[:self.vss_stages]

        # ---- Whether each VSS stage downsamples ----
        self.vss_downsamples = getattr(args, "vss_downsamples", [False, True])
        if isinstance(self.vss_downsamples, bool):
            self.vss_downsamples = [self.vss_downsamples] * self.vss_stages
        elif not isinstance(self.vss_downsamples, (list, tuple)):
            self.vss_downsamples = [False] * self.vss_stages
        else:
            self.vss_downsamples = list(self.vss_downsamples)

        if len(self.vss_downsamples) < self.vss_stages:
            self.vss_downsamples = self.vss_downsamples + [self.vss_downsamples[-1]] * (
                self.vss_stages - len(self.vss_downsamples)
            )
        elif len(self.vss_downsamples) > self.vss_stages:
            self.vss_downsamples = self.vss_downsamples[:self.vss_stages]

        # ---- Which dense stage outputs go to detector head ----
        default_output_indices = [self.vss_stages - 1] if self.num_scales == 1 else [0, self.vss_stages - 1]
        self.vss_output_indices = getattr(args, "vss_output_indices", default_output_indices)
        if isinstance(self.vss_output_indices, int):
            self.vss_output_indices = [self.vss_output_indices]
        else:
            self.vss_output_indices = list(self.vss_output_indices)

        if len(self.vss_output_indices) != self.num_scales:
            self.vss_output_indices = default_output_indices

        # ---- VSS block params ----
        self.vss_drop_path = float(getattr(args, "vss_drop_path", 0.0))
        self.vss_norm_layer = str(getattr(args, "vss_norm_layer", "ln2d"))

        self.vss_ssm_d_state = int(getattr(args, "vss_ssm_d_state", 16))
        self.vss_ssm_dt_rank = getattr(args, "vss_ssm_dt_rank", "auto")
        self.vss_ssm_ratio = float(getattr(args, "vss_ssm_ratio", 2.0))
        self.vss_ssm_act_layer = str(getattr(args, "vss_ssm_act_layer", "silu"))
        self.vss_ssm_conv = int(getattr(args, "vss_ssm_conv", 3))
        self.vss_ssm_conv_bias = bool(getattr(args, "vss_ssm_conv_bias", True))
        self.vss_ssm_drop_rate = float(getattr(args, "vss_ssm_drop_rate", 0.0))
        self.vss_ssm_forwardtype = str(getattr(args, "vss_ssm_forwardtype", "v05_noz"))

        self.vss_mlp_ratio = float(getattr(args, "vss_mlp_ratio", 2.0))
        self.vss_mlp_act_layer = str(getattr(args, "vss_mlp_act_layer", "gelu"))
        self.vss_mlp_drop_rate = float(getattr(args, "vss_mlp_drop_rate", 0.0))

        self.vss_gmlp = bool(getattr(args, "vss_gmlp", False))
        self.vss_use_checkpoint = bool(getattr(args, "vss_use_checkpoint", False))
        self.vss_post_norm = bool(getattr(args, "vss_post_norm", False))

        self.num_classes = dict(dsec=2, ncaltech101=100, pedro=1, gen1=3).get(args.dataset, 2)

        self.events_to_graph = EV_TGN(args)

        output_channels = channels[1:]
        if self.use_vss_backbone:
            self.out_channels = [self.vss_dim for _ in range(self.num_scales)]
        else:
            self.out_channels = output_channels[-2:]

        input_channels = channels[:-1]
        if self.use_image:
            input_channels = [input_channels[i] + self.net.feature_channels[i] for i in range(len(input_channels))]

        # parse x and y pooling dimensions at output
        poolings = compute_pooling_at_each_layer(args.pooling_dim_at_output, num_layers=4)
        self.poolings = poolings
        max_vals_for_cartesian = 2 * poolings[:, :2].max(-1).values

        # keep detector strides consistent with final 2 output scales
        self.strides = torch.ceil(poolings[-2:, 1] * height).numpy().astype("int32").tolist()
        self.strides = self.strides[-self.num_scales:]

        effective_radius = 2 * float(int(args.radius * width + 2) / width)
        self.edge_attrs = Cartesian(norm=True, cat=False, max_value=effective_radius)

        # ------------------------------------------------------------
        # Graph stages
        # ------------------------------------------------------------
        self.conv_block1 = Layer(2 + input_channels[0], output_channels[0], args=args)

        cart1 = T.Cartesian(norm=True, cat=False, max_value=2 * effective_radius)
        self.pool1 = Pooling(
            poolings[0], width=width, height=height, batch_size=args.batch_size,
            transform=cart1, aggr=args.pooling_aggr, keep_temporal_ordering=args.keep_temporal_ordering
        )

        if not self.use_vss_backbone or self.graph_stages >= 2:
            self.layer2 = Layer(input_channels[1] + 2, output_channels[1], args=args)
            cart2 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[1])
            self.pool2 = Pooling(
                poolings[1], width=width, height=height, batch_size=args.batch_size,
                transform=cart2, aggr=args.pooling_aggr, keep_temporal_ordering=args.keep_temporal_ordering
            )

        if not self.use_vss_backbone or self.graph_stages >= 3:
            self.layer3 = Layer(input_channels[2] + 2, output_channels[2], args=args)
            cart3 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[2])
            self.pool3 = Pooling(
                poolings[2], width=width, height=height, batch_size=args.batch_size,
                transform=cart3, aggr=args.pooling_aggr, keep_temporal_ordering=args.keep_temporal_ordering
            )

        if not self.use_vss_backbone:
            self.layer4 = Layer(input_channels[3] + 2, output_channels[3], args=args)
            cart4 = T.Cartesian(norm=True, cat=False, max_value=max_vals_for_cartesian[3])
            self.pool4 = Pooling(
                poolings[3], width=width, height=height, batch_size=args.batch_size,
                transform=cart4, aggr='mean', keep_temporal_ordering=args.keep_temporal_ordering
            )
            self.layer5 = Layer(input_channels[4] + 2, output_channels[4], args=args)

        # ------------------------------------------------------------
        # Dense VSS stages
        # Strictly fair:
        #   graph_stages=1 -> dense starts from pool1 scale
        #   graph_stages=2 -> dense starts from pool2 scale
        #   graph_stages=3 -> dense starts from pool3 scale
        # ------------------------------------------------------------
        if self.use_vss_backbone:
            # dense base pooling bound to graph_stages
            self.vss_base_pooling = poolings[self.graph_stages - 1].clone()

            # stage-wise poolings after each dense stage
            cur_pool = self.vss_base_pooling.clone()
            self.vss_stage_poolings = []
            for i in range(self.vss_stages):
                if bool(self.vss_downsamples[i]):
                    cur_pool = cur_pool.clone()
                    cur_pool[0] = cur_pool[0] * 2.0
                    cur_pool[1] = cur_pool[1] * 2.0
                self.vss_stage_poolings.append(cur_pool.clone())

            # graph_to_dense_mean appends 2 coord channels
            dense_in_channels = output_channels[self.graph_stages - 1] + 2

            self.vss_dense_stages = nn.ModuleList()
            in_ch = dense_in_channels
            for i in range(self.vss_stages):
                stage = VSSDenseStage(
                    in_channels=in_ch,
                    out_channels=self.vss_dim,
                    depth=int(self.vss_depths[i]),
                    downsample=bool(self.vss_downsamples[i]),
                    drop_path=self.vss_drop_path,
                    norm_layer=self.vss_norm_layer,
                    ssm_d_state=self.vss_ssm_d_state,
                    ssm_dt_rank=self.vss_ssm_dt_rank,
                    ssm_ratio=self.vss_ssm_ratio,
                    ssm_act_layer=self.vss_ssm_act_layer,
                    ssm_conv=self.vss_ssm_conv,
                    ssm_conv_bias=self.vss_ssm_conv_bias,
                    ssm_drop_rate=self.vss_ssm_drop_rate,
                    ssm_forwardtype=self.vss_ssm_forwardtype,
                    mlp_ratio=self.vss_mlp_ratio,
                    mlp_act_layer=self.vss_mlp_act_layer,
                    mlp_drop_rate=self.vss_mlp_drop_rate,
                    gmlp=self.vss_gmlp,
                    use_checkpoint=self.vss_use_checkpoint,
                    post_norm=self.vss_post_norm,
                )
                self.vss_dense_stages.append(stage)
                in_ch = self.vss_dim

            selected_poolings = [self.vss_stage_poolings[i] for i in self.vss_output_indices]
            self.strides = [int(torch.ceil(p[1] * height).item()) for p in selected_poolings]

        self.cache = []

        # SSM -> HyperGraph cross-guidance
        self.use_ssm_hyper = bool(getattr(args, 'use_ssm_hyper', False))
        # Probe variable: which feature drives the hyperedge incidence matrix H.
        self.hyperedge_source = str(getattr(args, 'hyperedge_source', 'ssm'))
        self._last_H = None
        self._last_h_ssm = None
        self._last_h_src = None
        if self.use_ssm_hyper:
            if self.use_vss_backbone:
                self.ssm_hyper_graph_stage = self.graph_stages
            else:
                # Pure graph tail enters at stage 3 and then forks to outputs.
                self.ssm_hyper_graph_stage = 3
            node_dim = output_channels[self.ssm_hyper_graph_stage - 1]
            self.num_hyperedges = int(getattr(args, 'num_hyperedges', 16))
            self.node_ssm = NodeSSM(node_dim, d_state=int(getattr(args, 'ssm_hyper_state', 16)))
            # ``hyperedge_source`` in {'ssm', 'semantic', 'motion', 'joint'} is the
            # ONLY variable across the four probe runs; it fixes H's input width.
            hyperedge_source_dims = {
                'ssm': node_dim,
                'semantic': node_dim,
                'motion': 3,
                'joint': 2 * node_dim + 3,
            }
            if self.hyperedge_source not in hyperedge_source_dims:
                raise ValueError(
                    f"unknown hyperedge_source {self.hyperedge_source!r}; "
                    f"expected one of {sorted(hyperedge_source_dims)}"
                )
            self.adaptive_hyperedge = AdaptiveHyperedge(
                hyperedge_source_dims[self.hyperedge_source], self.num_hyperedges
            )
            self.hyper_conv = HyperConv(node_dim)

    def get_output_sizes(self):
        if self.use_vss_backbone:
            selected_poolings = [self.vss_stage_poolings[i][:2] for i in self.vss_output_indices]
            output_sizes = [(1 / p + 1e-3).cpu().int().numpy().tolist()[::-1] for p in selected_poolings]
        else:
            poolings = [self.pool3.voxel_size[:2], self.pool4.voxel_size[:2]]
            output_sizes = [(1 / p + 1e-3).cpu().int().numpy().tolist()[::-1] for p in poolings]
        return output_sizes

    # ------------------------------------------------------------
    # Graph frontend helper stages
    # ------------------------------------------------------------
    def _run_graph_stage1(self, data, image_feat=None):
        if self.use_image:
            data.x = sampling_skip(data, image_feat[0].detach())
            data.skipped = True
            data.num_image_channels = image_feat[0].shape[1]

        data = self.edge_attrs(data)
        data.edge_attr = torch.clamp(data.edge_attr, min=0, max=1)
        rel_delta = data.pos[:, :2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.conv_block1(data)

        if self.use_image:
            data.x = sampling_skip(data, image_feat[1].detach())

        data = self.pool1(data)
        return data

    def _run_graph_stage2(self, data, image_feat=None):
        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[1].shape[1]

        rel_delta = data.pos[:, :2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer2(data)

        if self.use_image:
            data.x = sampling_skip(data, image_feat[2].detach())

        data = self.pool2(data)
        return data

    def _run_graph_stage3(self, data, image_feat=None):
        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[2].shape[1]

        rel_delta = data.pos[:, :2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer3(data)

        if self.use_image:
            data.x = sampling_skip(data, image_feat[3].detach())

        data = self.pool3(data)
        return data

    def _run_graph_stage4_to_outputs(self, data, image_feat=None):
        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[3].shape[1]

        rel_delta = data.pos[:, :2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer4(data)

        out3 = shallow_copy(data)
        out3.pooling = self.pool3.voxel_size[:3]

        if self.use_image:
            data.x = sampling_skip(data, image_feat[4].detach())

        data = self.pool4(data)

        if self.use_image:
            data.skipped = True
            data.num_image_channels = image_feat[4].shape[1]

        rel_delta = data.pos[:, :2]
        data.x = torch.cat((data.x, rel_delta), dim=1)
        data = self.layer5(data)

        out4 = data
        out4.pooling = self.pool4.voxel_size[:3]
        return [out3, out4]

    # ------------------------------------------------------------
    # Dense VSS backend
    # ------------------------------------------------------------
    def _run_vss_backend(self, data, image_outputs=None):
        if hasattr(data, "num_graphs"):
            batch_size = int(data.num_graphs)
        else:
            batch_size = int(data.batch.max().item()) + 1 if data.batch is not None else 1

        dense_in = graph_to_dense_mean(
            data,
            pooling=self.vss_base_pooling,
            batch_size=batch_size,
            add_coords=True,
        )

        x_dense = dense_in
        dense_stage_outputs = []
        for stage in self.vss_dense_stages:
            x_dense = stage(x_dense)
            dense_stage_outputs.append(x_dense)

        selected_outputs = [dense_stage_outputs[i] for i in self.vss_output_indices]

        if self.use_image:
            return selected_outputs[-self.num_scales:], image_outputs[-self.num_scales:]
        return selected_outputs[-self.num_scales:]

    def _run_ssm_hyper(self, data):
        n_nodes = data.x.shape[0]
        if n_nodes == 0:
            self._last_H = data.x.new_zeros((0, self.num_hyperedges))
            self._last_h_ssm = data.x
            self._last_h_src = data.x
            return data

        pos = getattr(data, "pos", None)
        if pos is not None and pos.shape[1] >= 3:
            t = pos[:, 2].reshape(-1)
            motion = pos[:, :3].to(data.x.dtype)
        else:
            t = getattr(data, "t", None)
            if t is None:
                t = torch.zeros(n_nodes, dtype=torch.float32, device=data.x.device)
            else:
                t = t.reshape(-1)[:n_nodes]
            # physical prior fallback: position unknown -> (0, 0, t)
            motion = torch.stack(
                (
                    data.x.new_zeros(n_nodes),
                    data.x.new_zeros(n_nodes),
                    t.to(data.x.dtype),
                ),
                dim=-1,
            )

        batch = getattr(data, "batch", None)
        h_ssm = self.node_ssm(data.x, t, batch=batch)

        # ------------------------------------------------------------
        # Probe variable: what drives the hyperedge incidence matrix H.
        # This is the ONLY difference between the four probe runs.
        # ------------------------------------------------------------
        if self.hyperedge_source == 'ssm':
            h_src = h_ssm
        elif self.hyperedge_source == 'semantic':
            h_src = data.x
        elif self.hyperedge_source == 'motion':
            h_src = motion
        else:  # 'joint'
            h_src = torch.cat((h_ssm, data.x, motion), dim=-1)

        H = self.adaptive_hyperedge(h_src)
        h_hyper = self.hyper_conv(data.x, H)
        data.x = h_ssm + h_hyper
        self._last_H = H
        self._last_h_ssm = h_ssm
        self._last_h_src = h_src
        return data

    def forward(self, data: Data, reset=True):
        if self.use_image:
            image_feat, image_outputs = self.net(data.image)

        if hasattr(data, 'reset'):
            reset = data.reset

        data = self.events_to_graph(data, reset=reset)

        # ----------------------------
        # Graph frontend
        # ----------------------------
        data = self._run_graph_stage1(data, image_feat if self.use_image else None)

        if self.use_vss_backbone and self.graph_stages == 1:
            if self.use_ssm_hyper:
                data = self._run_ssm_hyper(data)
            return self._run_vss_backend(data, image_outputs if self.use_image else None)

        data = self._run_graph_stage2(data, image_feat if self.use_image else None)

        if self.use_vss_backbone and self.graph_stages == 2:
            if self.use_ssm_hyper:
                data = self._run_ssm_hyper(data)
            return self._run_vss_backend(data, image_outputs if self.use_image else None)

        data = self._run_graph_stage3(data, image_feat if self.use_image else None)

        if self.use_vss_backbone and self.graph_stages == 3:
            if self.use_ssm_hyper:
                data = self._run_ssm_hyper(data)
            return self._run_vss_backend(data, image_outputs if self.use_image else None)

        # ----------------------------
        # Original pure-graph tail
        # ----------------------------
        if self.use_ssm_hyper:
            data = self._run_ssm_hyper(data)
        output = self._run_graph_stage4_to_outputs(data, image_feat if self.use_image else None)

        if self.use_image:
            return output[-self.num_scales:], image_outputs[-self.num_scales:]
        return output[-self.num_scales:]


def sample_features(data, image_feat, image_sample_mode="bilinear"):
    if data.batch is None or len(data.batch) != len(data.pos):
        data.batch = torch.zeros(len(data.pos), dtype=torch.long, device=data.x.device)
    return _sample_features(
        data.pos[:, 0] * data.width[0],
        data.pos[:, 1] * data.height[0],
        data.batch.float(),
        image_feat,
        data.width[0],
        data.height[0],
        image_feat.shape[0],
        image_sample_mode,
    )


def _sample_features(x, y, b, image_feat, width, height, batch_size, image_sample_mode):
    x = 2 * x / (width - 1) - 1
    y = 2 * y / (height - 1) - 1

    batch_size = batch_size if batch_size > 1 else 2
    b = 2 * b / (batch_size - 1) - 1

    grid = torch.stack((x, y, b), dim=-1).view(1, 1, 1, -1, 3)  # N x D_out x H_out x W_out x 3
    image_feat = image_feat.permute(1, 0, 2, 3).unsqueeze(0)  # N x C x D x H x W

    image_feat_sampled = torch.nn.functional.grid_sample(
        image_feat,
        grid=grid,
        mode=image_sample_mode,
        align_corners=True,
    )

    image_feat_sampled = image_feat_sampled.view(image_feat.shape[1], -1).t()

    return image_feat_sampled
