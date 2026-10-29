# DA3 滑窗重叠推理节点: 解决多批 multiview 深度的批次间闪烁
# 逻辑: 每轮推理 window_size 帧, 其中前 overlap 帧复用上一轮的最后 overlap 帧;
#       用重叠帧的深度做最小二乘尺度/偏移对齐后拼接, 保证全局深度尺度一致。
import torch
import comfy.model_management as mm
from comfy.ldm.depth_anything_3 import preprocess as da3_preprocess


class DA3SlideWindowInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "da3_model": ("DA3_MODEL",),
                "image": ("IMAGE",),
                "resolution": ("INT", {"default": 504, "min": 140, "max": 2520, "step": 14}),
                "window_size": ("INT", {"default": 24, "min": 4, "max": 128, "step": 1,
                                        "tooltip": "每轮 multiview 推理的帧数"}),
                "overlap": ("INT", {"default": 6, "min": 1, "max": 64, "step": 1,
                                    "tooltip": "重叠帧数: 上一轮最后 N 帧作为下一轮开头"}),
                "resize_method": (["upper_bound_resize", "lower_bound_resize"],),
                "ref_view_strategy": (["saddle_balanced", "saddle_sim_range", "first", "middle"],),
                "pose_method": (["cam_dec", "ray_pose"],),
            }
        }

    RETURN_TYPES = ("DA3_GEOMETRY",)
    FUNCTION = "run"
    CATEGORY = "image/geometry estimation"
    DISPLAY_NAME = "DA3 Slide Window Inference"

    @staticmethod
    def _fit_affine(cur, ref, max_samples=2_000_000):
        """最小二乘拟合 a*cur+b ~= ref (展平子采样)。失败时返回 (1.0, 0.0)。"""
        cur = cur.flatten().float()
        ref = ref.flatten().float()
        n = cur.numel()
        if n > max_samples:
            idx = torch.arange(0, n, n // max_samples)
            cur, ref = cur[idx], ref[idx]
        cur_mean, ref_mean = cur.mean(), ref.mean()
        var = (cur * cur).mean() - cur_mean * cur_mean
        if not torch.isfinite(var) or var < 1e-12:
            return 1.0, 0.0
        a = ((cur * ref).mean() - cur_mean * ref_mean) / var
        b = ref_mean - a * cur_mean
        if not torch.isfinite(a) or not torch.isfinite(b) or a <= 0:
            return 1.0, 0.0
        return float(a), float(b)

    def run(self, da3_model, image, resolution, window_size, overlap,
            resize_method, ref_view_strategy, pose_method):
        assert image.ndim == 4 and image.shape[-1] == 3
        S, H, W, _ = image.shape
        overlap = min(overlap, window_size - 2)
        stride = window_size - overlap

        # 生成窗口: [0,window), [stride,stride+window), ...; 尾部不足 stride 时并入上一窗
        windows = []
        start = 0
        while True:
            end = min(start + window_size, S)
            if S - end < stride or end >= S:
                windows.append((start, end))
                break
            windows.append((start, end))
            start += stride

        mm.load_model_gpu(da3_model)
        diffusion = da3_model.model.diffusion_model
        device = mm.get_torch_device()
        dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32
        use_ray_pose = (pose_method == "ray_pose")

        depth_parts = []
        sky_parts, conf_parts = [], []
        has_sky = has_conf = False
        prev_overlap_depth = None  # 上一轮已对齐的重叠帧深度

        for (s, e) in windows:
            frames = image[s:e].to(device)
            x = da3_preprocess.preprocess_image(frames, process_res=resolution, method=resize_method)
            x = x.to(dtype=dtype).unsqueeze(0)
            with torch.no_grad():
                out = diffusion(x, use_ray_pose=use_ray_pose, ref_view_strategy=ref_view_strategy)

            d = torch.nn.functional.interpolate(
                out["depth"].float().unsqueeze(1), size=(H, W),
                mode="bilinear", align_corners=False,
            ).squeeze(1).cpu()

            if prev_overlap_depth is not None:
                n_ov = min(prev_overlap_depth.shape[0], d.shape[0])
                a, b = self._fit_affine(d[:n_ov], prev_overlap_depth[-n_ov:])
                d = d * a + b

            if not depth_parts:
                depth_parts.append(d)                       # 首窗全部保留
            else:
                depth_parts.append(d[overlap:])             # 丢弃与上一窗重复的重叠帧
            prev_overlap_depth = d[-overlap:]

            if "sky" in out:
                has_sky = True
                sky_parts.append(torch.nn.functional.interpolate(
                    out["sky"].unsqueeze(1).float(), size=(H, W),
                    mode="bilinear", align_corners=False).squeeze(1).cpu())
            if "depth_conf" in out:
                has_conf = True
                conf_parts.append(torch.nn.functional.interpolate(
                    out["depth_conf"].unsqueeze(1).float(), size=(H, W),
                    mode="bilinear", align_corners=False).squeeze(1).cpu())

        depth = torch.cat(depth_parts, dim=0)[:S].contiguous()
        geometry = {
            "depth": depth,
            "mode": "multiview",
            "extrinsics": torch.eye(4)[None, None].expand(1, S, 4, 4).clone(),
            "intrinsics": torch.eye(3)[None, None].expand(1, S, 3, 3).clone(),
        }
        if has_sky and sum(p.shape[0] for p in sky_parts) >= S:
            geometry["sky"] = torch.cat(sky_parts, dim=0)[:S].contiguous()
        if has_conf and sum(p.shape[0] for p in conf_parts) >= S:
            geometry["confidence"] = torch.cat(conf_parts, dim=0)[:S].contiguous()
        return (geometry,)


NODE_CLASS_MAPPINGS = {"DA3SlideWindowInference": DA3SlideWindowInference}
NODE_DISPLAY_NAME_MAPPINGS = {"DA3SlideWindowInference": "DA3 Slide Window Inference"}
