import os
import json
import time
import tempfile
import numpy as np
import torch

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

try:
    from folder_paths import folder_names_and_paths
    import folder_paths
    MODEL_ROOTS = folder_names_and_paths.get("model", [[], []])[0]
except Exception:
    MODEL_ROOTS = []

# LUT Model

class LUT:
    def __init__(self, title: str = "", domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0),
                 table_1d: np.ndarray | None = None, table_3d: np.ndarray | None = None):
        self.title = title
        self.domain_min = np.array(domain_min, dtype=np.float32)
        self.domain_max = np.array(domain_max, dtype=np.float32)
        self.table_1d = table_1d
        self.table_3d = table_3d

    def size(self) -> int:
        if self.table_3d is not None:
            return int(self.table_3d.shape[0])
        if self.table_1d is not None:
            return int(self.table_1d.shape[0])
        return 0

# LUT Loader

class LUTLoader:
    BUILTIN_PRESETS = [
        ("Cinematic",     (0.0,  1.15,0.90, 0.35,0.95, 0.10, -0.05)),
        ("Vibrant",       (0.1,  1.10,1.25, 0.30,0.95, 0.05, 0.00)),
        ("Desaturated",   (0.0,  1.05,0.65,-0.10,1.05, 0.00, 0.00)),
        ("High Contrast", (0.0,  1.35,1.00, 0.10,0.95, 0.00, 0.00)),
        ("Soft",          (-0.05,0.90,0.95,-0.05,1.05, 0.00, 0.00)),
    ]

    @staticmethod
    def get_lut_dirs() -> list[Path]:
        dirs: list[Path] = []
        seen: set[Path] = set()
        try:
            for key, (paths, _exts) in folder_names_and_paths.items():
                for p in paths:
                    pp = Path(p)
                    models_dir = pp.parent
                    if models_dir.name != "models":
                        for ancestor in pp.parents:
                            if ancestor.name == "models":
                                models_dir = ancestor
                                break
                    if models_dir.name != "models":
                        continue
                    lut_dir = models_dir / "LUT"
                    if lut_dir.exists() and lut_dir.is_dir() and lut_dir not in seen:
                        seen.add(lut_dir)
                        dirs.append(lut_dir)
        except Exception:
            print("[WASLUT] Unable to load LUT directory.", flush=True)
            pass

        return dirs

    @staticmethod
    def discover_cube_files() -> list[Path]:
        out: list[Path] = []
        for d in LUTLoader.get_lut_dirs():
            try:
                for p in d.iterdir():
                    if p.is_file() and p.suffix.lower() == ".cube":
                        out.append(p)
            except Exception:
                continue
        return sorted(out, key=lambda p: p.name.lower())

    @staticmethod
    def luts_signature() -> str:
        items = []
        for p in LUTLoader.discover_cube_files():
            try:
                st = p.stat()
                items.append((str(p.resolve()), int(st.st_mtime)))
            except Exception:
                items.append((str(p), 0))
        return json.dumps(items, sort_keys=True)

    @staticmethod
    def load_cube(path: Path) -> LUT:
        title = path.stem
        domain_min = (0.0, 0.0, 0.0)
        domain_max = (1.0, 1.0, 1.0)
        size_1d: int | None = None
        size_3d: int | None = None
        data: list[tuple[float, float, float]] = []

        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                up = s.upper()
                if up.startswith("TITLE"):
                    q = s.find('"')
                    if q >= 0:
                        title = s[q+1:s.rfind('"')].strip()
                    else:
                        title = s
                    continue
                if up.startswith("DOMAIN_MIN"):
                    parts = s.split()
                    if len(parts) >= 4:
                        domain_min = (float(parts[1]), float(parts[2]), float(parts[3]))
                    continue
                if up.startswith("DOMAIN_MAX"):
                    parts = s.split()
                    if len(parts) >= 4:
                        domain_max = (float(parts[1]), float(parts[2]), float(parts[3]))
                    continue
                if up.startswith("LUT_1D_SIZE"):
                    parts = s.split()
                    size_1d = int(parts[1])
                    continue
                if up.startswith("LUT_3D_SIZE"):
                    parts = s.split()
                    size_3d = int(parts[1])
                    continue
                parts = s.split()
                if len(parts) >= 3:
                    data.append((float(parts[0]), float(parts[1]), float(parts[2])))

        if size_3d is not None:
            expected = size_3d ** 3
            if len(data) != expected:
                raise ValueError(f"{path.name}: expected {expected}, got {len(data)}")
            arr = np.asarray(data, dtype=np.float32).reshape(size_3d, size_3d, size_3d, 3)
            return LUT(title, domain_min, domain_max, None, arr)

        if size_1d is not None:
            expected = size_1d
            if len(data) != expected:
                raise ValueError(f"{path.name}: expected {expected}, got {len(data)}")
            arr = np.asarray(data, dtype=np.float32).reshape(size_1d, 3)
            return LUT(title, domain_min, domain_max, arr, None)

        n = len(data)
        k = round(n ** (1 / 3))
        if k * k * k == n and k > 1:
            arr = np.asarray(data, dtype=np.float32).reshape(k, k, k, 3)
            return LUT(title, domain_min, domain_max, None, arr)

        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[1] == 3:
            return LUT(title, domain_min, domain_max, arr, None)

        raise ValueError(f"{path.name}: invalid .cube")

    @staticmethod
    def save_cube(path: Path, lut: 'LUT') -> None:
        # Ensure 3D table
        if lut.table_3d is None:
            raise ValueError("save_cube expects a 3D LUT table")
        table = lut.table_3d
        N = int(table.shape[0])
        dom_min = np.asarray(lut.domain_min, dtype=np.float32).tolist()
        dom_max = np.asarray(lut.domain_max, dtype=np.float32).tolist()
        with path.open("w", encoding="utf-8") as f:
            f.write(f"TITLE \"{lut.title or path.stem}\"\n")
            f.write(f"LUT_3D_SIZE {N}\n")
            f.write(f"DOMAIN_MIN {dom_min[0]:.6f} {dom_min[1]:.6f} {dom_min[2]:.6f}\n")
            f.write(f"DOMAIN_MAX {dom_max[0]:.6f} {dom_max[1]:.6f} {dom_max[2]:.6f}\n")
            # Write values in r-fastest order matching our reader's reshape
            for r in range(N):
                for g in range(N):
                    for b in range(N):
                        R, G, B = table[r, g, b]
                        f.write(f"{float(R):.6f} {float(G):.6f} {float(B):.6f}\n")

    @staticmethod
    def synthesize_builtin_lut(name: str, size: int = 33) -> LUT:
        params = None
        for n, p in LUTLoader.BUILTIN_PRESETS:
            if n == name:
                params = p
                break
        if params is None:
            raise ValueError("Unknown builtin")
        ev, con, sat, vib, gam, tmp, tnt = params
        grid = torch.linspace(0, 1, steps=size)
        rr, gg, bb = torch.meshgrid(grid, grid, grid, indexing="ij")
        cube = torch.stack([rr, gg, bb], dim=-1).unsqueeze(0).to(torch.float32)
        x = cube
        x = WASLUT.apply_exposure(x, ev)
        x = WASLUT.apply_contrast(x, con)
        x = WASLUT.apply_saturation(x, sat)
        x = WASLUT.apply_vibrance(x, vib)
        x = WASLUT.apply_white_balance(x, tmp, tnt)
        x = WASLUT.apply_gamma(x, gam)
        table = x.squeeze(0).clamp(0, 1).cpu().numpy().astype(np.float32)
        return LUT(name, (0, 0, 0), (1, 1, 1), None, table)

# LUT Class

class WASLUT:
    @staticmethod
    def luma(x: torch.Tensor) -> torch.Tensor:
        w = torch.tensor([0.2126, 0.7152, 0.0722], dtype=x.dtype, device=x.device)
        return (x * w.view(1, 1, 1, 3)).sum(dim=-1, keepdim=True)

    @staticmethod
    def apply_exposure(x: torch.Tensor, ev: float) -> torch.Tensor:
        if ev == 0.0:
            return x
        return x * (2.0 ** ev)

    @staticmethod
    def apply_contrast(x: torch.Tensor, c: float) -> torch.Tensor:
        if abs(c - 1.0) < 1e-6:
            return x
        return (x - 0.5) * c + 0.5

    @staticmethod
    def apply_saturation(x: torch.Tensor, s: float) -> torch.Tensor:
        if abs(s - 1.0) < 1e-6:
            return x
        l = WASLUT.luma(x)
        return l + (x - l) * s

    @staticmethod
    def estimate_saturation(x: torch.Tensor) -> torch.Tensor:
        m = x.mean(dim=-1, keepdim=True)
        return (x - m).abs().mean(dim=-1, keepdim=True)

    @staticmethod
    def apply_vibrance(x: torch.Tensor, v: float) -> torch.Tensor:
        if abs(v) < 1e-6:
            return x
        sat = WASLUT.estimate_saturation(x).clamp(0, 1)
        factor = 1.0 + v * (1.0 - sat)
        l = WASLUT.luma(x)
        return l + (x - l) * factor

    @staticmethod
    def apply_gamma(x: torch.Tensor, g: float) -> torch.Tensor:
        if abs(g - 1.0) < 1e-6:
            return x
        x = x.clamp(0.0, 1.0)
        return torch.pow(x, 1.0 / max(g, 1e-6))

    @staticmethod
    def apply_white_balance(x: torch.Tensor, temp: float, tint: float) -> torch.Tensor:
        r_gain = 1.0 + 0.10 * temp - 0.10 * tint
        g_gain = 1.0 + 0.10 * tint
        b_gain = 1.0 - 0.10 * temp - 0.10 * tint
        gains = torch.tensor([r_gain, g_gain, b_gain], dtype=x.dtype, device=x.device)
        return x * gains.view(1, 1, 1, 3)

    @staticmethod
    def apply_color_balance(x: torch.Tensor, r_bal: float, g_bal: float, b_bal: float) -> torch.Tensor:
        """
        Per-channel color balance. Inputs are [-1, 1], where 0.0 is no change.
        Implemented as multiplicative gains: gain = 1 + balance.
        """
        r_gain = 1.0 + r_bal
        g_gain = 1.0 + g_bal
        b_gain = 1.0 + b_bal
        gains = torch.tensor([r_gain, g_gain, b_gain], dtype=x.dtype, device=x.device)
        return x * gains.view(1, 1, 1, 3)

    @staticmethod
    def convert_to_3d(lut: LUT, size: int) -> LUT:
        if lut.table_3d is not None and lut.table_3d.shape[0] == size:
            return lut
        if lut.table_3d is not None and lut.table_3d.shape[0] != size:
            src = torch.from_numpy(lut.table_3d).to(torch.float32)
            grid = torch.linspace(0, 1, steps=size)
            rr, gg, bb = torch.meshgrid(grid, grid, grid, indexing="ij")
            pos = torch.stack([rr, gg, bb], -1) * (src.shape[0] - 1)
            i0 = torch.floor(pos).to(torch.long).clamp(0, src.shape[0] - 1)
            i1 = torch.clamp(i0 + 1, max=src.shape[0] - 1)
            d = (pos - i0.to(pos.dtype)).clamp(0, 1)
            r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
            r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
            dr, dg, db = d[..., 0], d[..., 1], d[..., 2]

            def samp(rr, gg, bb): return src[rr, gg, bb]

            c000 = samp(r0, g0, b0)
            c100 = samp(r1, g0, b0)
            c010 = samp(r0, g1, b0)
            c110 = samp(r1, g1, b0)
            c001 = samp(r0, g0, b1)
            c101 = samp(r1, g0, b1)
            c011 = samp(r0, g1, b1)
            c111 = samp(r1, g1, b1)

            c00 = c000 * (1 - dr)[..., None] + c100 * dr[..., None]
            c01 = c001 * (1 - dr)[..., None] + c101 * dr[..., None]
            c10 = c010 * (1 - dr)[..., None] + c110 * dr[..., None]
            c11 = c011 * (1 - dr)[..., None] + c111 * dr[..., None]

            c0 = c00 * (1 - dg)[..., None] + c10 * dg[..., None]
            c1 = c01 * (1 - dg)[..., None] + c11 * dg[..., None]

            out = c0 * (1 - db)[..., None] + c1 * db[..., None]
            return LUT(lut.title, lut.domain_min, lut.domain_max, None, out.numpy().astype(np.float32))

        if lut.table_1d is not None:
            N = lut.table_1d.shape[0]
            grid = torch.linspace(0, 1, steps=size)
            rr, gg, bb = torch.meshgrid(grid, grid, grid, indexing="ij")
            r_idx = (rr * (N - 1)).clamp(0, N - 1)
            g_idx = (gg * (N - 1)).clamp(0, N - 1)
            b_idx = (bb * (N - 1)).clamp(0, N - 1)
            table = torch.from_numpy(lut.table_1d).to(torch.float32)

            def sample(idx, ch):
                i0 = torch.floor(idx).to(torch.long)
                i1 = torch.clamp(i0 + 1, max=N - 1)
                t = (idx - i0.to(idx.dtype)).clamp(0, 1)
                v0 = table[i0, ch]
                v1 = table[i1, ch]
                return v0 * (1 - t) + v1 * t

            out = torch.stack([sample(r_idx, 0), sample(g_idx, 1), sample(b_idx, 2)], -1)
            return LUT(lut.title, lut.domain_min, lut.domain_max, None, out.numpy().astype(np.float32))

        raise ValueError("Empty LUT")

    @staticmethod
    def apply_lut_3d(image: torch.Tensor, table: np.ndarray,
                     domain_min: np.ndarray, domain_max: np.ndarray) -> torch.Tensor:
        N = table.shape[0]
        lut = torch.from_numpy(table).to(image.device, dtype=image.dtype)
        dom_min = torch.tensor(domain_min, device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
        dom_max = torch.tensor(domain_max, device=image.device, dtype=image.dtype).view(1, 1, 1, 3)
        x = (image - dom_min) / torch.clamp(dom_max - dom_min, min=1e-8)
        x = x.clamp(0.0, 1.0)

        pos = x * (N - 1)
        i0 = torch.floor(pos).long().clamp(0, N - 1)
        i1 = torch.clamp(i0 + 1, max=N - 1)
        d = (pos - i0.to(pos.dtype)).clamp(0, 1)

        r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
        r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
        dr, dg, db = d[..., 0], d[..., 1], d[..., 2]

        def samp(rr, gg, bb): return lut[rr, gg, bb]

        c000 = samp(r0, g0, b0)
        c100 = samp(r1, g0, b0)
        c010 = samp(r0, g1, b0)
        c110 = samp(r1, g1, b0)
        c001 = samp(r0, g0, b1)
        c101 = samp(r1, g0, b1)
        c011 = samp(r0, g1, b1)
        c111 = samp(r1, g1, b1)

        c00 = c000 * (1 - dr)[..., None] + c100 * dr[..., None]
        c01 = c001 * (1 - dr)[..., None] + c101 * dr[..., None]
        c10 = c010 * (1 - dr)[..., None] + c110 * dr[..., None]
        c11 = c011 * (1 - dr)[..., None] + c111 * dr[..., None]

        c0 = c00 * (1 - dg)[..., None] + c10 * dg[..., None]
        c1 = c01 * (1 - dg)[..., None] + c11 * dg[..., None]

        out = c0 * (1 - db)[..., None] + c1 * db[..., None]
        return out.clamp(0.0, 1.0)

# RGB Parade Class

class WaveformScope:
    @staticmethod
    def stats_tensor(ch: torch.Tensor) -> tuple[float, float, float, float, float]:
        v = ch.flatten()
        return (
            float(v.min().item()),
            float(v.max().item()),
            float(v.mean().item()),
            float(v.std(unbiased=False).item()),
            float(v.median().item()),
        )

    @staticmethod
    def _font():
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    @staticmethod
    def _big_font():
        # Try to load a larger truetype font for better legibility; fallback to default
        # Common fonts to try across platforms
        candidates = [
            "DejaVuSansMono.ttf",
            "DejaVuSans.ttf",
            "Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for path in candidates:
            try:
                return ImageFont.truetype(path, size=14)
            except Exception:
                continue
        return WaveformScope._font()

    @staticmethod
    def make_waveform_gray(ch_gray: np.ndarray, out_h: int) -> np.ndarray:
        h, w = ch_gray.shape
        out_h = max(int(out_h), 128)
        idx = np.clip((ch_gray * (out_h - 1)).astype(np.int32), 0, out_h - 1)
        wf = np.zeros((out_h, w), dtype=np.float32)
        for x in range(w):
            counts = np.bincount(idx[:, x], minlength=out_h).astype(np.float32)
            if out_h >= 3:
                counts[1:-1] = counts[1:-1] * 0.5 + (counts[:-2] + counts[2:]) * 0.25
            wf[:, x] = counts
        wf = np.log1p(wf)
        m = wf.max()
        if m > 0:
            wf /= m
        wf = wf[::-1, :]
        return wf

    @staticmethod
    def add_grid_with_labels(base: Image.Image, left_pad: int = 56) -> Image.Image:
        w, h = base.size
        canvas = Image.new("RGB", (w + left_pad, h), (0, 0, 0))
        canvas.paste(base, (left_pad, 0))
        draw = ImageDraw.Draw(canvas)
        font = WaveformScope._font()
        for ire in (0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100):
            y = int(round(h - 1 - (ire / 100.0) * (h - 1)))
            draw.line([(left_pad, y), (left_pad + w - 1, y)], fill=(64, 64, 64), width=1)
            draw.text((6, max(0, y - 6)), f"{ire:g}", fill=(200, 200, 200), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        return canvas

    @staticmethod
    def compose_waveform_panel(wf: np.ndarray, color: str, stats_text: str,
                               pad: int = 36, left_pad: int = 56) -> np.ndarray:
        if color == "red":
            rgb = np.stack([wf, np.zeros_like(wf), np.zeros_like(wf)], -1)
        elif color == "green":
            rgb = np.stack([np.zeros_like(wf), wf, np.zeros_like(wf)], -1)
        else:
            rgb = np.stack([np.zeros_like(wf), np.zeros_like(wf), wf], -1)
        rgb = (rgb * 255.0 + 0.5).astype(np.uint8)
        panel = Image.fromarray(rgb)
        panel = WaveformScope.add_grid_with_labels(panel, left_pad=left_pad)
        W, H = panel.size
        canvas = Image.new("RGB", (W, H + pad), (0, 0, 0))
        canvas.paste(panel, (0, 0))
        d = ImageDraw.Draw(canvas)
        d.text((6, H + 6), stats_text, fill=(255, 255, 255), font=WaveformScope._font(), stroke_width=1, stroke_fill=(0, 0, 0))
        return np.array(canvas, dtype=np.uint8)

    @staticmethod
    def compose_parade(wfr: np.ndarray, wfg: np.ndarray, wfb: np.ndarray,
                       r_stats: tuple[float, float, float, float, float],
                       g_stats: tuple[float, float, float, float, float],
                       b_stats: tuple[float, float, float, float, float],
                       gap: int = 8, pad: int = 72, left_pad: int = 56) -> np.ndarray:
        h, w = wfr.shape
        pr = (np.stack([wfr, np.zeros_like(wfr), np.zeros_like(wfr)], -1) * 255.0 + 0.5).astype(np.uint8)
        pg = (np.stack([np.zeros_like(wfg), wfg, np.zeros_like(wfg)], -1) * 255.0 + 0.5).astype(np.uint8)
        pb = (np.stack([np.zeros_like(wfb), np.zeros_like(wfb), wfb], -1) * 255.0 + 0.5).astype(np.uint8)
        col_r = Image.fromarray(pr)
        col_g = Image.fromarray(pg)
        col_b = Image.fromarray(pb)
        total_w = left_pad + w * 3 + gap * 2
        panel = Image.new("RGB", (total_w, h), (0, 0, 0))
        panel.paste(col_r, (left_pad + 0 * (w + gap), 0))
        panel.paste(col_g, (left_pad + 1 * (w + gap), 0))
        panel.paste(col_b, (left_pad + 2 * (w + gap), 0))
        draw = ImageDraw.Draw(panel)
        font = WaveformScope._font()
        for ire in (0, 12.5, 25, 37.5, 50, 62.5, 75, 87.5, 100):
            y = int(round(h - 1 - (ire / 100.0) * (h - 1)))
            draw.line([(left_pad, y), (total_w - 1, y)], fill=(64, 64, 64), width=1)
            draw.text((6, max(0, y - 6)), f"{ire:g}", fill=(200, 200, 200), font=font, stroke_width=1, stroke_fill=(0, 0, 0))
        W, H = panel.size
        canvas = Image.new("RGB", (W, H + pad), (0, 0, 0))
        canvas.paste(panel, (0, 0))
        r_txt = f"R  min {r_stats[0]:.4f}  max {r_stats[1]:.4f}  mean {r_stats[2]:.4f}  std {r_stats[3]:.4f}  median {r_stats[4]:.4f}"
        g_txt = f"G  min {g_stats[0]:.4f}  max {g_stats[1]:.4f}  mean {g_stats[2]:.4f}  std {g_stats[3]:.4f}  median {g_stats[4]:.4f}"
        b_txt = f"B  min {b_stats[0]:.4f}  max {b_stats[1]:.4f}  mean {b_stats[2]:.4f}  std {b_stats[3]:.4f}  median {b_stats[4]:.4f}"
        d = ImageDraw.Draw(canvas)
        big_font = WaveformScope._big_font()
        # Estimate line height for spacing
        try:
            bbox = big_font.getbbox("Ag")
            line_h = (bbox[3] - bbox[1]) + 4
        except Exception:
            line_h = 18
        y0 = H + 6
        # X positions aligned under each channel
        x_r = left_pad + 0 * (w + gap) + 6
        x_g = left_pad + 1 * (w + gap) + 6
        x_b = left_pad + 2 * (w + gap) + 6
        d.text((x_r, y0), r_txt, fill=(255, 64, 64), font=big_font, stroke_width=1, stroke_fill=(0, 0, 0))
        d.text((x_g, y0), g_txt, fill=(64, 255, 64), font=big_font, stroke_width=1, stroke_fill=(0, 0, 0))
        d.text((x_b, y0), b_txt, fill=(64, 128, 255), font=big_font, stroke_width=1, stroke_fill=(0, 0, 0))
        return np.array(canvas, dtype=np.uint8)
        
#  Load LUT

def get_lut_choice_list() -> list[str]:
    cubes = LUTLoader.discover_cube_files()
    names = ["Custom"]
    names += [name for name, _ in LUTLoader.BUILTIN_PRESETS]
    names += [f"LUT: {p.name}" for p in cubes]
    return names

class WASLoadLUT:
    _last_sig = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "look": (get_lut_choice_list(),),
                "builtin_size": ("INT", {"default": 33, "min": 17, "max": 65, "step": 2}),
                "custom_ev": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.01}),
                "custom_contrast": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "custom_saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "custom_vibrance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "custom_gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01}),
                "custom_temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "custom_tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "custom_red_balance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "custom_green_balance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "custom_blue_balance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LUT",)
    RETURN_NAMES = ("lut",)

    FUNCTION = "run"
    CATEGORY = "WAS/Color/LUT"

    def run(self, look, builtin_size, custom_ev, custom_contrast, custom_saturation,
            custom_vibrance, custom_gamma, custom_temperature, custom_tint,
            custom_red_balance, custom_green_balance, custom_blue_balance):
        if look == "Custom":
            grid = torch.linspace(0, 1, steps=builtin_size)
            rr, gg, bb = torch.meshgrid(grid, grid, grid, indexing="ij")
            cube = torch.stack([rr, gg, bb], dim=-1).unsqueeze(0).to(torch.float32)
            x = cube
            x = WASLUT.apply_exposure(x, custom_ev)
            x = WASLUT.apply_contrast(x, custom_contrast)
            x = WASLUT.apply_saturation(x, custom_saturation)
            x = WASLUT.apply_vibrance(x, custom_vibrance)
            x = WASLUT.apply_white_balance(x, custom_temperature, custom_tint)
            x = WASLUT.apply_color_balance(x, custom_red_balance, custom_green_balance, custom_blue_balance)
            x = WASLUT.apply_gamma(x, custom_gamma)
            table = x.squeeze(0).clamp(0, 1).cpu().numpy().astype(np.float32)
            return (LUT("Custom", (0, 0, 0), (1, 1, 1), None, table),)

        if look.startswith("LUT: "):
            target = look[5:].strip()
            path = None
            for p in LUTLoader.discover_cube_files():
                if p.name == target:
                    path = p
                    break
            if path is None:
                raise ValueError("LUT not found")
            return (LUTLoader.load_cube(path),)

        return (LUTLoader.synthesize_builtin_lut(look, builtin_size),)

# LUT Blender

class LUTBlender:
    @staticmethod
    def blend_linear(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        return (a * (1.0 - t) + b * t).astype(np.float32)

    @staticmethod
    def blend_multiply(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        mix = a * b
        return (a * (1.0 - t) + mix * t).astype(np.float32)

    @staticmethod
    def blend_screen(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        mix = 1.0 - (1.0 - a) * (1.0 - b)
        return (a * (1.0 - t) + mix * t).astype(np.float32)

    @staticmethod
    def blend_overlay(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        mid = np.where(a <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
        return (a * (1.0 - t) + mid * t).astype(np.float32)

    @staticmethod
    def blend_cosine(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        tt = (1.0 - np.cos(np.pi * float(t))) * 0.5
        return (a * (1.0 - tt) + b * tt).astype(np.float32)

    @staticmethod
    def blend_smoothstep(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        tt = float(t)
        tt = tt * tt * (3.0 - 2.0 * tt)
        return (a * (1.0 - tt) + b * tt).astype(np.float32)

    @staticmethod
    def blend_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """
        Spherical linear interpolation per color vector. Treat each RGB as a vector in R^3.
        We preserve approximate magnitude by lerping magnitudes and slerping directions.
        Fallback to linear when angle is tiny or vectors are near zero.
        """
        eps = 1e-8
        ta = a.astype(np.float32)
        tb = b.astype(np.float32)

        na = np.linalg.norm(ta, axis=-1, keepdims=True)
        nb = np.linalg.norm(tb, axis=-1, keepdims=True)
        ua = ta / np.clip(na, eps, None)
        ub = tb / np.clip(nb, eps, None)

        dot = np.clip(np.sum(ua * ub, axis=-1, keepdims=True), -1.0, 1.0)
        omega = np.arccos(dot)
        sin_omega = np.sin(omega)
        tt = float(t)
        mask_small = (sin_omega < 1e-4).astype(np.float32)

        coeff_a = np.where(mask_small == 1.0, 1.0 - tt, np.sin((1.0 - tt) * omega) / np.clip(sin_omega, eps, None))
        coeff_b = np.where(mask_small == 1.0, tt, np.sin(tt * omega) / np.clip(sin_omega, eps, None))
        u = coeff_a * ua + coeff_b * ub

        nu = np.linalg.norm(u, axis=-1, keepdims=True)
        u = u / np.clip(nu, eps, None)

        mag = (1.0 - tt) * na + tt * nb
        out = u * mag
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        s = np.where(maxc > 0, (maxc - minc) / np.clip(maxc, 1e-8, None), 0.0)
        rc = (maxc - r) / np.clip(maxc - minc, 1e-8, None)
        gc = (maxc - g) / np.clip(maxc - minc, 1e-8, None)
        bc = (maxc - b) / np.clip(maxc - minc, 1e-8, None)
        h = np.zeros_like(maxc, dtype=np.float32)
        h = np.where((maxc == r) & (maxc != minc), (bc - gc) / 6.0, h)
        h = np.where((maxc == g) & (maxc != minc), (2.0 + rc - bc) / 6.0, h)
        h = np.where((maxc == b) & (maxc != minc), (4.0 + gc - rc) / 6.0, h)
        h = (h % 1.0).astype(np.float32)
        return h, s.astype(np.float32), v.astype(np.float32)

    @staticmethod
    def _hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
        h6 = (h % 1.0) * 6.0
        i = np.floor(h6).astype(np.int32)
        f = h6 - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i_mod = i % 6
        rgb = np.zeros((*h.shape, 3), dtype=np.float32)

        mask = (i_mod == 0)
        rgb[..., 0] = np.where(mask, v, rgb[..., 0])
        rgb[..., 1] = np.where(mask, t, rgb[..., 1])
        rgb[..., 2] = np.where(mask, p, rgb[..., 2])

        mask = (i_mod == 1)
        rgb[..., 0] = np.where(mask, q, rgb[..., 0])
        rgb[..., 1] = np.where(mask, v, rgb[..., 1])
        rgb[..., 2] = np.where(mask, p, rgb[..., 2])

        mask = (i_mod == 2)
        rgb[..., 0] = np.where(mask, p, rgb[..., 0])
        rgb[..., 1] = np.where(mask, v, rgb[..., 1])
        rgb[..., 2] = np.where(mask, t, rgb[..., 2])

        mask = (i_mod == 3)
        rgb[..., 0] = np.where(mask, p, rgb[..., 0])
        rgb[..., 1] = np.where(mask, q, rgb[..., 1])
        rgb[..., 2] = np.where(mask, v, rgb[..., 2])

        mask = (i_mod == 4)
        rgb[..., 0] = np.where(mask, t, rgb[..., 0])
        rgb[..., 1] = np.where(mask, p, rgb[..., 1])
        rgb[..., 2] = np.where(mask, v, rgb[..., 2])

        mask = (i_mod == 5)
        rgb[..., 0] = np.where(mask, v, rgb[..., 0])
        rgb[..., 1] = np.where(mask, p, rgb[..., 1])
        rgb[..., 2] = np.where(mask, q, rgb[..., 2])

        return rgb.astype(np.float32)

    @staticmethod
    def blend_hsv(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        ha, sa, va = LUTBlender._rgb_to_hsv(a)
        hb, sb, vb = LUTBlender._rgb_to_hsv(b)
        tt = float(t)
        dh = ((hb - ha + 0.5) % 1.0) - 0.5
        h = (ha + tt * dh) % 1.0
        s = sa * (1.0 - tt) + sb * tt
        v = va * (1.0 - tt) + vb * tt
        out = LUTBlender._hsv_to_rgb(h, s, v)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)

    @staticmethod
    def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32)
        xc = np.clip(x, 0.0, None)
        out = np.where(xc <= 0.0031308, xc * 12.92, 1.055 * (xc ** (1/2.4)) - 0.055)
        return out.astype(np.float32)

    @staticmethod
    def _rgb_linear_to_xyz(rgb: np.ndarray) -> np.ndarray:
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=np.float32)
        xyz = np.tensordot(rgb, M.T, axes=1).astype(np.float32)
        return np.nan_to_num(xyz, nan=0.0, posinf=1e6, neginf=-1e6)

    @staticmethod
    def _xyz_to_rgb_linear(xyz: np.ndarray) -> np.ndarray:
        M = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ], dtype=np.float32)
        rgb = np.tensordot(xyz, M.T, axes=1).astype(np.float32)
        return np.nan_to_num(rgb, nan=0.0, posinf=1e6, neginf=-1e6)

    @staticmethod
    def _xyz_d65_to_d50(xyz: np.ndarray) -> np.ndarray:
        M = np.array([
            [ 0.9555766, -0.0230393,  0.0631636],
            [-0.0282895,  1.0099416,  0.0210077],
            [ 0.0122982, -0.0204830,  1.3299098],
        ], dtype=np.float32)
        return np.tensordot(xyz, M.T, axes=1).astype(np.float32)

    @staticmethod
    def _xyz_d50_to_d65(xyz: np.ndarray) -> np.ndarray:
        M = np.array([
            [ 1.0478112,  0.0228866, -0.0501270],
            [ 0.0295424,  0.9904844, -0.0170491],
            [-0.0092345,  0.0150436,  0.7521316],
        ], dtype=np.float32)
        return np.tensordot(xyz, M.T, axes=1).astype(np.float32)

    @staticmethod
    def _rgb_to_lab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lin = LUTBlender._srgb_to_linear(rgb)
        xyz_d65 = LUTBlender._rgb_linear_to_xyz(lin)
        xyz = LUTBlender._xyz_d65_to_d50(xyz_d65)
        # Avoid tiny negative values from numeric error before cbrt
        xyz = np.clip(xyz, 0.0, None)
        Xn, Yn, Zn = 0.96422, 1.0, 0.82521
        x = xyz[..., 0] / np.clip(Xn, 1e-8, None)
        y = xyz[..., 1] / np.clip(Yn, 1e-8, None)
        z = xyz[..., 2] / np.clip(Zn, 1e-8, None)
        e = (6/29) ** 3
        k = (29/6) ** 2 / 3
        f = lambda t: np.where(t > e, np.cbrt(t), k * t + 4/29)
        fx, fy, fz = f(x), f(y), f(z)
        L = 116 * fy - 16
        a = 500 * (fx - fy)
        b = 200 * (fy - fz)
        return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)

    @staticmethod
    def _lab_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        fy = (L + 16.0) / 116.0
        fx = fy + (a / 500.0)
        fz = fy - (b / 200.0)
        e = (6/29)
        e3 = e ** 3
        k = 3 * (e ** 2)
        invf = lambda t: np.where(t > e, t ** 3, (t - 4/29) / k)

        Xn, Yn, Zn = 0.96422, 1.0, 0.82521
        x = invf(fx) * Xn
        y = invf(fy) * Yn
        z = invf(fz) * Zn
        xyz_d50 = np.stack([x, y, z], axis=-1).astype(np.float32)

        xyz_d65 = LUTBlender._xyz_d50_to_d65(xyz_d50)
        lin = LUTBlender._xyz_to_rgb_linear(xyz_d65)
        rgb = LUTBlender._linear_to_srgb(lin)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _rgb_to_oklab(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lin = LUTBlender._srgb_to_linear(rgb)
        M1 = np.array([
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ], dtype=np.float32)
        lms = np.tensordot(lin, M1.T, axes=1).astype(np.float32)
        l_, m_, s_ = np.cbrt(lms[..., 0]), np.cbrt(lms[..., 1]), np.cbrt(lms[..., 2])
        L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
        a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
        b = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
        return L.astype(np.float32), a.astype(np.float32), b.astype(np.float32)

    @staticmethod
    def _oklab_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        l_ = L + 0.3963377774 * a + 0.2158037573 * b
        m_ = L - 0.1055613458 * a - 0.0638541728 * b
        s_ = L - 0.0894841775 * a - 1.2914855480 * b
        l = l_ ** 3
        m = m_ ** 3
        s = s_ ** 3
        M2 = np.array([
            [ 4.0767416621, -3.3077115913,  0.2309699292],
            [-1.2684380046,  2.6097574011, -0.3413193965],
            [-0.0041960863, -0.7034186147,  1.7076147010],
        ], dtype=np.float32)
        lin = np.tensordot(np.stack([l, m, s], axis=-1), M2.T, axes=1).astype(np.float32)
        rgb = LUTBlender._linear_to_srgb(lin)
        return np.clip(rgb, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def blend_lab(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        La, aa, ba = LUTBlender._rgb_to_lab(a)
        Lb, ab, bb = LUTBlender._rgb_to_lab(b)
        tt = float(t)
        L = La * (1.0 - tt) + Lb * tt
        A = aa * (1.0 - tt) + ab * tt
        B = ba * (1.0 - tt) + bb * tt
        # Clamp to valid Lab ranges to reduce out-of-gamut artifacts
        L = np.clip(L, 0.0, 100.0)
        A = np.clip(A, -128.0, 128.0)
        B = np.clip(B, -128.0, 128.0)
        out = LUTBlender._lab_to_rgb(L, A, B)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def blend_oklab(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        La, aa, ba = LUTBlender._rgb_to_oklab(a)
        Lb, ab, bb = LUTBlender._rgb_to_oklab(b)
        tt = float(t)
        L = La * (1.0 - tt) + Lb * tt
        A = aa * (1.0 - tt) + ab * tt
        B = ba * (1.0 - tt) + bb * tt
        out = LUTBlender._oklab_to_rgb(L, A, B)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def blend_auto(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        """
        Heuristic: use slerp for voxels where the RGB direction differs a lot (angle>~15deg),
        otherwise linear. This often gives a pleasing middle ground.
        """
        eps = 1e-8
        ta = a.astype(np.float32)
        tb = b.astype(np.float32)
        na = np.linalg.norm(ta, axis=-1, keepdims=True)
        nb = np.linalg.norm(tb, axis=-1, keepdims=True)
        ua = ta / np.clip(na, eps, None)
        ub = tb / np.clip(nb, eps, None)
        dot = np.clip(np.sum(ua * ub, axis=-1, keepdims=True), -1.0, 1.0)
        angle = np.arccos(dot)
        use_slerp = (angle > (15.0 * np.pi / 180.0)).astype(np.float32)
        lin = LUTBlender.blend_linear(a, b, t)
        slerp = LUTBlender.blend_slerp(a, b, t)
        return (lin * (1.0 - use_slerp) + slerp * use_slerp).astype(np.float32)

    @staticmethod
    def get_modes() -> list[str]:
        return [
            "linear",
            "cosine",
            "smoothstep",
            "slerp",
            "hsv",
            "lab",
            "oklab",
            "auto",
            "multiply",
            "screen",
            "overlay",
        ]


class WASCombineLUT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lut_a": ("LUT",),
                "lut_b": ("LUT",),
                "mode": (LUTBlender.get_modes(),),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "output_size": ("INT", {"default": 33, "min": 17, "max": 65, "step": 2}),
            }
        }

    RETURN_TYPES = ("LUT",)
    RETURN_NAMES = ("lut",)

    FUNCTION = "run"
    CATEGORY = "WAS/Color/LUT"

    def run(self, lut_a, lut_b, mode, strength, output_size):
        A = WASLUT.convert_to_3d(lut_a, output_size).table_3d
        B = WASLUT.convert_to_3d(lut_b, output_size).table_3d
        if mode == "linear":
            C = LUTBlender.blend_linear(A, B, strength)
        elif mode == "cosine":
            C = LUTBlender.blend_cosine(A, B, strength)
        elif mode == "smoothstep":
            C = LUTBlender.blend_smoothstep(A, B, strength)
        elif mode == "slerp":
            C = LUTBlender.blend_slerp(A, B, strength)
        elif mode == "hsv":
            C = LUTBlender.blend_hsv(A, B, strength)
        elif mode == "lab":
            C = LUTBlender.blend_lab(A, B, strength)
        elif mode == "oklab":
            C = LUTBlender.blend_oklab(A, B, strength)
        elif mode == "auto":
            C = LUTBlender.blend_auto(A, B, strength)
        elif mode == "multiply":
            C = LUTBlender.blend_multiply(A, B, strength)
        elif mode == "screen":
            C = LUTBlender.blend_screen(A, B, strength)
        else:
            C = LUTBlender.blend_overlay(A, B, strength)
        C = np.clip(C, 0.0, 1.0).astype(np.float32)
        return (LUT(f"{lut_a.title}+{lut_b.title}", (0, 0, 0), (1, 1, 1), None, C),)

# Apply LUT

class WASApplyLUT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut": ("LUT",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES =  ("image",)

    FUNCTION = "run"
    CATEGORY = "WAS/Color/LUT"

    def run(self, image, lut, strength):
        size = lut.size() if lut.size() > 1 else 33
        lut3 = WASLUT.convert_to_3d(lut, size)
        y = WASLUT.apply_lut_3d(image, lut3.table_3d, lut3.domain_min, lut3.domain_max).clamp(0, 1)
        if strength < 1.0:
            y = image * (1.0 - strength) + y * strength
        return (y.clamp(0, 1),)

# Save LUT

class WASSaveLUT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lut": ("LUT",),
                "filename": ("STRING", {"default": "CustomLUT"}),
                "output_size": ("INT", {"default": 33, "min": 17, "max": 65, "step": 2}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LUT",)
    RETURN_NAMES = ("lut",)
    FUNCTION = "run"
    CATEGORY = "WAS/Color/LUT"

    def run(self, lut, filename, output_size, overwrite):
        lut_dirs = LUTLoader.get_lut_dirs()
        if not lut_dirs:
            raise RuntimeError("No models/LUT directory found. Please create one under your ComfyUI models folder.")
        dst_dir = lut_dirs[0]
        dst_dir.mkdir(parents=True, exist_ok=True)
        name = filename.strip()
        if not name.lower().endswith(".cube"):
            name += ".cube"
        path = dst_dir / name
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists. Enable overwrite to replace.")

        lut3 = WASLUT.convert_to_3d(lut, output_size)
        LUTLoader.save_cube(path, lut3)

        return (lut3,)

# RGB PARADE

def get_temp_dir() -> str:
    try:
        return folder_paths.get_temp_directory()
    except Exception:
        root = os.path.join(tempfile.gettempdir(), "comfyui_temp")
        os.makedirs(root, exist_ok=True)
        return root

def save_waveform(img: Image.Image, prefix: str = "rgb_parade") -> dict:
    temp_dir = get_temp_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{stamp}_{int(time.time() * 1000) % 100000}.png"
    path = os.path.join(temp_dir, fname)
    img.save(path, compress_level=4)
    return {"filename": fname, "subfolder": "", "type": "temp"}


def np_rgb_to_image_tensor(np_img: np.ndarray) -> torch.Tensor:
    if np_img.ndim != 3 or np_img.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB, got {np_img.shape}")
    if np_img.dtype != np.float32:
        np_img = np_img.astype(np.float32) / 255.0
    np_img = np.ascontiguousarray(np_img)
    return torch.from_numpy(np_img).to(torch.float32).unsqueeze(0)


class WASChannelWaveform:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "waveform_height": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("red_waveform", "green_waveform", "blue_waveform", "rgb_parade")
    OUTPUT_NODE = True

    FUNCTION = "run"
    CATEGORY = "WAS/Image/Scopes"

    def run(self, image, waveform_height):
        b, h, w, _ = image.shape
        red_list, green_list, blue_list, parade_list = [], [], [], []
        ui_entries = []

        for i in range(b):
            frame = image[i]
            r = frame[..., 0].detach().cpu().numpy().astype(np.float32)
            g = frame[..., 1].detach().cpu().numpy().astype(np.float32)
            bl = frame[..., 2].detach().cpu().numpy().astype(np.float32)

            r_stats = WaveformScope.stats_tensor(frame[..., 0])
            g_stats = WaveformScope.stats_tensor(frame[..., 1])
            b_stats = WaveformScope.stats_tensor(frame[..., 2])

            wr = WaveformScope.make_waveform_gray(r, waveform_height)
            wg = WaveformScope.make_waveform_gray(g, waveform_height)
            wb = WaveformScope.make_waveform_gray(bl, waveform_height)

            r_txt = f"min {r_stats[0]:.4f}  max {r_stats[1]:.4f}  mean {r_stats[2]:.4f}  std {r_stats[3]:.4f}  median {r_stats[4]:.4f}"
            g_txt = f"min {g_stats[0]:.4f}  max {g_stats[1]:.4f}  mean {g_stats[2]:.4f}  std {g_stats[3]:.4f}  median {g_stats[4]:.4f}"
            b_txt = f"min {b_stats[0]:.4f}  max {b_stats[1]:.4f}  mean {b_stats[2]:.4f}  std {b_stats[3]:.4f}  median {b_stats[4]:.4f}"

            r_img = WaveformScope.compose_waveform_panel(wr, "red", r_txt)
            g_img = WaveformScope.compose_waveform_panel(wg, "green", g_txt)
            b_img = WaveformScope.compose_waveform_panel(wb, "blue", b_txt)
            parade_np = WaveformScope.compose_parade(wr, wg, wb, r_stats, g_stats, b_stats)

            red_list.append(np_rgb_to_image_tensor(r_img))
            green_list.append(np_rgb_to_image_tensor(g_img))
            blue_list.append(np_rgb_to_image_tensor(b_img))
            parade_list.append(np_rgb_to_image_tensor(parade_np))

            parade_pil = Image.fromarray(parade_np).convert("RGB")
            ui_entries.append(save_waveform(parade_pil, prefix="RGB_Parade"))

        red_batch = torch.cat(red_list, dim=0)
        green_batch = torch.cat(green_list, dim=0)
        blue_batch = torch.cat(blue_list, dim=0)
        parade_batch = torch.cat(parade_list, dim=0)

        return {"ui": {"images": ui_entries}, "result": (red_batch, green_batch, blue_batch, parade_batch)}


NODE_CLASS_MAPPINGS = {
    "WASLoadLUT": WASLoadLUT,
    "WASCombineLUT": WASCombineLUT,
    "WASApplyLUT": WASApplyLUT,
    "WASSaveLUT": WASSaveLUT,
    "WASChannelWaveform": WASChannelWaveform,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WASLoadLUT": "WAS Load LUT",
    "WASCombineLUT": "WAS LUT Blender",
    "WASApplyLUT": "WAS Apply LUT",
    "WASSaveLUT": "WAS Save LUT (.cube)",
    "WASChannelWaveform": "WAS Channel Waveform (Parade)",
}
