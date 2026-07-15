"""IMX662 Bayer RAW I/O and physically-plausible sensor simulation.

The stack works on demosaiced linear RGB in [0, 1]. This module either:
  * loads a real Bayer RAW frame supplied by the user, or
  * synthesises a plausible IMX662 capture at an extreme analog gain so the
    demo always has a genuinely noisy frame to denoise.

The noise model is a standard photon-transfer model: Poisson shot noise on the
collected electrons plus Gaussian read noise, both scaled by the analog gain.
A clean reference ("ground truth") is produced by temporally averaging many
simulated frames - exactly how a long-exposure / multi-frame reference is built.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .sensors import SensorProfile, get_sensor

BLACK_LEVEL = 0.015  # normalised pedestal


def _synthetic_scene(h: int, w: int, seed: int) -> np.ndarray:
    """A structured low-light test scene with edges, gradients and fine detail.

    Returns clean linear RGB in [0, 1]. Structure (edges, text-like bars, a
    colour chart) makes denoising quality visually obvious in the demo.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yn, xn = yy / h, xx / w

    # Soft vignetted background gradient.
    img = 0.18 + 0.22 * (1.0 - ((xn - 0.5) ** 2 + (yn - 0.5) ** 2))
    img = np.clip(img, 0, 1)
    img = np.stack([img * 0.9, img, img * 1.05], axis=-1)

    # A colour-checker style patch grid (tests chroma fidelity).
    chart = np.array(
        [
            [0.45, 0.32, 0.27], [0.76, 0.58, 0.50], [0.36, 0.47, 0.61],
            [0.34, 0.42, 0.26], [0.51, 0.50, 0.66], [0.40, 0.74, 0.67],
        ],
        dtype=np.float32,
    )
    ph, pw = h // 6, w // 9
    for i, colour in enumerate(chart):
        r0 = h // 12 + (i // 3) * ph
        c0 = w // 12 + (i % 3) * pw
        img[r0 : r0 + ph - 4, c0 : c0 + pw - 4] = colour

    # High-frequency resolution bars (tests detail preservation).
    bar_region = img[int(h * 0.62) : int(h * 0.92), int(w * 0.55) : int(w * 0.92)]
    bh, bw, _ = bar_region.shape
    freq = np.linspace(2, 30, bw)
    bars = 0.25 + 0.45 * (0.5 + 0.5 * np.sin(np.cumsum(freq) * 0.30))
    bar_region[:] = bars[None, :, None]

    # A couple of bright point sources (tests highlight handling).
    for _ in range(6):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        cv2.circle(img, (int(cx), int(cy)), rng.integers(2, 5), (0.95, 0.95, 0.9), -1)

    # Do NOT Gaussian-blur the clean scene. A soft GT teaches every denoiser to
    # blur; real multi-frame averages of a static target keep full edge contrast.
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def _capture(clean: np.ndarray, gain: int, sensor: SensorProfile,
             rng: np.random.Generator, prnu: np.ndarray) -> np.ndarray:
    """Simulate one noisy read of a clean linear-RGB scene for a given sensor.

    Photon-transfer model: shot noise on the collected electrons (scaled by
    quantum efficiency and full-well capacity) + read noise, with a fixed-pattern
    PRNU gain map and low-frequency chroma cross-talk. Lower-grade sensors (low
    QE / high read noise / high chroma) therefore produce visibly messier frames.
    """
    eff = sensor.full_well * sensor.qe / float(gain)
    photons = np.clip(clean, 0, 1) * eff * prnu
    electrons = rng.poisson(np.maximum(photons, 0.0)).astype(np.float32)
    electrons += rng.normal(0.0, sensor.read_noise, size=clean.shape).astype(np.float32)
    noisy = electrons / eff

    if sensor.chroma_noise > 0:
        cn = rng.normal(0.0, sensor.chroma_noise, size=clean.shape).astype(np.float32)
        cn = cv2.GaussianBlur(cn, (0, 0), 2.2)        # low-frequency splotches
        cn -= cn.mean(axis=2, keepdims=True)          # opponent (chroma-only) noise
        noisy += cn

    noisy += BLACK_LEVEL
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def _to_bayer(rgb: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Mosaic an RGB image down to a single-channel Bayer plane."""
    h, w, _ = rgb.shape
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    bayer = np.zeros((h, w), dtype=np.float32)
    bayer[0::2, 0::2] = r[0::2, 0::2]
    bayer[0::2, 1::2] = g[0::2, 1::2]
    bayer[1::2, 0::2] = g[1::2, 0::2]
    bayer[1::2, 1::2] = b[1::2, 1::2]
    return bayer


def _demosaic(bayer: np.ndarray) -> np.ndarray:
    """Demosaic a normalised RGGB Bayer plane back to linear RGB."""
    b16 = np.clip(bayer * 65535.0, 0, 65535).astype(np.uint16)
    rgb = cv2.cvtColor(b16, cv2.COLOR_BAYER_RG2RGB)
    return (rgb.astype(np.float32) / 65535.0)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
RAW_EXTS = {".npy", ".dng", ".raw"}
SUPPORTED_EXTS = IMAGE_EXTS | RAW_EXTS


def resolve_dataset(path: str | None, seed: int) -> Path | None:
    """Resolve a dataset path to a single frame.

    Accepts a direct file, or a directory (searched recursively) from which one
    supported frame is picked (seeded, so different seeds show different frames).
    Returns ``None`` if nothing usable is found.
    """
    if not path:
        return None
    p = Path(path).expanduser()
    if p.is_file():
        return p
    if p.is_dir():
        files = sorted(f for f in p.rglob("*")
                       if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS)
        if not files:
            return None
        idx = int(np.random.default_rng(seed).integers(0, len(files)))
        return files[idx]
    return None


def _load_dng_linear_rgb(path: Path) -> np.ndarray:
    """DNG → linear RGB [0,1] with the SAME pipeline used to build burst GT.

    Critical: ``rawpy.postprocess`` defaults apply a camera gamma (~sRGB) and a
    different demosaic than OpenCV's Bayer path. Training noisy(gamma-ISP) against
    gt(linear OpenCV demosaic of a Bayer mean) is a domain mismatch — the network
    learns a soft compromise and can never match the sharp linear GT.

    We black-subtract, white-normalise, and demosaic with ``COLOR_BAYER_RG2RGB``
    exactly like ``scratchpad/build_dng_pairs.demosaic_mean`` / packed-RAW GT.
    """
    import rawpy
    with rawpy.imread(str(path)) as raw:
        bayer = raw.raw_image_visible.astype(np.float32)
        black = float(np.mean(raw.black_level_per_channel))
        white = float(raw.white_level)
        # Prefer documented CFA; fall back to RGGB (IMX662).
        pattern = None
        try:
            pattern = raw.raw_pattern
        except Exception:
            pattern = None
    norm = np.clip((bayer - black) / max(white - black, 1.0), 0.0, 1.0)
    return _demosaic_bayer_norm(norm, pattern)


def _demosaic_bayer_norm(norm01: np.ndarray, pattern=None) -> np.ndarray:
    """Normalised Bayer [0,1] → linear RGB [0,1] via OpenCV."""
    b16 = (np.clip(norm01, 0, 1) * 65535.0 + 0.5).astype(np.uint16)
    # IMX662 / most Pi DNGs are RGGB. raw_pattern is 2x2 with color indices.
    code = cv2.COLOR_BAYER_RG2RGB
    if pattern is not None:
        try:
            p = np.asarray(pattern).reshape(2, 2)
            # rawpy: 0=R, 1=G, 2=B, 3=G
            key = tuple(int(x) for x in p.ravel())
            code = {
                (0, 1, 3, 2): cv2.COLOR_BAYER_RG2RGB,
                (1, 0, 2, 3): cv2.COLOR_BAYER_GR2RGB,
                (3, 2, 0, 1): cv2.COLOR_BAYER_BG2RGB,
                (2, 3, 1, 0): cv2.COLOR_BAYER_GB2RGB,
            }.get(key, cv2.COLOR_BAYER_RG2RGB)
        except Exception:
            pass
    rgb16 = cv2.cvtColor(b16, code)
    return rgb16.astype(np.float32) / 65535.0


def _load_any(path: Path) -> np.ndarray:
    """Decode any supported frame file to normalised linear RGB [0, 1]."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 2:
            arr = _demosaic(arr / max(float(arr.max()), 1e-6))
        return arr / max(float(arr.max()), 1e-6)
    if suffix == ".dng":
        try:
            return _load_dng_linear_rgb(path)
        except Exception as exc:
            raise RuntimeError(f"DNG support needs 'rawpy' (pip install rawpy): {exc}")
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read frame: {path}")
    if img.ndim == 2:
        return _demosaic(img.astype(np.float32) / max(float(img.max()), 1e-6))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # 16-bit TIF GTs from build_dng_pairs are already linear; 8-bit PNGs are
    # display-referred. Normalise by bit-depth, not per-image max (max-norm
    # desynchronises noisy/gt scales when highlight clipping differs).
    if img.dtype == np.float32 and img.max() <= 1.0:
        return np.clip(img, 0.0, 1.0)
    scale = 65535.0 if img.max() > 255.5 else 255.0
    return np.clip(img / scale, 0.0, 1.0)


def load_real_raw(path: str, patch: int) -> np.ndarray:
    """Load a frame file and fit it to the working patch (resize short side + crop)."""
    arr = _load_any(Path(path))
    return _fit_and_crop(arr, patch)


def _fit_and_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Scale so the short side == size, then centre-crop a square patch."""
    h, w = img.shape[:2]
    s = size / float(min(h, w))
    nh, nw = max(size, int(round(h * s))), max(size, int(round(w * s)))
    if (nh, nw) != (h, w):
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_LINEAR
        img = cv2.resize(img, (nw, nh), interpolation=interp)
    return _center_crop(img, size)


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    size = min(size, h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return img[y0 : y0 + size, x0 : x0 + size]


def _classical_reference(noisy: np.ndarray) -> np.ndarray:
    """Build a clean reference for a single real capture (no temporal GT exists).

    Uses non-local-means + edge-preserving smoothing so the network has a usable
    supervised target and PSNR remains meaningful for real frames.
    """
    img8 = (np.clip(noisy, 0, 1) * 255.0).astype(np.uint8)
    den = cv2.fastNlMeansDenoisingColored(img8, None, 9, 9, 7, 21)
    den = cv2.bilateralFilter(den, 5, 45, 45)
    return den.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Real-dataset ingestion  (logic adapted from davidplowman/denoise-hw)
#   * paired noisy/gt folders        (his folders.find_folders convention)
#   * detail-scored patch selection  (his dataset._patch_detail)
#   * keyword filtering              (his --filter tokens)
# ---------------------------------------------------------------------------

_LAP = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)


def _detail_score(gray: np.ndarray) -> float:
    """Laplacian-variance / mean^2 detail score (after denoise-hw)."""
    lap = cv2.filter2D(gray.astype(np.float32), -1, _LAP)
    mean_sq = float(gray.mean()) ** 2 + 1e-6
    return 20.0 * float(lap.var() / mean_sq)


def _detail_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Pick the most detailed square crop of `img` (sharp, interesting region)."""
    img = _fit_min_side(img, size)
    h, w = img.shape[:2]
    if h == size and w == size:
        return img
    gray = cv2.cvtColor((np.clip(img, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)
    step = max(1, size // 2)
    best, best_xy = -1.0, (0, 0)
    for y0 in range(0, h - size + 1, step):
        for x0 in range(0, w - size + 1, step):
            s = _detail_score(gray[y0:y0 + size, x0:x0 + size])
            if s > best:
                best, best_xy = s, (y0, x0)
    y0, x0 = best_xy
    return img[y0:y0 + size, x0:x0 + size]


def _fit_min_side(img: np.ndarray, size: int) -> np.ndarray:
    """Scale so the short side is >= size (never upscale beyond ~1.6x crop budget)."""
    h, w = img.shape[:2]
    s = size / float(min(h, w))
    if s < 1.0:
        interp = cv2.INTER_AREA
        img = cv2.resize(img, (max(size, int(round(w * s))),
                               max(size, int(round(h * s)))), interpolation=interp)
    elif min(h, w) < size:
        img = cv2.resize(img, (max(size, w), max(size, h)), interpolation=cv2.INTER_LINEAR)
    return img


# Preference order when a folder holds more than one file for the same role
# (e.g. a leftover noisy.png next to a real noisy.dng) — always prefer the raw
# capture, then the highest-precision derived format, over an 8-bit PNG.
_EXT_RANK = {".dng": 0, ".raw": 1, ".npy": 1, ".tif": 2, ".tiff": 2,
            ".png": 3, ".jpg": 4, ".jpeg": 4, ".bmp": 4, ".webp": 4}


def _pair_in_folder(folder: Path) -> tuple[Path, Path] | None:
    """Return (noisy, gt) paths if `folder` holds a paired capture, else None.

    Mirrors denoise-hw's noisy.dng / gt.dng convention but accepts any supported
    extension so it also works without rawpy (e.g. noisy.png + gt.png). When a
    folder has more than one candidate for a role (a stale noisy.png left next
    to a real noisy.dng), the raw/highest-precision file always wins.
    """
    noisy = gt = None
    noisy_rank = gt_rank = 99
    for f in folder.iterdir():
        if not f.is_file():
            continue
        stem, ext = f.stem.lower(), f.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        rank = _EXT_RANK.get(ext, 9)
        if stem == "noisy" and rank < noisy_rank:
            noisy, noisy_rank = f, rank
        elif stem in ("gt", "clean", "reference") and rank < gt_rank:
            gt, gt_rank = f, rank
    if noisy is not None and gt is not None:
        return noisy, gt
    return None


def find_paired_folders(root: str, filter_tokens: list[str] | None = None) -> list[Path]:
    """Recursively find folders containing a paired noisy/gt capture.

    Adapted from denoise-hw's ``folders.find_folders``: a folder qualifies when
    it holds both a noisy and a gt frame and its path contains every filter token.
    """
    root_p = Path(root).expanduser()
    if not root_p.exists():
        return []
    out: list[Path] = []
    for folder in [root_p, *(p for p in root_p.rglob("*") if p.is_dir())]:
        if _pair_in_folder(folder) is None:
            continue
        s = str(folder)
        if filter_tokens and not all(t in s for t in filter_tokens):
            continue
        out.append(folder)
    return sorted(set(out))


def list_frames(path: str | None, filter_tokens: list[str] | None = None,
                limit: int = 0) -> list[dict]:
    """Enumerate usable real frames under `path`.

    Returns a list of frame-source dicts:
      {"noisy": Path, "gt": Path|None, "name": str}
    Paired folders (noisy/gt) are preferred; otherwise every supported loose file
    is returned with no ground-truth pair. Keyword filtering follows denoise-hw.
    """
    if not path:
        return []
    try:
        from nsa.denoise_hw_data import normalize_dataset_root
        p = normalize_dataset_root(path)
    except Exception:
        p = Path(path).expanduser()
    frames: list[dict] = []

    if p.is_file():
        frames.append({"noisy": p, "gt": None, "name": p.name})
    elif p.is_dir():
        pairs = find_paired_folders(str(p), filter_tokens)
        if pairs:
            for folder in pairs:
                pr = _pair_in_folder(folder)
                if pr:
                    frames.append({"noisy": pr[0], "gt": pr[1], "name": folder.name})
        else:
            files = sorted(f for f in p.rglob("*")
                           if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
                           and f.stem.lower() not in ("gt", "clean", "reference"))
            for f in files:
                s = str(f)
                if filter_tokens and not all(t in s for t in filter_tokens):
                    continue
                frames.append({"noisy": f, "gt": None, "name": f.name})

    if limit and limit > 0:
        frames = frames[:limit]
    return frames


class Frame:
    """Bundle of the tensors the rest of the stack needs."""

    def __init__(self, noisy_rgb, clean_rgb, bayer, gain, source, sensor,
                 gt_kind="temporal"):
        self.noisy_rgb = noisy_rgb          # HxWx3 float32 [0,1]  (Panel A)
        self.clean_rgb = clean_rgb          # HxWx3 float32 [0,1]  (Panel B / GT)
        self.bayer = bayer                  # HxW   float32 [0,1]  (the RAW plane)
        self.gain = gain
        self.source = source                # "synthetic" | path | folder name
        self.sensor = sensor                # SensorProfile
        self.gt_kind = gt_kind              # temporal | paired | reference
        self.height, self.width = noisy_rgb.shape[:2]


def _synth_noisy_gt(clean, gain, sensor, temporal_frames, seed):
    """Inject sensor noise on a clean scene; build temporal-average ground truth."""
    rng = np.random.default_rng(seed)
    prnu = (1.0 + np.random.default_rng(seed + 777).normal(
        0.0, sensor.prnu, size=clean.shape)).astype(np.float32)
    noisy = _capture(clean, gain, sensor, rng, prnu)
    acc = np.zeros_like(clean)
    for _ in range(max(1, temporal_frames)):
        acc += _capture(clean, gain, sensor, rng, prnu)
    return noisy, acc / max(1, temporal_frames)


def build_burst(clean, gain, sensor, n: int, seed: int):
    """Generate an ordered burst of independent noisy reads of one clean scene.

    Used by the temporal video-denoise mode: the scene is fixed, only the
    sensor noise differs frame-to-frame (the low-motion video case).
    """
    if isinstance(sensor, str):
        sensor = get_sensor(sensor)
    rng = np.random.default_rng(seed)
    prnu = (1.0 + np.random.default_rng(seed + 777).normal(
        0.0, sensor.prnu, size=clean.shape)).astype(np.float32)
    return [_capture(clean, gain, sensor, rng, prnu) for _ in range(max(2, n))]


def _load_pair(noisy_path: Path, gt_path: Path, patch: int):
    """Load a paired noisy/gt capture and crop both at the same detailed window."""
    noisy = _fit_min_side(_load_any(noisy_path), patch)
    gt = _fit_min_side(_load_any(gt_path), patch)
    h = min(noisy.shape[0], gt.shape[0])
    w = min(noisy.shape[1], gt.shape[1])
    noisy, gt = noisy[:h, :w], gt[:h, :w]
    if (h, w) == (patch, patch):
        return noisy, gt
    gray = cv2.cvtColor((np.clip(noisy, 0, 1) * 255).astype(np.uint8),
                        cv2.COLOR_RGB2GRAY).astype(np.float32)
    step = max(1, patch // 2)
    best, bxy = -1.0, (0, 0)
    for y0 in range(0, h - patch + 1, step):
        for x0 in range(0, w - patch + 1, step):
            s = _detail_score(gray[y0:y0 + patch, x0:x0 + patch])
            if s > best:
                best, bxy = s, (y0, x0)
    y0, x0 = bxy
    return noisy[y0:y0 + patch, x0:x0 + patch], gt[y0:y0 + patch, x0:x0 + patch]


def build_frame_from_source(
    source: dict,
    gain: int,
    temporal_frames: int,
    patch: int,
    sensor: SensorProfile | str,
    seed: int,
    simulate_noise: bool = False,
) -> Frame:
    """Build a Frame from a real frame-source dict (see ``list_frames``).

    * paired noisy/gt           -> real ground truth (denoise-hw convention)
    * paired + simulate_noise   -> use gt as clean, inject sensor noise on top
    * loose file                -> frame is the noisy input; NL-means reference
    * loose file + simulate_noise-> treat file as clean, inject sensor noise
    """
    if isinstance(sensor, str):
        sensor = get_sensor(sensor)
    name = source.get("name", str(source["noisy"]))
    gt_path = source.get("gt")

    if gt_path is not None:
        noisy, gt = _load_pair(Path(source["noisy"]), Path(gt_path), patch)
        if simulate_noise:                       # clean gt -> simulate target sensor
            noisy, gt = _synth_noisy_gt(gt, gain, sensor, temporal_frames, seed)
            kind = "paired+sim"
        else:
            kind = "paired"
    else:
        img = _detail_crop(_load_any(Path(source["noisy"])), patch)
        if simulate_noise:                       # treat file as clean source
            noisy, gt = _synth_noisy_gt(img, gain, sensor, temporal_frames, seed)
            kind = "clean+sim"
        else:                                    # file IS the noisy capture
            noisy, gt = img, _classical_reference(img)
            kind = "reference"

    bayer = _to_bayer(noisy, sensor.bayer)
    return Frame(noisy, gt, bayer, gain, name, sensor, gt_kind=kind)


def _cap_long_side(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the longest side is <= max_side (never upscale)."""
    if max_side <= 0:
        return img
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    s = max_side / float(longest)
    return cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                      interpolation=cv2.INTER_AREA)


def _find_burst_dir(dataset_root: Path, folder: Path) -> Path | None:
    """Map a PI_RAW test folder (``<scene>/imx662[h]_ag<N>_test``) to its raw
    burst directory (``imx662_project/bursts/<scene>/ag<N>``), if one exists.

    The single ``noisy.dng`` copied into a PI_RAW folder is just one frame from
    a much larger burst (up to 512 raw frames) captured at that scene/gain.
    Finding the sibling burst lets training draw a different real frame per
    crop instead of reusing the same one repeatedly.
    """
    import re
    m = re.match(r"imx662h?_ag(\d+)_test$", folder.name)
    if not m:
        return None
    gain_tag = f"ag{m.group(1)}"
    scene = folder.parent.name
    for base in (dataset_root.parent / "imx662_project" / "bursts",
                 dataset_root / "imx662_project" / "bursts"):
        d = base / scene / gain_tag
        if d.is_dir() and len(list(d.glob("*.dng"))) > 1:
            return d
    return None


def load_training_pairs(
    path: str | None,
    filter_tokens: list[str] | None = None,
    sensor: SensorProfile | str = "imx662",
    gain: int = 256,
    simulate_noise: bool = False,
    seed: int = 0,
    temporal_frames: int = 8,
    max_side: int = 1024,
    min_patch: int = 64,
    with_names: bool = False,
    tile: int = 0,
    tiles_per_image: int = 4,
    use_burst_frames: bool = True,
) -> list:
    """Load EVERY paired capture under ``path`` as full-image (noisy, clean) pairs.

    Unlike :func:`build_frame_from_source` (which returns a single detail crop per
    folder), this keeps whole images so an extended-training pass can draw many
    diverse random crops from the entire PI_RAW dataset — the "patches across all
    images" strategy that trains a much stronger denoiser than a single frame.

    * paired noisy/gt          -> real ground truth (denoise-hw convention)
    * paired + simulate_noise  -> use gt as clean, inject sensor noise on top

    ``tile`` > 0 (recommended): cut ``tiles_per_image`` random NATIVE-resolution
    ``tile``×``tile`` squares from each capture instead of resizing. Downscaling
    a noisy frame averages the grain away, so a model trained on resized images
    systematically under-estimates real sensor noise; native tiles preserve the
    true noise statistics while bounding memory. ``tile`` = 0 restores the
    legacy resize-to-``max_side`` behaviour.

    Returns ``(noisy_rgb, clean_rgb)`` float32 [0,1] pairs — or, with
    ``with_names=True``, ``(folder_name, noisy_rgb, clean_rgb)`` triples so the
    caller can weight sampling by each capture's analogue-gain tag.
    """
    if not path:
        return []
    if isinstance(sensor, str):
        sensor = get_sensor(sensor)
    try:
        from nsa.denoise_hw_data import normalize_dataset_root
        root = str(normalize_dataset_root(path))
    except Exception:
        root = str(Path(path).expanduser())

    pairs: list = []
    for i, folder in enumerate(find_paired_folders(root, filter_tokens)):
        pr = _pair_in_folder(folder)
        if pr is None:
            continue
        try:
            noisy = _load_any(pr[0])
            gt = _load_any(pr[1])
            if not tile:                         # legacy: bound memory by resizing
                noisy = _cap_long_side(noisy, max_side)
                gt = _cap_long_side(gt, max_side)
        except Exception:
            continue
        h = min(noisy.shape[0], gt.shape[0])
        w = min(noisy.shape[1], gt.shape[1])
        if h < min_patch or w < min_patch:
            continue
        noisy, gt = noisy[:h, :w], gt[:h, :w]
        if simulate_noise:                       # gt is clean -> simulate sensor noise
            noisy, gt = _synth_noisy_gt(gt, gain, sensor, temporal_frames, seed + i)
        if tile and tile > 0:
            # Native-resolution random tiles: true noise statistics, bounded memory.
            t = min(int(tile), h, w)
            rng = np.random.default_rng(seed * 100003 + i)
            burst_dir = (_find_burst_dir(Path(root), folder)
                        if use_burst_frames and not simulate_noise else None)
            burst_files = sorted(burst_dir.glob("*.dng")) if burst_dir else None
            for _k in range(max(1, int(tiles_per_image))):
                src = noisy
                if burst_files:
                    # A different real captured frame per crop (up to 512 in
                    # the burst) instead of reusing the one noisy.dng copy —
                    # exposes the model to far more real noise realizations.
                    fpath = burst_files[int(rng.integers(0, len(burst_files)))]
                    try:
                        frame = _load_any(fpath)
                        fh, fw = min(frame.shape[0], h), min(frame.shape[1], w)
                        src = frame[:fh, :fw]
                    except Exception:
                        src = noisy
                iy = int(rng.integers(0, h - t + 1))
                ix = int(rng.integers(0, w - t + 1))
                item = (src[iy:iy + t, ix:ix + t].astype(np.float32),
                        gt[iy:iy + t, ix:ix + t].astype(np.float32))
                pairs.append((folder.name, *item) if with_names else item)
            continue
        item = (noisy.astype(np.float32), gt.astype(np.float32))
        pairs.append((folder.name, *item) if with_names else item)
    return pairs


def analog_gain_from_name(name) -> int | None:
    """Parse the analogue gain from a capture folder name (``…ag<N>…``).

    Dataset folders follow the denoise-hw convention ``imx662_ag<gain>_test``;
    returns the gain (e.g. 512) or ``None`` when the name carries no ag tag.
    """
    import re
    m = re.search(r"ag(\d+)", str(name or "").lower())
    return int(m.group(1)) if m else None


def training_sample_weights(names, cleans, *, gain_exp: float = 0.5,
                            dark_emphasis: float = 2.0) -> list[float]:
    """Per-capture sampling weights emphasising high-gain, low-intensity frames.

    High analogue gain means far heavier grain, and dark scenes are where the
    denoiser is actually needed — but uniform sampling gives an ag512 dark
    capture the same training attention as an ag1 bright one. Weight:

        w = gain^gain_exp · (1 + dark_emphasis · darkness)

    where ``darkness`` ramps 0→1 as the clean frame's mean intensity falls
    below 0.35. With the defaults (sqrt gain, dark ×3), an ag512 dark capture
    is sampled ~20-60× more often than an ag1 bright one while every folder
    keeps a non-zero share. Captures without an ag tag get the median gain of
    the tagged ones (neutral), so mixed datasets stay balanced.
    """
    gains = [analog_gain_from_name(n) for n in names]
    known = [g for g in gains if g]
    fill = float(np.median(known)) if known else 1.0
    weights = []
    for g_val, clean in zip(gains, cleans):
        gw = float(g_val if g_val else fill) ** max(0.0, float(gain_exp))
        luma = float(np.mean(clean))
        dark = min(max((0.35 - luma) / 0.35, 0.0), 1.0)
        weights.append(gw * (1.0 + max(0.0, float(dark_emphasis)) * dark))
    return weights


def build_frame(
    input_raw: str | None,
    gain: int,
    temporal_frames: int,
    patch: int,
    sensor: SensorProfile | str,
    seed: int,
    real_capture: bool = False,
    simulate_noise: bool = False,
    filter_tokens: list[str] | None = None,
) -> Frame:
    """Produce the (noisy, reference, bayer) frame bundle for the chosen sensor.

    Modes:
      * real_capture=True  -> load a real frame/dataset (paired noisy/gt preferred)
        and use it as the noisy input; optionally simulate sensor noise instead.
      * input_raw set      -> treat the loaded image as a clean source and inject
        the sensor's physical noise on top of it.
      * neither            -> synthesise a clean scene and inject sensor noise.
    """
    if isinstance(sensor, str):
        sensor = get_sensor(sensor)

    if real_capture:
        frames = list_frames(input_raw, filter_tokens)
        if frames:
            idx = int(np.random.default_rng(seed).integers(0, len(frames)))
            try:
                return build_frame_from_source(
                    frames[idx], gain, temporal_frames, patch, sensor, seed,
                    simulate_noise=simulate_noise)
            except Exception:
                pass  # fall back to synthetic below

    if input_raw and not real_capture:
        clean = load_real_raw(input_raw, patch)
        source = input_raw
    else:
        clean = _center_crop(_synthetic_scene(patch, patch, seed), patch)
        source = "synthetic"

    noisy, gt = _synth_noisy_gt(clean, gain, sensor, temporal_frames, seed)
    bayer = _to_bayer(noisy, sensor.bayer)
    return Frame(noisy, gt, bayer, gain, source, sensor, gt_kind="temporal")
