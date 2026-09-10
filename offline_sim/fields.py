"""
示向度误差场（可插拔策略）。

题面对示向度误差**只给了三条硬约束**，没有给出概率分布：
  (i)   全局有界：误差 ∈ [-1°, +1°]；
  (ii)  地点决定：同一地点电磁环境固定 → 重复检测误差不变；
  (iii) 空间统计性：不同地点、不同电磁环境下误差才呈统计规律。

题面【没有】规定：分布形状（均匀/高斯/…）、是否零均值、不同地点是否独立、
空间相关长度。因此本模块不假设唯一分布，而是提供**多种都满足 (i)(ii)(iii)** 的
误差场，用于鲁棒性 / 最坏情况测试：

  - iid          不同地点独立 U[-1,1]（最普通随机基准；会奖励“原地微动多测取平均”）
  - smooth       空间相关（相关长度 ℓ 可调；微动几乎测到同一误差 → 惩罚“微动平均”）
  - biased       整体偏置（均值≠0）→ 打击“误差均值为 0”的最小二乘假设
  - adversarial  误差贴近 ±1°（分块或常偏）→ 最坏情况保证
  - piecewise    分区电磁环境，每块各有相关偏差

所有场都严格保证 |error_deg| ≤ 1，且 error_deg(x,y) 对同一 (x,y) 恒定（落实 (ii)）。

配置可序列化：field.config() → {"kind","seed","params"}；make_error_field(**config) 复原。
"""

from __future__ import annotations

import math
import struct
import hashlib
import random
from typing import Optional


def _clamp1(v: float) -> float:
    if v > 1.0:
        return 1.0
    if v < -1.0:
        return -1.0
    return v


class BaseErrorField:
    """误差场基类。子类实现 error_deg(x,y)∈[-1,1]，对同一坐标恒定。"""

    kind = "base"

    def __init__(self, seed: int, **params):
        self.seed = int(seed)
        self.params = dict(params)

    def error_deg(self, x: float, y: float) -> float:  # pragma: no cover - 抽象
        raise NotImplementedError

    def config(self) -> dict:
        return {"kind": self.kind, "seed": self.seed, "params": dict(self.params)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(seed={self.seed}, params={self.params})"


class SmoothErrorField(BaseErrorField):
    """
    空间相关误差场：若干正弦分量之和，振幅归一化使 Σ|amp|=amplitude ≤ 1 → 严格有界。
    特征相关长度由 length_scale 控制（波长在 ℓ 附近对数正态抖动）。
    这是默认场：邻近点误差高度相关，"原地微动多测"几乎测到同一误差。
    """

    kind = "smooth"

    def __init__(self, seed: int, length_scale: float = 300.0,
                 n_components: int = 12, amplitude: float = 1.0):
        super().__init__(seed, length_scale=float(length_scale),
                         n_components=int(n_components), amplitude=float(amplitude))
        self.length_scale = float(length_scale)
        self.n_components = int(n_components)
        self.amplitude = float(amplitude)
        rng = random.Random(self.seed ^ 0x9E3779B9)
        amps = [rng.random() for _ in range(self.n_components)]
        s = sum(amps) or 1.0
        amps = [a / s * self.amplitude for a in amps]   # Σ|amp| = amplitude
        self._comps = []
        for a in amps:
            wavelength = self.length_scale * math.exp(rng.gauss(0.0, 0.5))
            wavelength = max(5.0, wavelength)
            k = 2.0 * math.pi / wavelength
            theta = rng.uniform(0.0, 2.0 * math.pi)
            phase = rng.uniform(0.0, 2.0 * math.pi)
            self._comps.append((a, k * math.cos(theta), k * math.sin(theta), phase))

    def error_deg(self, x: float, y: float) -> float:
        e = 0.0
        for a, kx, ky, ph in self._comps:
            e += a * math.sin(kx * x + ky * y + ph)
        return _clamp1(e)


class IIDErrorField(BaseErrorField):
    """
    地点独立误差场：每个不同 (x,y) 独立 ~ U[-magnitude, magnitude]（默认 magnitude=1）。
    同一坐标恒定（哈希确定）。作为"最普通随机"基准；在此场上"原地微动重采样取平均"
    会异常有效，正好用来暴露依赖该假设的策略在相关场上的脆弱性。
    """

    kind = "iid"

    def __init__(self, seed: int, magnitude: float = 1.0):
        super().__init__(seed, magnitude=float(magnitude))
        self.magnitude = _clamp1(abs(float(magnitude)))

    def error_deg(self, x: float, y: float) -> float:
        b = struct.pack("<ddq", float(x), float(y), self.seed & 0x7FFFFFFFFFFFFFFF)
        h = hashlib.blake2b(b, digest_size=8).digest()
        u = int.from_bytes(h, "little") / float(1 << 64)   # [0,1)
        return _clamp1((2.0 * u - 1.0) * self.magnitude)


class BiasedErrorField(BaseErrorField):
    """
    偏置误差场：ε = clamp(bias + 相关噪声)。均值≈bias（截断略有偏移）。
    专门打击"误差均值为 0"的最小二乘 / 平均去噪方法。
    """

    kind = "biased"

    def __init__(self, seed: int, bias: float = 0.6, noise_amp: float = 0.4,
                 length_scale: float = 300.0, n_components: int = 12):
        super().__init__(seed, bias=float(bias), noise_amp=float(noise_amp),
                         length_scale=float(length_scale), n_components=int(n_components))
        self.bias = _clamp1(float(bias))
        self._noise = SmoothErrorField(seed, length_scale=length_scale,
                                       n_components=n_components,
                                       amplitude=abs(float(noise_amp)))

    def error_deg(self, x: float, y: float) -> float:
        return _clamp1(self.bias + self._noise.error_deg(x, y))


class AdversarialErrorField(BaseErrorField):
    """
    最坏情况误差场：误差贴近 ±magnitude。
      - mode="boundary"：按空间分块取 +mag / -mag（块内恒定，块界翻转）；
      - mode="constant"：全域恒为 +magnitude（最坏常偏，检验最坏保证）。
    专门测试策略在"误差恰好取极值"时是否仍能 100% 清除。
    """

    kind = "adversarial"

    def __init__(self, seed: int, magnitude: float = 1.0,
                 length_scale: float = 150.0, mode: str = "boundary"):
        super().__init__(seed, magnitude=float(magnitude),
                         length_scale=float(length_scale), mode=str(mode))
        self.magnitude = _clamp1(abs(float(magnitude)))
        self.mode = str(mode)
        self._sign = SmoothErrorField(seed, length_scale=length_scale,
                                      n_components=6, amplitude=1.0)

    def error_deg(self, x: float, y: float) -> float:
        if self.mode == "constant":
            return self.magnitude
        v = self._sign.error_deg(x, y)
        s = 1.0 if v >= 0.0 else -1.0
        return _clamp1(s * self.magnitude)


class PiecewiseErrorField(BaseErrorField):
    """
    分区电磁环境：区域被 n_blocks 个随机中心划成 Voronoi 块，每块有各自相关偏差
    b_k ∈ [-block_bias, block_bias]，叠加小幅相关噪声。跨块误差不连续、块内相关。
    """

    kind = "piecewise"

    def __init__(self, seed: int, n_blocks: int = 8, block_bias: float = 0.7,
                 noise_amp: float = 0.2, length_scale: float = 120.0,
                 radius: float = 1800.0):
        super().__init__(seed, n_blocks=int(n_blocks), block_bias=float(block_bias),
                         noise_amp=float(noise_amp), length_scale=float(length_scale),
                         radius=float(radius))
        self.n_blocks = int(n_blocks)
        rng = random.Random(self.seed ^ 0x2545F491)
        self._centers = []
        for _ in range(self.n_blocks):
            r = float(radius) * math.sqrt(rng.random())
            t = rng.uniform(0.0, 2.0 * math.pi)
            self._centers.append((r * math.cos(t), r * math.sin(t)))
        self._biases = [rng.uniform(-abs(block_bias), abs(block_bias))
                        for _ in range(self.n_blocks)]
        self._noise = SmoothErrorField(self.seed + 1, length_scale=length_scale,
                                       n_components=6, amplitude=abs(float(noise_amp)))

    def error_deg(self, x: float, y: float) -> float:
        best_i, best_d = 0, float("inf")
        for i, (cx, cy) in enumerate(self._centers):
            d = (x - cx) * (x - cx) + (y - cy) * (y - cy)
            if d < best_d:
                best_d, best_i = d, i
        return _clamp1(self._biases[best_i] + self._noise.error_deg(x, y))


_REGISTRY = {
    "smooth": SmoothErrorField,
    "iid": IIDErrorField,
    "biased": BiasedErrorField,
    "adversarial": AdversarialErrorField,
    "piecewise": PiecewiseErrorField,
}

FIELD_KINDS = tuple(_REGISTRY.keys())


def make_error_field(kind: str = "smooth", seed: int = 0, params: Optional[dict] = None,
                     **kwargs) -> BaseErrorField:
    """
    工厂：按 kind 构造误差场。params（来自 config()）与 kwargs 合并传入构造器。
    未知 kind → ValueError。
    """
    cls = _REGISTRY.get(kind)
    if cls is None:
        raise ValueError(f"unknown error field kind: {kind!r}; valid: {FIELD_KINDS}")
    merged = dict(params or {})
    merged.update(kwargs)
    return cls(seed=seed, **merged)
