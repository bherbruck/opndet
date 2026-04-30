from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    img_h: int = 384
    img_w: int = 512
    in_ch: int = 3
    stride: int = 4
    base_ch: int = 32
    stage_ch: tuple[int, ...] = (64, 128, 192, 256)
    stage_n: tuple[int, ...] = (1, 2, 3, 2)
    neck_ch: int = 64
    head_ch: int = 64
    peak_kernel: int = 3
    out_h: int = 384 // 4
    out_w: int = 512 // 4

    def __post_init__(self):
        assert self.img_h % 32 == 0 and self.img_w % 32 == 0, "dims must be /32"
        assert self.stride in (2, 4), "stride 2 or 4 only"
        assert len(self.stage_ch) == 4 and len(self.stage_n) == 4
