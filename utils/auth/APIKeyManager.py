import os
import re
import time
import random
from typing import List, Dict, Optional, Tuple


class APIOverloadException(Exception):
    """Exception raised when all API keys are overloaded or unavailable"""
    pass


class APIKeyManager:
    """
    Quản lý pool API key với:
    - Cooldown theo RetryInfo/Retry-After (khi dính 429/quota).
    - Trọng số (weighted) để ưu tiên key 'xịn' hơn.
    - Đánh dấu lỗi tạm thời (5xx) và lỗi vĩnh viễn (key invalid).
    - Chọn key khả dụng tiếp theo (bỏ qua key đang cooldown/failed).

    ENV hỗ trợ:
      GEMINI_API_KEY           = "sk-..."                 # 1 key
      GEMINI_API_KEYS          = "sk-1, sk-2, sk-3"       # nhiều key
      GEMINI_KEY_WEIGHTS       = "1,3,1"                  # (tuỳ chọn) cùng số phần tử với keys
      GEMINI_DEFAULT_COOLDOWN  = "60"                     # (tuỳ) giây khi không có RetryInfo
      GEMINI_TRANSIENT_COOLDOWN= "10"                     # (tuỳ) giây cho lỗi tạm thời (5xx)
    """

    _RETRY_DELAY_PATTERNS = [
        re.compile(r"retryDelay['\"]?\s*[:=]\s*['\"]?(\d+)s", re.I),
        re.compile(r"retry in ([0-9.]+)s", re.I),
        re.compile(r"Retry-After:\s*([0-9]+)", re.I),  # header dạng giây
    ]

    def __init__(self) -> None:
        self.api_keys: List[str] = []
        self.failed_keys: set[int] = set()        # lỗi vĩnh viễn (invalid key)
        self.cooldown_until: Dict[int, float] = {}  # idx -> unix_ts
        self.weights: Dict[int, int] = {}         # idx -> weight >=1
        self.current_idx: Optional[int] = None
        self.default_cooldown = int(os.getenv("GEMINI_DEFAULT_COOLDOWN", "60"))
        self.transient_cooldown = int(os.getenv("GEMINI_TRANSIENT_COOLDOWN", "10"))
        self._load_api_keys()
        self._load_weights()
        random.shuffle(self.api_keys)  # tránh dồn 1 key nếu không set weight
        # map lại weights theo thứ tự sau shuffle
        self._reindex_weights_after_shuffle()

    # ---------- Loaders ----------

    def _load_api_keys(self) -> None:
        multi = os.getenv("GEMINI_API_KEYS", "")
        if multi.strip():
            self.api_keys = [k.strip() for k in multi.split(",") if k.strip()]
        else:
            single = os.getenv("GEMINI_API_KEY", "").strip()
            if not single:
                raise RuntimeError("No Gemini API keys configured")
            self.api_keys = [single]

        if not self.api_keys:
            raise RuntimeError("Empty key pool")

    def _load_weights(self) -> None:
        raw = os.getenv("GEMINI_KEY_WEIGHTS", "").strip()
        if not raw:
            # mặc định tất cả weight=1
            self.weights = {i: 1 for i in range(len(self.api_keys))}
            return

        parts = [p.strip() for p in raw.split(",")]
        ints: List[int] = []
        for p in parts:
            try:
                w = int(p)
                ints.append(max(1, w))
            except Exception:
                ints.append(1)

        # khớp chiều dài với số key
        if len(ints) < len(self.api_keys):
            ints += [1] * (len(self.api_keys) - len(ints))
        elif len(ints) > len(self.api_keys):
            ints = ints[:len(self.api_keys)]

        self.weights = {i: ints[i] for i in range(len(self.api_keys))}

    def _reindex_weights_after_shuffle(self) -> None:
        # Sau khi shuffle keys, weight theo index cũ không còn đúng.
        # Ở đây đơn giản: reset về 1 hết, hoặc bạn có thể bỏ shuffle
        # nếu muốn giữ đúng weight đã mapping theo thứ tự keys ban đầu.
        if not self.weights:
            self.weights = {i: 1 for i in range(len(self.api_keys))}
        else:
            self.weights = {i: self.weights.get(i, 1) for i in range(len(self.api_keys))}

    # ---------- Core Utils ----------

    def _now(self) -> float:
        return time.time()

    def is_available(self, idx: int) -> bool:
        if idx in self.failed_keys:
            return False
        until = self.cooldown_until.get(idx, 0.0)
        return self._now() >= until

    def set_cooldown(self, idx: int, seconds: int) -> None:
        seconds = max(1, int(seconds))
        self.cooldown_until[idx] = self._now() + seconds

    def clear_cooldown(self, idx: int) -> None:
        self.cooldown_until.pop(idx, None)

    def reset_failed_keys(self) -> None:
        self.failed_keys.clear()

    def parse_retry_delay(self, err_msg: str) -> int:
        if not err_msg:
            return self.default_cooldown
        for pat in self._RETRY_DELAY_PATTERNS:
            m = pat.search(err_msg)
            if m:
                try:
                    return max(1, int(float(m.group(1))))
                except Exception:
                    continue
        return self.default_cooldown

    # ---------- Markers ----------

    def mark_quota_exhausted(self, idx: int, err_msg: Optional[str] = None) -> None:
        """429/RESOURCE_EXHAUSTED → cooldown theo RetryInfo/Retry-After."""
        delay = self.parse_retry_delay(err_msg or "")
        self.set_cooldown(idx, delay)

    def mark_transient_error(self, idx: int) -> None:
        """5xx/overload → cooldown ngắn để tránh dội cùng key."""
        self.set_cooldown(idx, self.transient_cooldown)

    def mark_permanent_fail(self, idx: int) -> None:
        """Key invalid (401/403/404 NOT_FOUND model, v.v.) → loại khỏi pool."""
        self.failed_keys.add(idx)
        self.clear_cooldown(idx)

    # ---------- Picking ----------

    def _weighted_random_choice(self, candidates: List[int]) -> int:
        # Chọn theo trọng số (đơn giản, đủ dùng)
        weights = [max(1, self.weights.get(i, 1)) for i in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]

    def pick_key(self) -> Tuple[str, int]:
        """
        Trả về (api_key, idx). Ưu tiên key KHẢ DỤNG (không cooldown, không failed).
        Nếu tất cả đang cooldown → chọn key có cooldown sắp hết nhất.
        """
        available = [i for i in range(len(self.api_keys)) if self.is_available(i)]
        if available:
            idx = self._weighted_random_choice(available)
            self.current_idx = idx
            return self.api_keys[idx], idx

        # không có key available → chọn key có cooldown ngắn nhất (để gọi sớm nhất có thể)
        idx_min = min(range(len(self.api_keys)), key=lambda i: self.cooldown_until.get(i, float("inf")))
        self.current_idx = idx_min
        return self.api_keys[idx_min], idx_min

    # ---------- Introspection ----------

    def status(self) -> Dict:
        return {
            "total": len(self.api_keys),
            "failed": sorted(list(self.failed_keys)),
            "cooldowns": {i: max(0, int(self.cooldown_until.get(i, 0) - self._now()))
                          for i in range(len(self.api_keys))
                          if self.cooldown_until.get(i, 0) > self._now()},
            "weights": {i: self.weights.get(i, 1) for i in range(len(self.api_keys))},
            "current_idx": self.current_idx,
        }
api_manager = APIKeyManager()
