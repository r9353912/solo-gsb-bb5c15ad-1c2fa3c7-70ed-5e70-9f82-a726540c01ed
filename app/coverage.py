"""展柜采样覆盖率：周期时隙与已合并读数的一对一容差匹配（纯计算，只读）。

对每个展柜的**已合并读数**评估周期性应采计划的覆盖情况（全程不修改
读数、分段或版本号）：

1. **时隙生成**：以 ``start`` 为首个应采时刻，按 ``period_seconds``
   等距推进，只保留**不晚于** ``end`` 的时隙（恰好等于 ``end`` 计入）；
   时隙数 ``= floor((end − start) / period) + 1``（``start ≤ end``
   由接口层保证，故至少 1 个）；
2. **容差扩展取数**：只取 ``[start − tolerance, 末时隙 + tolerance]``
   内的读数（范围外的读数不可能覆盖任何时隙），按时间升序逐一匹配；
3. **一对一匹配**：读数可覆盖满足 ``|读数时刻 − 时隙时刻| ≤ tolerance``
   （**含等于**）的时隙；每条读数最多覆盖**一个**时隙、每个时隙最多
   接收**一条**读数；读数按时间顺序优先匹配**最早可覆盖**（下标最小）
   的时隙，该时隙被占后由后续读数递补；
4. **缺口合并**：连续缺测时隙合并为一个缺口，给出首/末缺测时隙的
   应采时刻与缺测数；缺口按时间升序。

全部计算确定性：相同输入恒得相同结果。匹配过程按时隙下标单调推进，
不逐时隙展开，时隙数很大时开销仍只与读数数量级相关。
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Protocol


class _SampleLike(Protocol):
    ts: dt.datetime


@dataclass(frozen=True)
class CoverageGap:
    """一段连续缺测时隙合并出的缺口（时刻为首/末缺测时隙的应采时刻）。"""

    start_ts: dt.datetime
    end_ts: dt.datetime
    missing_count: int


@dataclass(frozen=True)
class SamplingCoverageResult:
    expected_count: int
    covered_count: int
    gaps: list[CoverageGap]


def compute_sampling_coverage(
    samples: list[_SampleLike],
    *,
    start: dt.datetime,
    end: dt.datetime,
    period_seconds: float,
    tolerance_seconds: float,
) -> SamplingCoverageResult:
    """对周期时隙与已合并读数做一对一容差匹配（只读纯计算）。

    samples 无需预先排序；同一展柜的读数应已按 (device_sn, ts) 合并。
    参数合法性（``start ≤ end``、``period_seconds > 0``、
    ``0 ≤ tolerance_seconds < period_seconds``）由接口层校验，
    本函数假定入参合法。
    """
    period = dt.timedelta(seconds=period_seconds)
    tolerance = dt.timedelta(seconds=tolerance_seconds)

    # 时隙数：floor((end − start) / period) + 1；timedelta 乘法精确到微秒
    span = (end - start).total_seconds()
    n_slots = int(math.floor(span / period_seconds)) + 1
    # 浮点换算边界微调：保证 slot_{n-1} ≤ end < slot_n
    while n_slots > 1 and start + (n_slots - 1) * period > end:
        n_slots -= 1
    while start + n_slots * period <= end:
        n_slots += 1

    def slot_ts(index: int) -> dt.datetime:
        return start + index * period

    # 容差扩展范围外的读数不可能覆盖任何时隙
    lower_bound = start - tolerance
    upper_bound = slot_ts(n_slots - 1) + tolerance
    readings = sorted(
        (s for s in samples if lower_bound <= s.ts <= upper_bound),
        key=lambda s: s.ts,
    )

    missed: list[list[int]] = []  # 缺测时隙下标闭区间 [首, 末]，按时间顺序

    def record_missed(first: int, last: int) -> None:
        """登记缺测时隙闭区间；与上一段相邻（中间无已覆盖时隙）则合并。"""
        if first > last:
            return
        if missed and missed[-1][1] == first - 1:
            missed[-1][1] = last
        else:
            missed.append([first, last])

    covered = 0
    j = 0  # 最早未覆盖时隙下标（之前的时隙已覆盖或已记缺测）
    for reading in readings:
        # 对当前及之后读数都太早（时隙 + tolerance < 读数时刻）的时隙
        # 永远不可能再被覆盖，记为缺测；即下标 < lo 的时隙，其中 lo 为
        # 满足 slot_lo ≥ 读数时刻 − tolerance 的最小下标
        lo = max(
            0,
            math.ceil(
                ((reading.ts - start).total_seconds() - tolerance_seconds)
                / period_seconds
            ),
        )
        lower = reading.ts - tolerance
        # 浮点估算可能偏差 ±1，用精确比较微调
        while lo > 0 and slot_ts(lo - 1) >= lower:
            lo -= 1
        while lo < n_slots and slot_ts(lo) < lower:
            lo += 1
        if lo > j:
            record_missed(j, lo - 1)
            j = lo
        # 最早未覆盖时隙在容差内即覆盖之（亦即最早可覆盖时隙）；
        # 否则读数早于 slot_j − tolerance，不覆盖任何时隙
        if j < n_slots and slot_ts(j) <= reading.ts + tolerance:
            covered += 1
            j += 1
    if j < n_slots:
        record_missed(j, n_slots - 1)

    return SamplingCoverageResult(
        expected_count=n_slots,
        covered_count=covered,
        gaps=[
            CoverageGap(
                start_ts=slot_ts(first),
                end_ts=slot_ts(last),
                missing_count=last - first + 1,
            )
            for first, last in missed
        ],
    )
