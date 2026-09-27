"""采样覆盖率：周期时隙 + 容差一对一匹配（接口集成测试）。"""

from __future__ import annotations

import os

os.environ["CASE_DB_URL"] = "sqlite:///:memory:"
os.environ["CASE_PENALTY"] = "4.0"

import pytest
from fastapi.testclient import TestClient

from app.database import (
    Base,
    CaseState,
    Reading,
    SegmentSnapshot,
    SessionLocal,
    engine,
    init_db,
)
from app.main import app


@pytest.fixture()
def client():
    Base.metadata.drop_all(bind=engine)
    init_db()
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def _reading(sn, ts, t=20.0, h=50.0):
    return {"device_sn": sn, "ts": ts, "temperature": t, "humidity": h}


def _ts(minutes, day=1):
    hh, mm = divmod(minutes, 60)
    return f"2026-09-{day:02d}T{hh:02d}:{mm:02d}:00Z"


# 时隙：00:00 / 00:10 / 00:20 / 00:30，容差 ±60 秒
PARAMS = {
    "start": "2026-09-01T00:00:00Z",
    "end": "2026-09-01T00:30:00Z",
    "period_seconds": 600,
    "tolerance_seconds": 60,
}


def _get(client, sn, params=None):
    return client.get(f"/api/v1/cases/{sn}/sampling-coverage", params=params or PARAMS)


# ---------- 基本覆盖与缺口 ----------


def test_full_coverage_no_gaps(client):
    payload = [_reading("CASE-S1", _ts(m)) for m in (0, 10, 20, 30)]
    client.post("/api/v1/readings/batch", json={"readings": payload})

    r = _get(client, "CASE-S1")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["device_sn"] == "CASE-S1"
    assert body["start"] == "2026-09-01T00:00:00Z"
    assert body["end"] == "2026-09-01T00:30:00Z"
    assert body["period_seconds"] == 600.0
    assert body["tolerance_seconds"] == 60.0
    assert body["expected_count"] == 4
    assert body["covered_count"] == 4
    assert body["missing_count"] == 0
    assert body["coverage_ratio"] == pytest.approx(1.0)
    assert body["gap_count"] == 0
    assert body["gaps"] == []


def test_missing_slots_merged_into_single_gap(client):
    payload = [_reading("CASE-S2", _ts(m)) for m in (0, 30)]
    client.post("/api/v1/readings/batch", json={"readings": payload})

    body = _get(client, "CASE-S2").json()
    assert body["expected_count"] == 4
    assert body["covered_count"] == 2
    assert body["missing_count"] == 2
    assert body["coverage_ratio"] == pytest.approx(0.5)
    assert body["gap_count"] == 1
    assert body["gaps"] == [
        {
            "start_ts": "2026-09-01T00:10:00Z",
            "end_ts": "2026-09-01T00:20:00Z",
            "missing_count": 2,
        }
    ]


def test_head_and_tail_gaps(client):
    # 只有 00:10 一条读数：首槽 00:00 与末两槽 00:20、00:30 各成缺口
    client.post(
        "/api/v1/readings/batch",
        json={"readings": [_reading("CASE-S3", _ts(10))]},
    )
    body = _get(client, "CASE-S3").json()
    assert body["covered_count"] == 1
    assert body["gaps"] == [
        {
            "start_ts": "2026-09-01T00:00:00Z",
            "end_ts": "2026-09-01T00:00:00Z",
            "missing_count": 1,
        },
        {
            "start_ts": "2026-09-01T00:20:00Z",
            "end_ts": "2026-09-01T00:30:00Z",
            "missing_count": 2,
        },
    ]


def test_no_readings_in_window_all_missed(client):
    # 展柜有读数但全部落在查询窗扩展范围外：全部时隙缺测
    client.post(
        "/api/v1/readings/batch",
        json={"readings": [_reading("CASE-S4", _ts(0, day=2))]},
    )
    r = _get(client, "CASE-S4")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["covered_count"] == 0
    assert body["coverage_ratio"] == pytest.approx(0.0)
    assert body["gaps"] == [
        {
            "start_ts": "2026-09-01T00:00:00Z",
            "end_ts": "2026-09-01T00:30:00Z",
            "missing_count": 4,
        }
    ]


# ---------- 容差边界 ----------


def test_tolerance_boundary_inclusive(client):
    # 偏差恰好等于容差（±60 秒整）仍可覆盖
    payload = [
        _reading("CASE-S5", "2026-09-01T00:01:00Z"),
        _reading("CASE-S5", "2026-09-01T00:11:00Z"),
        _reading("CASE-S5", "2026-09-01T00:19:00Z"),
        _reading("CASE-S5", "2026-09-01T00:29:00Z"),
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = _get(client, "CASE-S5").json()
    assert body["covered_count"] == 4
    assert body["gaps"] == []


def test_beyond_tolerance_not_covered(client):
    # 61 秒偏差 > 容差 60：读数在扩展范围内但不覆盖任何时隙
    payload = [
        _reading("CASE-S6", "2026-09-01T00:01:01Z"),
        _reading("CASE-S6", "2026-09-01T00:11:01Z"),
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = _get(client, "CASE-S6").json()
    assert body["covered_count"] == 0
    assert body["gaps"] == [
        {
            "start_ts": "2026-09-01T00:00:00Z",
            "end_ts": "2026-09-01T00:30:00Z",
            "missing_count": 4,
        }
    ]


def test_zero_tolerance_requires_exact_hit(client):
    params = {**PARAMS, "tolerance_seconds": 0}
    payload = [
        _reading("CASE-S7", _ts(0)),
        _reading("CASE-S7", "2026-09-01T00:10:01Z"),  # 偏 1 秒：不覆盖
        _reading("CASE-S7", _ts(30)),
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = _get(client, "CASE-S7", params).json()
    assert body["covered_count"] == 2
    assert body["gaps"] == [
        {
            "start_ts": "2026-09-01T00:10:00Z",
            "end_ts": "2026-09-01T00:20:00Z",
            "missing_count": 2,
        }
    ]


# ---------- 一对一匹配 ----------


def test_earliest_coverable_slot_preferred(client):
    # 容差 400：00:05 的读数同时可覆盖 00:00 与 00:10，必须取最早的 00:00；
    # 00:07 的读数只能覆盖 00:10（距 00:00 为 420 秒 > 400）
    params = {**PARAMS, "end": "2026-09-01T00:10:00Z", "tolerance_seconds": 400}
    payload = [
        _reading("CASE-S8", "2026-09-01T00:05:00Z"),
        _reading("CASE-S8", "2026-09-01T00:07:00Z"),
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = _get(client, "CASE-S8", params).json()
    assert body["expected_count"] == 2
    assert body["covered_count"] == 2
    assert body["gaps"] == []


def test_one_slot_accepts_at_most_one_reading(client):
    # 单时隙：两条读数都在容差内，先到的覆盖，后到的不重复计入
    params = {**PARAMS, "end": "2026-09-01T00:00:00Z"}
    payload = [
        _reading("CASE-S9", "2026-09-01T00:00:10Z"),
        _reading("CASE-S9", "2026-09-01T00:00:20Z"),
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = _get(client, "CASE-S9", params).json()
    assert body["expected_count"] == 1
    assert body["covered_count"] == 1
    assert body["coverage_ratio"] == pytest.approx(1.0)
    assert body["gaps"] == []


def test_taken_slot_backfilled_by_later_reading(client):
    # 00:00 的读数占去首槽后，00:00:30 的读数不能重复覆盖首槽，
    # 也不能覆盖 00:10（570 秒 > 60）：首槽覆盖一次，其余缺测
    payload = [
        _reading("CASE-S10", _ts(0)),
        _reading("CASE-S10", "2026-09-01T00:00:30Z"),
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = _get(client, "CASE-S10").json()
    assert body["covered_count"] == 1
    assert body["missing_count"] == 3


def test_merged_duplicate_reports_cover_once(client):
    # 同时间戳重复上报先合并为一条：只覆盖一个时隙
    payload = [
        _reading("CASE-S11", _ts(0), t=20.0),
        _reading("CASE-S11", _ts(0), t=22.0),
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = _get(client, "CASE-S11").json()
    assert body["covered_count"] == 1
    assert body["missing_count"] == 3


# ---------- 时隙生成与扩展范围 ----------


def test_slot_generation_stops_at_end(client):
    payload = [_reading("CASE-S12", _ts(m)) for m in (0, 10, 20, 30)]
    client.post("/api/v1/readings/batch", json={"readings": payload})

    # end=00:25：00:30 时隙不生成
    body = _get(client, "CASE-S12", {**PARAMS, "end": "2026-09-01T00:25:00Z"}).json()
    assert body["expected_count"] == 3
    assert body["covered_count"] == 3

    # end=00:30：恰好等于 end 的时隙计入
    body = _get(client, "CASE-S12", PARAMS).json()
    assert body["expected_count"] == 4
    assert body["covered_count"] == 4


def test_readings_outside_extended_range_ignored(client):
    # 扩展范围为 [00:00−60s, 00:30+60s]：范围外读数不参与匹配
    payload = [
        _reading("CASE-S13", "2026-08-31T23:58:00Z"),  # 早于 start−容差
        _reading("CASE-S13", "2026-09-01T00:00:30Z"),  # 覆盖 00:00
        _reading("CASE-S13", "2026-09-01T00:32:00Z"),  # 晚于末时隙+容差
    ]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = _get(client, "CASE-S13").json()
    assert body["covered_count"] == 1
    assert body["gaps"] == [
        {
            "start_ts": "2026-09-01T00:10:00Z",
            "end_ts": "2026-09-01T00:30:00Z",
            "missing_count": 3,
        }
    ]


def test_fractional_period_slots(client):
    # 非整秒周期：90 秒周期，00:00 起 40 分钟窗 → 00:00…00:39 共 27 槽
    params = {
        "start": "2026-09-01T00:00:00Z",
        "end": "2026-09-01T00:40:00Z",
        "period_seconds": 90,
        "tolerance_seconds": 10,
    }
    payload = [_reading("CASE-S14", "2026-09-01T00:00:00Z")]
    client.post("/api/v1/readings/batch", json={"readings": payload})
    body = _get(client, "CASE-S14", params).json()
    assert body["expected_count"] == 27
    assert body["covered_count"] == 1
    assert body["gaps"] == [
        {
            "start_ts": "2026-09-01T00:01:30Z",
            "end_ts": "2026-09-01T00:39:00Z",
            "missing_count": 26,
        }
    ]


# ---------- 404 / 422 ----------


def test_unknown_case_404(client):
    r = _get(client, "NOPE")
    assert r.status_code == 404


def test_start_after_end_422(client):
    client.post(
        "/api/v1/readings/batch",
        json={"readings": [_reading("CASE-S15", _ts(0))]},
    )
    r = _get(
        client,
        "CASE-S15",
        {**PARAMS, "start": "2026-09-01T01:00:00Z", "end": "2026-09-01T00:00:00Z"},
    )
    assert r.status_code == 422


def test_validation_422(client):
    client.post(
        "/api/v1/readings/batch",
        json={"readings": [_reading("CASE-S16", _ts(0))]},
    )
    base = "/api/v1/cases/CASE-S16/sampling-coverage"
    ok = "start=2026-09-01T00:00:00Z&end=2026-09-01T00:30:00Z&period_seconds=600&tolerance_seconds=60"
    assert client.get(f"{base}?{ok}").status_code == 200
    # 缺少必填参数
    assert client.get(base).status_code == 422
    assert client.get(f"{base}?end=2026-09-01T00:30:00Z&period_seconds=600&tolerance_seconds=60").status_code == 422
    assert client.get(f"{base}?start=2026-09-01T00:00:00Z&period_seconds=600&tolerance_seconds=60").status_code == 422
    assert client.get(f"{base}?start=2026-09-01T00:00:00Z&end=2026-09-01T00:30:00Z&tolerance_seconds=60").status_code == 422
    assert client.get(f"{base}?start=2026-09-01T00:00:00Z&end=2026-09-01T00:30:00Z&period_seconds=600").status_code == 422
    # 周期必须为正
    for bad in ("0", "-600"):
        assert client.get(f"{base}?{ok.replace('period_seconds=600', f'period_seconds={bad}')}").status_code == 422
    # 容差必须 ≥ 0 且 < 周期
    assert client.get(f"{base}?{ok.replace('tolerance_seconds=60', 'tolerance_seconds=-1')}").status_code == 422
    assert client.get(f"{base}?{ok.replace('tolerance_seconds=60', 'tolerance_seconds=600')}").status_code == 422
    assert client.get(f"{base}?{ok.replace('tolerance_seconds=60', 'tolerance_seconds=601')}").status_code == 422
    # 非法时间格式
    assert client.get(f"{base}?{ok.replace('start=2026-09-01T00:00:00Z', 'start=not-a-time')}").status_code == 422


# ---------- 只读 & 确定性 ----------


def test_read_only_and_deterministic(client, db):
    payload = [_reading("CASE-S17", _ts(m)) for m in (0, 30)]
    client.post("/api/v1/readings/batch", json={"readings": payload})

    state = db.get(CaseState, "CASE-S17")
    version_before = state.version
    readings_before = [
        (r.ts, r.temperature, r.humidity, r.report_count)
        for r in db.query(Reading).filter(Reading.device_sn == "CASE-S17").order_by(Reading.ts)
    ]
    snaps_before = db.query(SegmentSnapshot).filter(
        SegmentSnapshot.device_sn == "CASE-S17"
    ).count()

    r1 = _get(client, "CASE-S17")
    r2 = _get(client, "CASE-S17")
    assert r1.status_code == 200
    assert r1.json() == r2.json()

    db.expire_all()
    assert db.get(CaseState, "CASE-S17").version == version_before
    readings_after = [
        (r.ts, r.temperature, r.humidity, r.report_count)
        for r in db.query(Reading).filter(Reading.device_sn == "CASE-S17").order_by(Reading.ts)
    ]
    assert readings_after == readings_before
    assert (
        db.query(SegmentSnapshot).filter(
            SegmentSnapshot.device_sn == "CASE-S17"
        ).count()
        == snaps_before
    )
