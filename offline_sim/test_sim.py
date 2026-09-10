"""
自测：验证离线模拟器严格符合文件规则。分两部分：

A. 引擎级单元测试（不经 HTTP）：
   - 复现附件2 §10 / 附件1 §3 的计时示例，虚拟时刻必须精确为 105 / 111 / 194 / 199。
   - 物理判定：全向/定向覆盖、near、no_signal、清除半径、示向度误差范围与“同点重复不变”。

B. HTTP 协议级测试（起真实服务器 + urllib 客户端）：
   - 200/accepted、200/accepted=false（未知字段、robot_id 不符、重复 enter）、
     400（缺字段/坐标越界/channel 非整数）、404、405、415、幂等复用与 409、结束后连接关闭。

运行：python -m offline_sim.test_sim
"""

from __future__ import annotations

import math
import threading
import time

import random

from .case import (Case, Jammer, ErrorField, generate_case, generate_stress_case,
                   norm_deg, ang_diff, STRESS_TYPES, ARENA_RADIUS_M)
from .engine import (Engine, NEAR_THRESHOLD_M, CLEAR_RADIUS_M,
                     SPEED_MPS, MEASURE_ACTION_S, CH_SWITCH_S)
from .fields import (make_error_field, FIELD_KINDS, IIDErrorField, SmoothErrorField,
                     BiasedErrorField, AdversarialErrorField, PiecewiseErrorField)
from .server import SimSession, SimServer, _content_type_ok
from .client import SimClient
from .harness import run_episode, evaluate, aggregate, EpisodeResult


PASS, FAIL = "  [PASS]", "  [FAIL]"
_results = []


def check(name, cond, extra=""):
    _results.append(bool(cond))
    print((PASS if cond else FAIL), name, ("" if cond else "-> " + extra))


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# ---------------- A. 引擎级 ----------------
def build_timing_case() -> Case:
    """
    构造与附件2 §10 计时示例一致的场景：
    步骤 2 measure ch1 @ (300,400) → 有信号；步骤 3 measure ch2 @ (300,400)；
    步骤 4 clear ch3 @ (300,0) → 未发现；步骤 5 measure ch2 @ (300,0)。
    为使耗时与示例一致，示例本身与检测结果无关（耗时只取决于移动/切换/动作），
    因此干扰源布局只需保证 ch3 在 (300,0) 20m 内“无可清除目标”。
    这里放几个远处的源即可。
    """
    jammers = [
        Jammer(1, 1000.0, 0.0, 1200.0, "omni"),
        Jammer(2, -1000.0, 0.0, 1200.0, "omni"),
        Jammer(3, -1500.0, 100.0, 1200.0, "omni"),   # 离 (300,0) 远，clear 必然 no_target
    ]
    field = ErrorField(seed=123)
    return Case(jammers, field, seed=1)


def test_timing_example():
    eng = Engine(build_timing_case())
    eng.enter()
    check("§10 步1 /enter 虚拟时刻=0", approx(eng.st.virtual_time_s, 0.0), str(eng.st.virtual_time_s))

    # 步2: (0,0)->(300,400) 距离500 移动100 + ch1 与初始ch1 相同(0) + 检测5 = 105
    resp, _ = eng.measure(300, 400, 1)
    check("§10 步2 measure ch1 @(300,400) → 105", approx(resp["virtual_time_s"], 105.0),
          str(resp["virtual_time_s"]))

    # 步3: 原地 + ch1->ch2 切换1 + 检测5 = +6 = 111
    resp, _ = eng.measure(300, 400, 2)
    check("§10 步3 measure ch2 原地 → 111", approx(resp["virtual_time_s"], 111.0),
          str(resp["virtual_time_s"]))

    # 步4: (300,400)->(300,0) 距离400 移动80 + clear未发现3 = +83 = 194
    resp, _ = eng.clear(300, 0, 3)
    check("§10 步4 clear ch3 @(300,0) 未发现 → 194", approx(resp["virtual_time_s"], 194.0),
          str(resp["virtual_time_s"]))

    # 步5: 原地 measure ch2（当前频道仍为步3的ch2，不切）+ 检测5 = +5 = 199
    resp, _ = eng.measure(300, 0, 2)
    check("§10 步5 measure ch2 原地(频道未变) → 199", approx(resp["virtual_time_s"], 199.0),
          str(resp["virtual_time_s"]))


def test_clear_timing_hit():
    """成功清除耗时 = 移动 + 5s。"""
    j = Jammer(5, 100.0, 0.0, 1200.0, "omni")
    eng = Engine(Case([j], ErrorField(1), 1))
    eng.enter()
    # 到 (100,0) 移动 100/5=20s，命中(距离0≤20) → +5 = 25
    resp, out = eng.clear(100, 0, 5)
    check("clear 命中耗时=移动20+5=25", approx(resp["virtual_time_s"], 25.0), str(resp["virtual_time_s"]))
    check("clear 命中 result=success", out.result == "success", out.result)
    # 重复清除同一源 → no_target（同源只能清一次）
    resp2, out2 = eng.clear(100, 0, 5)
    check("重复清除 → no_target_in_range", out2.result == "no_target_in_range", out2.result)


def test_near_and_nosignal():
    j = Jammer(7, 0.0, 0.0, 1200.0, "omni")
    eng = Engine(Case([j], ErrorField(1), 1))
    eng.enter()
    # 距离 3m ≤5 → near
    _, out = eng.measure(3, 0, 7)
    check("距离≤5m → near（无 svd）", out.result == "near" and out.svd_deg is None, out.result)
    # 距离 500m ≤ R_eff → direction
    _, out = eng.measure(500, 0, 7)
    check("在接收半径内 → direction", out.result == "direction", out.result)
    # 距离 1300m > R_eff(1200) → no_signal
    _, out = eng.measure(1300, 0, 7)
    check("超接收半径 → no_signal", out.result == "no_signal", out.result)
    # 不存在的频道 → no_signal
    _, out = eng.measure(10, 0, 9)
    check("无源频道 → no_signal", out.result == "no_signal", out.result)


def test_directional_coverage():
    # 定向源在原点，方向 0°（正东），覆盖 [-90°,90°] 即东半平面
    j = Jammer(8, 0.0, 0.0, 1200.0, "dir", direction_deg=0.0)
    eng = Engine(Case([j], ErrorField(1), 1))
    eng.enter()
    # 检测点在正东 (500,0)：源指向检测点方向=0°，在覆盖内 → direction
    _, out = eng.measure(500, 0, 8)
    check("定向源-覆盖内(正东) → direction", out.result == "direction", out.result)
    # 检测点在正西 (-500,0)：源指向检测点方向=180°，与0°夹角180>90 → no_signal
    _, out = eng.measure(-500, 0, 8)
    check("定向源-盲区(正西) → no_signal", out.result == "no_signal", out.result)
    # 边界：正北 (0,500)：源指向检测点=90°，夹角=90（含边界）→ direction
    _, out = eng.measure(0, 500, 8)
    check("定向源-边界(正北,夹角90含边界) → direction", out.result == "direction", out.result)
    # 定向源清除不受朝向限制：在盲区侧 20m 内也能清
    _, out = eng.clear(-10, 0, 8)
    check("定向源清除不受朝向限制(盲区侧20m内可清)", out.result == "success", out.result)


def test_svd_error_bounds_and_repeatability():
    j = Jammer(9, 800.0, 600.0, 1400.0, "omni")
    case = Case([j], ErrorField(seed=777), 1)
    eng = Engine(case)
    eng.enter()
    max_err = 0.0
    pts = [(0, 0), (100, 50), (-200, 300), (500, -400), (700, 100)]
    for (x, y) in pts:
        _, out = eng.measure(x, y, 9)
        if out.result == "direction":
            true_b = norm_deg(math.degrees(math.atan2(j.y - y, j.x - x)))
            d = abs(((out.svd_deg - true_b + 180) % 360) - 180)
            max_err = max(max_err, d)
    check("示向度误差 ≤ 1°（+两位小数舍入容差）", max_err <= 1.0 + 5e-3, f"max_err={max_err:.4f}")

    # 同一地点重复测量：误差不变（示向度完全一致）
    eng2 = Engine(Case([j], ErrorField(seed=777), 1))
    eng2.enter()
    _, o1 = eng2.measure(123.0, -45.0, 9)
    _, o2 = eng2.measure(123.0, -45.0, 9)   # 原地再测
    same = (o1.result == "direction" and o2.result == "direction"
            and approx(o1.svd_deg, o2.svd_deg, 1e-9))
    check("同一地点重复测量示向度不变", same, f"{o1.svd_deg} vs {o2.svd_deg}")


def test_channel_state_update():
    """measure 后当前频道更新；clear 不改当前频道。"""
    j1 = Jammer(1, 500, 0, 1200, "omni")
    j4 = Jammer(4, 0, 500, 1200, "omni")
    eng = Engine(Case([j1, j4], ErrorField(1), 1))
    eng.enter()
    check("初始当前频道=1", eng.st.channel == 1)
    eng.measure(10, 0, 4)                 # 切到4
    check("measure后当前频道=4", eng.st.channel == 4, str(eng.st.channel))
    eng.clear(10, 0, 1)                   # clear ch1，不应改当前频道
    check("clear后当前频道仍=4", eng.st.channel == 4, str(eng.st.channel))
    # 再 measure ch4：与当前频道相同 → 无切换耗时
    before = eng.st.virtual_time_s
    resp, _ = eng.measure(10, 0, 4)       # 原地、频道不变 → 只加5
    check("measure频道未变无切换耗时(仅+5)", approx(resp["virtual_time_s"] - before, 5.0),
          str(resp["virtual_time_s"] - before))


def test_case_generation_constraints():
    for prob in (3, 4):
        for k in range(5):
            c = generate_case(seed=1000 + k, problem=prob)
            check(f"生成案例 总数∈[10,16] (p{prob},{k})", 10 <= c.total <= 16, str(c.total))
            chans = [j.channel for j in c.jammers]
            check(f"频道唯一且∈[1,20] (p{prob},{k})",
                  len(set(chans)) == len(chans) and all(1 <= ch <= 20 for ch in chans), str(chans))
            check(f"R_eff∈[1000,1500] (p{prob},{k})",
                  all(1000 <= j.r_eff <= 1500 for j in c.jammers), "")
            check(f"位置在半径1800内 (p{prob},{k})",
                  all(math.hypot(j.x, j.y) <= 1800 for j in c.jammers), "")
            if prob == 3:
                check(f"problem3 全为omni ({k})", c.n_dir == 0, str(c.n_dir))


# ---------------- B. HTTP 协议级 ----------------
def with_server(fn):
    """起一个真实服务器，跑 fn(client, session)。用随机端口避免占用。"""
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    case = build_timing_case()
    session = SimSession(case, robot_id="TEAM1")
    server = SimServer(session, host="127.0.0.1", port=port)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.15)
    try:
        client = SimClient(f"http://127.0.0.1:{port}", robot_id="TEAM1")
        fn(client, session, port)
    finally:
        server.shutdown()
        server.server_close()


def test_http_happy_path(client: SimClient, session, port):
    st, b = client.enter()
    check("HTTP /enter 200+accepted", st == 200 and b and b["accepted"] is True, str(b))
    check("/enter 返回 remaining_real_duration_s",
          b is not None and "remaining_real_duration_s" in b, str(b))
    st, b = client.measure(300, 400, 1)
    check("HTTP /measure 200+accepted+virtual_time",
          st == 200 and b["accepted"] and approx(b["virtual_time_s"], 105.0), str(b))


def test_http_accepted_false(client: SimClient, session, port):
    client.enter()
    # 未知字段 → 200 accepted=false（用底层 _post 手工构造）
    p = {"arena_id": "default", "robot_id": "TEAM1", "request_id": "x-unknown",
         "position": {"x": 0, "y": 0}, "channel": 1, "typo_field": 1}
    st, b = client._post("/measure", p)
    check("未知字段 → 200 accepted=false", st == 200 and b["accepted"] is False, str((st, b)))
    check("accepted=false 仅含3字段且 virtual_time_s=0",
          b is not None and set(b.keys()) == {"accepted", "real_timestamp_ms", "virtual_time_s"}
          and b["virtual_time_s"] == 0, str(b))
    # robot_id 不符
    p2 = {"arena_id": "default", "robot_id": "WRONG", "request_id": "x-robot",
          "position": {"x": 0, "y": 0}, "channel": 1}
    st, b = client._post("/measure", p2)
    check("robot_id 不符 → accepted=false", st == 200 and b["accepted"] is False, str((st, b)))
    # 重复 enter
    st, b = client.enter()
    check("重复 /enter → accepted=false", st == 200 and b["accepted"] is False, str((st, b)))


def test_http_400s(client: SimClient, session, port):
    client.enter()
    # 缺字段 channel
    st, b = client._post("/measure", {"arena_id": "default", "robot_id": "TEAM1",
                                      "request_id": "m400a", "position": {"x": 0, "y": 0}})
    check("缺 channel → 400", st == 400, str((st, b)))
    # channel 非整数 1.5
    st, b = client._post("/measure", {"arena_id": "default", "robot_id": "TEAM1",
                                      "request_id": "m400b", "position": {"x": 0, "y": 0},
                                      "channel": 1.5})
    check("channel=1.5 → 400", st == 400, str((st, b)))
    # channel=1.0（整数值）→ 接受
    st, b = client._post("/measure", {"arena_id": "default", "robot_id": "TEAM1",
                                      "request_id": "m200c", "position": {"x": 0, "y": 0},
                                      "channel": 1.0})
    check("channel=1.0 → 200 accepted", st == 200 and b["accepted"] is True, str((st, b)))
    # 坐标越界
    st, b = client._post("/measure", {"arena_id": "default", "robot_id": "TEAM1",
                                      "request_id": "m400d",
                                      "position": {"x": 3_000_000, "y": 0}, "channel": 1})
    check("坐标越界 → 400", st == 400, str((st, b)))


def test_http_404_405_415(client: SimClient, session, port):
    import urllib.request
    base = client.base_url
    # 404 未知路径
    st, b = client._post("/unknown", {"arena_id": "default", "robot_id": "TEAM1", "request_id": "z"})
    check("未知路径 → 404", st == 404, str((st, b)))
    # 405 已知路径非POST
    try:
        req = urllib.request.Request(base + "/enter", method="GET")
        try:
            r = urllib.request.urlopen(req, timeout=5)
            st = r.status
        except urllib.error.HTTPError as e:
            st = e.code
    except Exception as e:
        st = -1
    check("/enter 用 GET → 405", st == 405, str(st))
    # 415 错误 Content-Type
    import json as _json
    data = _json.dumps({"arena_id": "default", "robot_id": "TEAM1", "request_id": "ct"}).encode()
    try:
        req = urllib.request.Request(base + "/enter", data=data,
                                     headers={"Content-Type": "text/plain"}, method="POST")
        try:
            r = urllib.request.urlopen(req, timeout=5)
            st = r.status
        except urllib.error.HTTPError as e:
            st = e.code
    except Exception:
        st = -1
    check("Content-Type text/plain → 415", st == 415, str(st))


def test_http_idempotency(client: SimClient, session, port):
    client.enter()
    # 同 request_id 同内容 → 返回首次响应、不重复推进虚拟时间
    p = {"arena_id": "default", "robot_id": "TEAM1", "request_id": "idem-1",
         "position": {"x": 300, "y": 400}, "channel": 1}
    st1, b1 = client._post("/measure", p)
    vt1 = b1["virtual_time_s"]
    st2, b2 = client._post("/measure", p)     # 完全相同
    check("幂等：同id同内容返回相同结果", st2 == 200 and approx(b2["virtual_time_s"], vt1),
          str((b1, b2)))
    # 同 id 改内容 → 409
    p2 = dict(p)
    p2["channel"] = 2
    st3, b3 = client._post("/measure", p2)
    check("同id改内容 → 409", st3 == 409, str((st3, b3)))


def test_http_close_after_exit(client: SimClient, session, port):
    client.enter()
    st, b = client.exit()
    check("/exit → 200 accepted user_exit",
          st == 200 and b["accepted"] and b.get("exit_reason") == "user_exit", str((st, b)))
    # 结束后再来动作 → 连接直接关闭（客户端得到 (0, None)）
    st, b = client.measure(0, 0, 1)
    check("结束后再 measure → 连接关闭(无JSON体)", st == 0 and b is None, str((st, b)))


def test_content_type_helper():
    check("CT: application/json ok", _content_type_ok("application/json"))
    check("CT: +charset=utf-8 ok", _content_type_ok("application/json; charset=utf-8"))
    check("CT: 其他参数 拒绝", not _content_type_ok("application/json; boundary=x"))
    check("CT: text/plain 拒绝", not _content_type_ok("text/plain"))


# ---------------- C. 性质 / 不变量测试（随机大量状态）----------------
def _random_case_for_props(seed):
    return generate_case(seed=seed, problem=4, field_kind="smooth", mode="formal")


def test_property_invariants():
    """随机上万个 (状态, 动作) 验证题面不变量恒成立。"""
    rng = random.Random(20260911)
    n = 4000
    bad_near, bad_clear, bad_after, bad_timing, bad_near_cov = 0, 0, 0, 0, 0
    for _ in range(n):
        case = _random_case_for_props(rng.randint(0, 1 << 30))
        eng = Engine(case)
        eng.enter()
        j = rng.choice(case.jammers)
        # 在源附近随机取点
        x = j.x + rng.uniform(-30, 30)
        y = j.y + rng.uniform(-30, 30)
        dist = math.hypot(x - j.x, y - j.y)
        in_cov = eng._in_coverage(x, y, j)

        prev_x, prev_y = eng.st.pos_x, eng.st.pos_y
        prev_ch = eng.st.channel
        prev_us = eng.st.virtual_time_us
        resp, out = eng.measure(x, y, j.channel)

        # 计时不变量：T_{k+1}-T_k = 移动/5 + 5 + 1_{切换}
        move_s = math.hypot(x - prev_x, y - prev_y) / SPEED_MPS
        sw = CH_SWITCH_S if j.channel != prev_ch else 0.0
        expect_us = (int(round(move_s * 1e6)) + int(round(sw * 1e6))
                     + int(round(MEASURE_ACTION_S * 1e6)))
        if (eng.st.virtual_time_us - prev_us) != expect_us:
            bad_timing += 1

        # near ⇒ 覆盖内且无 svd；且 near 必须 d≤5 且 in_cov
        if out.result == "near":
            if out.svd_deg is not None or not in_cov or dist > NEAR_THRESHOLD_M:
                bad_near += 1
        # d≤5 且 in_cov ⇒ 必须 near（P4：near 需在覆盖半平面内）
        if in_cov and dist <= NEAR_THRESHOLD_M and out.result != "near":
            bad_near_cov += 1

        # clear：d≤20 未清 ⇒ success（与朝向无关）
        cx = j.x + rng.uniform(-25, 25)
        cy = j.y + rng.uniform(-25, 25)
        cdist = math.hypot(cx - j.x, cy - j.y)
        was_cleared = j.cleared
        _, cout = eng.clear(cx, cy, j.channel)
        if (not was_cleared) and cdist <= CLEAR_RADIUS_M and cout.result != "success":
            bad_clear += 1

        # 清除后 measure 同频道 ⇒ no_signal
        if j.cleared:
            _, out2 = eng.measure(x, y, j.channel)
            if out2.result != "no_signal":
                bad_after += 1

    check(f"计时不变量 T差=移动/5+5+切换 ({n}例)", bad_timing == 0, f"{bad_timing} bad")
    check(f"near⇒覆盖内且无svd且d≤5 ({n}例)", bad_near == 0, f"{bad_near} bad")
    check(f"d≤5且覆盖内⇒near ({n}例)", bad_near_cov == 0, f"{bad_near_cov} bad")
    check(f"d≤20未清⇒clear成功(与朝向无关) ({n}例)", bad_clear == 0, f"{bad_clear} bad")
    check(f"清除后同频道measure⇒no_signal ({n}例)", bad_after == 0, f"{bad_after} bad")


def test_rejected_and_idem_no_state_change():
    """时限外/幂等重试不重复推进 position/channel/time/clear。"""
    # 幂等：引擎本身无幂等（在 server 层），这里验证 server 幂等不重复移动/clear
    from .server import SimSession
    case = build_timing_case()
    sess = SimSession(case, robot_id="T")
    sess.handle("/enter", {"arena_id": "default", "robot_id": "T", "request_id": "e"})
    p = {"arena_id": "default", "robot_id": "T", "request_id": "m1",
         "position": {"x": 300, "y": 400}, "channel": 1}
    sess.handle("/measure", p)
    vt1 = sess.engine.st.virtual_time_s
    pos1 = (sess.engine.st.pos_x, sess.engine.st.pos_y)
    sess.handle("/measure", p)   # 同 id 同内容
    vt2 = sess.engine.st.virtual_time_s
    pos2 = (sess.engine.st.pos_x, sess.engine.st.pos_y)
    check("幂等重试不推进虚拟时间", approx(vt1, vt2), f"{vt1} vs {vt2}")
    check("幂等重试不改变位置", pos1 == pos2, f"{pos1} vs {pos2}")

    # 幂等 clear 不重复清除（只清一次）
    j = Jammer(9, 100, 0, 1200, "omni")
    sess2 = SimSession(Case([j], ErrorField(1), 1), robot_id="T")
    sess2.handle("/enter", {"arena_id": "default", "robot_id": "T", "request_id": "e"})
    cp = {"arena_id": "default", "robot_id": "T", "request_id": "c1",
          "position": {"x": 100, "y": 0}, "channel": 9}
    _, r1 = sess2.handle("/clear", cp)
    _, r2 = sess2.handle("/clear", cp)   # 同 id 同内容 → 缓存
    check("幂等 clear 返回相同结果", r1 == r2, str((r1, r2)))
    check("幂等 clear 只清一次", sess2.case.cleared_count == 1, str(sess2.case.cleared_count))


def test_enter_initial_state():
    """/enter 后位置=(0,0)、当前频道=1（题面：初始频道 1）。"""
    eng = Engine(build_timing_case())
    eng.enter()
    check("/enter 后位置=(0,0)", eng.st.pos_x == 0.0 and eng.st.pos_y == 0.0,
          f"{(eng.st.pos_x, eng.st.pos_y)}")
    check("/enter 后当前频道=1", eng.st.channel == 1, str(eng.st.channel))


# ---------------- D. 边界测试（浮点最易出 bug 处）----------------
def test_boundary_distances():
    j = Jammer(3, 0.0, 0.0, 1200.0, "omni")
    eng = Engine(Case([j], ErrorField(1), 1))
    eng.enter()
    eps = 1e-6
    # d=5 精确 → near（≤5 含边界）
    _, o = eng.measure(5.0, 0.0, 3)
    check("d=5 → near(含边界)", o.result == "near", o.result)
    # d=5+eps → direction（>5）
    eng2 = Engine(Case([Jammer(3, 0.0, 0.0, 1200.0, "omni")], ErrorField(1), 1)); eng2.enter()
    _, o = eng2.measure(5.0 + eps, 0.0, 3)
    check("d=5+ε → direction", o.result == "direction", o.result)
    # d=5-eps → near
    eng3 = Engine(Case([Jammer(3, 0.0, 0.0, 1200.0, "omni")], ErrorField(1), 1)); eng3.enter()
    _, o = eng3.measure(5.0 - eps, 0.0, 3)
    check("d=5-ε → near", o.result == "near", o.result)


def test_boundary_clear_radius():
    def fresh():
        e = Engine(Case([Jammer(3, 0.0, 0.0, 1200.0, "omni")], ErrorField(1), 1)); e.enter(); return e
    eps = 1e-6
    e = fresh(); _, o = e.clear(20.0, 0.0, 3)
    check("clear d=20 → success(含边界)", o.result == "success", o.result)
    e = fresh(); _, o = e.clear(20.0 + eps, 0.0, 3)
    check("clear d=20+ε → no_target", o.result == "no_target_in_range", o.result)
    e = fresh(); _, o = e.clear(20.0 - eps, 0.0, 3)
    check("clear d=20-ε → success", o.result == "success", o.result)


def test_boundary_directional_90():
    """定向覆盖边界 Δφ=90° 含边界；90+ε 盲区。"""
    j = Jammer(4, 0.0, 0.0, 1200.0, "dir", direction_deg=0.0)  # 朝正东，覆盖[-90,90]
    def fresh():
        e = Engine(Case([Jammer(4, 0.0, 0.0, 1200.0, "dir", direction_deg=0.0)],
                        ErrorField(1), 1)); e.enter(); return e
    # 正北 (0,500)：源→点=90° → 含边界 direction
    e = fresh(); _, o = e.measure(0.0, 500.0, 4)
    check("定向 Δφ=90° → direction(含边界)", o.result == "direction", o.result)
    # 略过边界：角度 90°+微小 → 盲区。取点 (−ε, 500) 使源→点略大于 90°
    e = fresh(); _, o = e.measure(-0.5, 500.0, 4)
    check("定向 Δφ=90°+ε → no_signal(盲区)", o.result == "no_signal", o.result)


def test_svd_wraparound():
    """svd_deg 必须 ∈[0,360)：构造真实方位≈0 且误差为负，验证不会输出 360.00。"""
    # 用常偏 adversarial 场：ε=+1；把点选在真实方位≈359.5 处，+1 →360.5→round→…→必须回卷
    for tb_target in (359.996, 359.999, 0.001, 359.5):
        # 造一个源使 检测点→源 的真实方位 = tb_target
        # 检测点在原点，源方位角 = tb_target
        ang = math.radians(tb_target)
        j = Jammer(5, 500 * math.cos(ang), 500 * math.sin(ang), 1200.0, "omni")
        fld = make_error_field("adversarial", seed=1, params={"mode": "constant", "magnitude": 1.0})
        eng = Engine(Case([j], fld, 1)); eng.enter()
        _, o = eng.measure(0.0, 0.0, 5)
        if o.result == "direction":
            check(f"svd∈[0,360) (真方位≈{tb_target})",
                  0.0 <= o.svd_deg < 360.0, f"svd={o.svd_deg}")


# ---------------- E. 误差场家族 ----------------
def test_error_fields_bounded_and_fixed():
    rng = random.Random(7)
    pts = [(rng.uniform(-1800, 1800), rng.uniform(-1800, 1800)) for _ in range(500)]
    for kind in FIELD_KINDS:
        fld = make_error_field(kind, seed=42)
        mx = 0.0
        ok_fixed = True
        for (x, y) in pts:
            e1 = fld.error_deg(x, y)
            e2 = fld.error_deg(x, y)   # 同点重复
            if e1 != e2:
                ok_fixed = False
            mx = max(mx, abs(e1))
        check(f"[{kind}] 误差∈[-1,1]", mx <= 1.0 + 1e-12, f"max|e|={mx}")
        check(f"[{kind}] 同点重复不变", ok_fixed, "")


def test_error_field_smooth_vs_iid_microstep():
    """smooth 场微动几乎不变；iid 场微动可显著变化 → 二者行为可区分。"""
    x, y = 123.4, -567.8
    d = 0.01  # 1cm 微动
    smooth = make_error_field("smooth", seed=3, params={"length_scale": 300.0})
    iid = make_error_field("iid", seed=3)
    ds = abs(smooth.error_deg(x, y) - smooth.error_deg(x + d, y))
    check("smooth 场 1cm 微动误差变化极小", ds < 1e-3, f"Δ={ds}")
    # iid：统计多点，至少某些点微动后明显不同
    rng = random.Random(11)
    max_di = 0.0
    for _ in range(200):
        px, py = rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)
        max_di = max(max_di, abs(iid.error_deg(px, py) - iid.error_deg(px + d, py)))
    check("iid 场微动可产生显著误差变化", max_di > 0.1, f"max Δ={max_di}")


def test_biased_field_nonzero_mean():
    fld = make_error_field("biased", seed=5, params={"bias": 0.6, "noise_amp": 0.3})
    rng = random.Random(9)
    vals = [fld.error_deg(rng.uniform(-1500, 1500), rng.uniform(-1500, 1500)) for _ in range(2000)]
    mean = sum(vals) / len(vals)
    check("biased 场均值显著非零", mean > 0.2, f"mean={mean:.3f}")


def test_field_config_roundtrip():
    for kind in FIELD_KINDS:
        fld = make_error_field(kind, seed=17)
        cfg = fld.config()
        fld2 = make_error_field(**cfg)
        # 同点取值应一致
        same = approx(fld.error_deg(100.0, 200.0), fld2.error_deg(100.0, 200.0), 1e-12)
        check(f"[{kind}] config 往返一致", same, "")


# ---------------- F. 压力案例 + 蒙特卡洛试验场 ----------------
def _greedy_omni_policy(runner):
    """
    简易参考策略（仅用于验证 harness / 压力生成器可跑通，不追求成绩）：
    对每个频道在原点测一次；若拿到 svd，就沿该方位向外推进若干距离再测一次，
    两次示向线交点作为估计位置，前往清除。仅处理 omni 的 P3 场景足够跑通试验场。
    """
    runner.enter()
    for ch in range(1, 21):
        r = runner.measure(0.0, 0.0, ch)
        if r is None:
            break
        if r.get("measure_result") != "direction":
            continue
        b1 = math.radians(r["svd_deg"])
        # 第二点：垂直于视线方向偏移，拿第二条线
        ox, oy = 400.0 * math.cos(b1 + math.pi / 2), 400.0 * math.sin(b1 + math.pi / 2)
        r2 = runner.measure(ox, oy, ch)
        if r2 is None:
            break
        if r2.get("measure_result") == "near":
            runner.clear(ox, oy, ch)
            continue
        if r2.get("measure_result") != "direction":
            continue
        b2 = math.radians(r2["svd_deg"])
        # 解两条射线交点：p1 + t1*d1 = p2 + t2*d2
        d1 = (math.cos(b1), math.sin(b1))
        d2 = (math.cos(b2), math.sin(b2))
        den = d1[0] * (-d2[1]) - d1[1] * (-d2[0])
        if abs(den) < 1e-9:
            continue
        rx, ry = ox - 0.0, oy - 0.0
        t1 = (rx * (-d2[1]) - ry * (-d2[0])) / den
        ex, ey = d1[0] * t1, d1[1] * t1
        runner.clear(ex, ey, ch)
    runner.exit()


def test_harness_runs_and_metrics():
    """试验场能跑通、指标字段齐全、成功率∈[0,1]。"""
    m = evaluate(_greedy_omni_policy, n_cases=15, problem=3, base_seed=100,
                 field_kind="smooth")
    check("harness 跑通 15 局", m.n_episodes == 15, str(m.n_episodes))
    check("success_rate∈[0,1]", 0.0 <= m.success_rate <= 1.0, str(m.success_rate))
    check("时间分位单调 P50≤P90≤P95≤max",
          m.time_p50 <= m.time_p90 + 1e-9 <= m.time_p95 + 1e-9 <= m.time_max + 1e-9,
          f"{m.time_p50}/{m.time_p90}/{m.time_p95}/{m.time_max}")
    check("measure_mean>0", m.measure_mean > 0, str(m.measure_mean))


def test_stress_generators_valid():
    """每种压力案例都满足题面硬约束。"""
    scan = [(0, 0), (600, 0), (-600, 0), (0, 600), (0, -600)]
    for stype in STRESS_TYPES:
        for k in range(3):
            prob = 4 if stype.startswith("dir_") else 3
            c = generate_stress_case(stype, seed=500 + k, problem=prob,
                                     scan_points=scan)
            ok_n = 10 <= c.total <= 16
            ok_ch = len(set(j.channel for j in c.jammers)) == c.total
            ok_pos = all(math.hypot(j.x, j.y) <= ARENA_RADIUS_M + 1e-6 for j in c.jammers)
            ok_reff = all(1000.0 <= j.r_eff <= 1500.0 for j in c.jammers)
            ok_mix = True
            if prob == 4:
                ok_mix = c.n_omni >= 1 and c.n_dir >= 1
            check(f"[stress:{stype}#{k}] 约束(数量/频道/位置/Reff/混合)",
                  ok_n and ok_ch and ok_pos and ok_reff and ok_mix,
                  f"n={c.total} ch_uniq={ok_ch} pos={ok_pos} reff={ok_reff} "
                  f"omni={c.n_omni} dir={c.n_dir}")


def test_p4_generation_forces_mix():
    """P4 随机生成始终两类各≥1（含显式 n_directional=0 被夹回）。"""
    bad = 0
    for k in range(40):
        c = generate_case(seed=800 + k, problem=4, mode="formal")
        if not (c.n_omni >= 1 and c.n_dir >= 1):
            bad += 1
    check("P4 随机生成恒含全向+定向各≥1", bad == 0, f"{bad} bad")
    c0 = generate_case(seed=1, problem=4, n_jammers=12, n_directional=0)
    check("P4 显式 n_dir=0 被夹回(dir≥1)", c0.n_dir >= 1, str(c0.n_dir))


def test_reff_allows_duplicates():
    """R_eff 不强制互异：允许重复（用 case 文件往返验证任意值可载入）。"""
    js = [Jammer(1, 100, 0, 1200.0, "omni"), Jammer(2, -100, 0, 1200.0, "omni")]
    c = Case(js, ErrorField(1), 1)
    check("两源 R_eff 可相同", c.jammers[0].r_eff == c.jammers[1].r_eff, "")


def main():
    print("=" * 60)
    print("A. 引擎级单元测试")
    print("=" * 60)
    test_timing_example()
    test_clear_timing_hit()
    test_near_and_nosignal()
    test_directional_coverage()
    test_svd_error_bounds_and_repeatability()
    test_channel_state_update()
    test_case_generation_constraints()
    test_content_type_helper()

    print("=" * 60)
    print("C. 性质 / 不变量测试")
    print("=" * 60)
    test_property_invariants()
    test_rejected_and_idem_no_state_change()
    test_enter_initial_state()

    print("=" * 60)
    print("D. 边界测试")
    print("=" * 60)
    test_boundary_distances()
    test_boundary_clear_radius()
    test_boundary_directional_90()
    test_svd_wraparound()

    print("=" * 60)
    print("E. 误差场家族")
    print("=" * 60)
    test_error_fields_bounded_and_fixed()
    test_error_field_smooth_vs_iid_microstep()
    test_biased_field_nonzero_mean()
    test_field_config_roundtrip()

    print("=" * 60)
    print("F. 压力案例 + 蒙特卡洛试验场")
    print("=" * 60)
    test_harness_runs_and_metrics()
    test_stress_generators_valid()
    test_p4_generation_forces_mix()
    test_reff_allows_duplicates()

    print("=" * 60)
    print("B. HTTP 协议级测试")
    print("=" * 60)
    with_server(test_http_happy_path)
    with_server(test_http_accepted_false)
    with_server(test_http_400s)
    with_server(test_http_404_405_415)
    with_server(test_http_idempotency)
    with_server(test_http_close_after_exit)

    print("=" * 60)
    total = len(_results)
    passed = sum(_results)
    print(f"结果：{passed}/{total} 通过")
    print("=" * 60)
    return 0 if passed == total else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
