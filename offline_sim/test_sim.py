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

from .case import Case, Jammer, ErrorField, generate_case, norm_deg
from .engine import Engine
from .server import SimSession, SimServer, _content_type_ok
from .client import SimClient


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
