"""
HTTP+JSON 服务层（server.py）：把附件2 §5 的通信总规则逐条落实为一个本地服务器。

关键落实点：
- §5.1 请求格式：方法必须 POST；路径精确 /enter /measure /clear /exit（不接受尾斜杠/查询）；
  Content-Type 必须 application/json（仅允许附加 charset=utf-8）否则 415；
  Content-Encoding 省略或 identity，否则 415；请求体无 BOM 的 UTF-8 JSON 对象、不得重复键、
  嵌套 ≤16 层、≤65536 字节（否则 413）；未声明字段 → 200 + accepted=false；
  channel 必须 1..20 整数（1.0 接受、1.5 → 400）；坐标有限且 |·|≤2e6，否则 400；
  arena_id 必须 "default"；robot_id 必须等于登录队号；request_id 幂等键。
- §5.2 响应格式：所有业务响应含 accepted / real_timestamp_ms / virtual_time_s；
  accepted=false 时仅这三个字段。
- §5.3 HTTP 状态与 accepted 的对应表（200/400/404/405/409/413/415/429/500）。
- §4.5/§1.5 接口未开放（未 enter 前的连接是开放的，但“测试结束/未开放”场景）→
  本模拟器用“连接直接关闭（无 JSON 体）”来复现 finished 之后再来请求的情形。
- 幂等：同 request_id 同内容 → 返回首次完整响应，不重复推进；同 id 改内容 → 409；
  结构错误/未知字段/arena_id/robot_id 不匹配 → 不占用该 id（附件2 §5.3 末、§12）。

设计：用标准库 http.server，单线程串行处理（附件2：不得并发不同动作）。
"""

from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn
from typing import Optional

from .engine import Engine, COORD_ABS_MAX
from .case import Case, CHANNEL_MIN, CHANNEL_MAX


VALID_PATHS = {"/enter", "/measure", "/clear", "/exit"}
MAX_BODY_BYTES = 65536
MAX_JSON_DEPTH = 16
CONTROL_CHARS_RE = re.compile(
    "["
    "\x00-\x1f\x7f\x80-\x9f"          # C0 / DEL / C1
    "\u200b-\u200f\u202a-\u202e"      # 零宽字符 / 双向控制符
    "\u2060\u2028\u2029\ufeff"        # word-joiner / 行分隔 / BOM
    "]"
)


class BusinessError(Exception):
    """携带 HTTP 状态码的错误（用于 400/413/415/409 等）。"""

    def __init__(self, http_status: int, detail: str = ""):
        super().__init__(detail)
        self.http_status = http_status
        self.detail = detail


class CloseConnection(Exception):
    """需要“直接关闭连接、无 JSON 体”的情形（接口未开放 / 测试已结束）。"""


def _json_depth(obj, depth: int = 1) -> int:
    """计算 JSON 对象的最大嵌套深度。"""
    if isinstance(obj, dict):
        return max([depth] + [_json_depth(v, depth + 1) for v in obj.values()])
    if isinstance(obj, list):
        return max([depth] + [_json_depth(v, depth + 1) for v in obj])
    return depth


def _reject_duplicate_keys(pairs):
    """object_pairs_hook：检测 JSON 对象内重复键（附件2 §5.1 不能有重复键 → 400）。"""
    seen = {}
    for k, v in pairs:
        if k in seen:
            raise BusinessError(400, f"duplicate key: {k}")
        seen[k] = v
    return seen


def _is_integer_number(v) -> bool:
    """channel 允许 JSON number 且数值恰为整数（1.0 接受，1.5 → 400）。bool 不算。"""
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    if isinstance(v, float):
        return v.is_integer()
    return False


def _is_finite_number(v) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        f = float(v)
        return f == f and f not in (float("inf"), float("-inf"))
    return False


class SimSession:
    """
    单局会话：绑定一个 Case + Engine，维护登录队号、幂等记录、接口开放状态。
    server 层每个请求都通过这里做“业务级”处理（HTTP 校验在 Handler 里先做）。
    """

    def __init__(self, case: Case, robot_id: str, real_clock=None):
        self.case = case
        self.robot_id = robot_id
        self.engine = Engine(case) if real_clock is None else Engine(case, real_clock)
        self._idem: dict[str, dict] = {}     # request_id -> (path, canonical_body, response, http_status)
        self.lock = threading.Lock()
        # 接口开放：真实模拟器在倒计时/结束后关闭；离线版默认从创建即开放，
        # engine.finished 之后视为“已结束”，再来的动作 → CloseConnection。
        self.interface_open = True

    # ---------- 业务分发 ----------
    def handle(self, path: str, body: dict) -> tuple[int, dict]:
        """
        返回 (http_status, response_dict)。可能抛 BusinessError / CloseConnection。
        进入此函数前，Handler 已完成：方法、路径、Content-Type/Encoding、体大小、
        JSON 语法/重复键/深度 的校验。
        """
        # 幂等键必须存在且合法（此处仅做业务层幂等；格式校验在 _validate_identifiers）
        self._validate_common_fields(path, body)
        request_id = body["request_id"]

        # ---- 幂等处理（附件2 §5.3）----
        canonical = _canonical(path, body)
        if request_id in self._idem:
            rec = self._idem[request_id]
            if rec["canonical"] == canonical:
                # 同 id 同内容 → 返回首次完整响应，不重复执行
                return rec["http_status"], rec["response"]
            else:
                # 同 id 改内容 → 409
                raise BusinessError(409, "request_id reused with different content")

        # ---- arena_id / robot_id / 未知字段 → 200 + accepted=false（不占用 id）----
        acc_false = self._check_accept_false(path, body)
        if acc_false is not None:
            return 200, self._wrap_accepted_false()

        # ---- 若测试已结束/接口未开放：连接直接关闭（无 JSON 体）----
        if self.engine.st.finished or not self.interface_open:
            # /enter 之外的动作在结束后到达 → 关闭连接
            raise CloseConnection()

        # ---- 正常执行 ----
        http_status, response = self._dispatch(path, body)

        # 业务上被接受的动作占用该 request_id（附件2 §5.3）
        if response.get("accepted") is True:
            self._idem[request_id] = {
                "canonical": canonical,
                "response": response,
                "http_status": http_status,
            }
        return http_status, response

    def _dispatch(self, path: str, body: dict) -> tuple[int, dict]:
        eng = self.engine
        if path == "/enter":
            if eng.st.entered and not eng.st.finished:
                # 重复调用 /enter → 200 + accepted=false（附件2 §6.3）
                return 200, self._wrap_accepted_false()
            extra = eng.enter()
            return 200, self._wrap_accepted_true(extra)

        # /measure /clear /exit 需已 enter
        if not eng.st.entered:
            return 200, self._wrap_accepted_false()

        if path == "/measure":
            self._validate_position_channel(body)
            resp, _ = eng.measure(body["position"]["x"], body["position"]["y"], int(body["channel"]))
            if resp.get("__finished__"):
                raise CloseConnection()
            return 200, self._wrap_accepted_true(resp)

        if path == "/clear":
            self._validate_position_channel(body)
            resp, _ = eng.clear(body["position"]["x"], body["position"]["y"], int(body["channel"]))
            if resp.get("__finished__"):
                raise CloseConnection()
            return 200, self._wrap_accepted_true(resp)

        if path == "/exit":
            resp = eng.exit()
            return 200, self._wrap_accepted_true(resp)

        raise BusinessError(404, "unknown path")

    # ---------- 字段校验 ----------
    def _validate_common_fields(self, path: str, body: dict) -> None:
        # 必填：arena_id / robot_id / request_id（缺失 → 400）
        for key in ("arena_id", "robot_id", "request_id"):
            if key not in body:
                raise BusinessError(400, f"missing field: {key}")
        self._validate_identifiers(body)

    def _validate_identifiers(self, body: dict) -> None:
        arena = body["arena_id"]
        robot = body["robot_id"]
        req = body["request_id"]
        if not isinstance(arena, str):
            raise BusinessError(400, "arena_id must be string")
        if not isinstance(robot, str):
            raise BusinessError(400, "robot_id must be string")
        if not isinstance(req, str):
            raise BusinessError(400, "request_id must be string")
        # 长度与控制字符（附件2 §5.1）
        rb = robot.encode("utf-8")
        if not (1 <= len(rb) <= 64):
            raise BusinessError(400, "robot_id length out of range")
        qb = req.encode("utf-8")
        if not (1 <= len(qb) <= 128):
            raise BusinessError(400, "request_id length out of range")
        if CONTROL_CHARS_RE.search(robot) or CONTROL_CHARS_RE.search(req):
            raise BusinessError(400, "identifier contains control/format char")

    def _check_accept_false(self, path: str, body: dict) -> Optional[str]:
        """arena_id / robot_id 不匹配、未知字段 → 返回原因字符串（→ accepted=false）。"""
        if body["arena_id"] != "default":
            return "arena_id"
        if body["robot_id"] != self.robot_id:
            return "robot_id"
        # 未知字段（顶层 + position 下）
        allowed_top = {"/enter": {"arena_id", "robot_id", "request_id"},
                       "/exit": {"arena_id", "robot_id", "request_id"},
                       "/measure": {"arena_id", "robot_id", "request_id", "position", "channel"},
                       "/clear": {"arena_id", "robot_id", "request_id", "position", "channel"}}[path]
        for k in body.keys():
            if k not in allowed_top:
                return f"unknown field: {k}"
        if path in ("/measure", "/clear"):
            pos = body.get("position")
            if isinstance(pos, dict):
                for k in pos.keys():
                    if k not in {"x", "y"}:
                        return f"unknown field in position: {k}"
        return None

    def _validate_position_channel(self, body: dict) -> None:
        """/measure /clear 的 position 与 channel（→ 400）。"""
        if "position" not in body:
            raise BusinessError(400, "missing position")
        if "channel" not in body:
            raise BusinessError(400, "missing channel")
        pos = body["position"]
        if not isinstance(pos, dict) or "x" not in pos or "y" not in pos:
            raise BusinessError(400, "invalid position")
        x, y = pos["x"], pos["y"]
        if not _is_finite_number(x) or not _is_finite_number(y):
            raise BusinessError(400, "position not finite")
        if abs(float(x)) > COORD_ABS_MAX or abs(float(y)) > COORD_ABS_MAX:
            raise BusinessError(400, "position out of range")
        ch = body["channel"]
        if not _is_integer_number(ch):
            raise BusinessError(400, "channel not integer")
        chi = int(ch)
        if not (CHANNEL_MIN <= chi <= CHANNEL_MAX):
            raise BusinessError(400, "channel out of range")

    # ---------- 响应包装 ----------
    def _wrap_accepted_true(self, extra: dict) -> dict:
        resp = {
            "accepted": True,
            "real_timestamp_ms": self.engine.real_timestamp_ms(),
        }
        resp.update(extra)   # 含 virtual_time_s 等
        return resp

    def _wrap_accepted_false(self) -> dict:
        # accepted=false：只含三个字段，virtual_time_s=0（附件2 §4.1/§5.2）
        return {
            "accepted": False,
            "real_timestamp_ms": self.engine.real_timestamp_ms(),
            "virtual_time_s": 0,
        }


def _canonical(path: str, body: dict) -> str:
    """请求内容规范化（用于幂等比较）：路径 + 排序后的 JSON。"""
    return path + "|" + json.dumps(body, sort_keys=True, ensure_ascii=False)


class SimHTTPRequestHandler(BaseHTTPRequestHandler):
    """把一个 SimSession 暴露为 HTTP 接口。server.session 持有会话。"""

    protocol_version = "HTTP/1.1"

    # 静默默认日志，改由外部 logger 记录
    def log_message(self, fmt, *args):
        if getattr(self.server, "verbose", False):
            super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _close_no_body(self) -> None:
        """直接关闭连接、无 JSON 体（复现接口未开放/已结束）。"""
        try:
            self.close_connection = True
        except Exception:
            pass

    # 非 POST 的已知路径 → 405；未知路径 → 404
    def do_GET(self):
        self._method_not_post()

    def do_PUT(self):
        self._method_not_post()

    def do_DELETE(self):
        self._method_not_post()

    def _method_not_post(self):
        path = self.path
        if path in VALID_PATHS:
            self._error_json(405)
        else:
            self._error_json(404)

    def do_POST(self):
        session: SimSession = self.server.session
        path = self.path

        # 先读取（并因此清空）请求体，避免未读体导致 HTTP/1.1 keep-alive 连接错位。
        # 大小限制（§5.1）→ 413（超限则读取并丢弃，仍需清空缓冲以保持连接可用）。
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length > 0 else b""
        if length > MAX_BODY_BYTES or len(raw) > MAX_BODY_BYTES:
            self._error_json(413)
            return

        # 路径必须精确（不接受尾斜杠/查询参数）
        if path not in VALID_PATHS:
            self._error_json(404)
            return

        # Content-Type / Content-Encoding 校验（§5.1）→ 415
        if not self._check_headers():
            self._error_json(415)
            return

        # 解析 JSON（§5.1）→ 400（含 BOM、重复键、语法、深度）
        try:
            body = self._parse_json(raw)
        except BusinessError as be:
            self._error_json(be.http_status)
            return
        except Exception:
            self._error_json(400)
            return

        # 业务处理（串行）
        with session.lock:
            try:
                status, payload = session.handle(path, body)
            except CloseConnection:
                self._close_no_body()
                return
            except BusinessError as be:
                self._error_json(be.http_status)
                return
            except Exception:
                self._error_json(500)
                return
        self._send_json(status, payload)

    # ---- 头校验 ----
    def _check_headers(self) -> bool:
        ctype = self.headers.get("Content-Type", "")
        if not _content_type_ok(ctype):
            return False
        cenc = self.headers.get("Content-Encoding", None)
        if cenc is not None and cenc.strip().lower() not in ("", "identity"):
            return False
        return True

    def _parse_json(self, raw: bytes) -> dict:
        # 无 BOM（§5.1）
        if raw.startswith(b"\xef\xbb\xbf"):
            raise BusinessError(400, "BOM not allowed")
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise BusinessError(400, "not valid utf-8")
        try:
            obj = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except BusinessError:
            raise
        except Exception:
            raise BusinessError(400, "invalid json")
        if not isinstance(obj, dict):
            raise BusinessError(400, "body must be json object")
        if _json_depth(obj) > MAX_JSON_DEPTH:
            raise BusinessError(400, "json too deep")
        return obj

    def _error_json(self, status: int) -> None:
        """
        所有能形成 HTTP 响应的错误都返回含 accepted/real_timestamp_ms/virtual_time_s
        的 JSON（附件2 §5.3 末段）。
        """
        session: SimSession = self.server.session
        payload = {
            "accepted": False,
            "real_timestamp_ms": session.engine.real_timestamp_ms(),
            "virtual_time_s": 0,
        }
        self._send_json(status, payload)


def _content_type_ok(ctype: str) -> bool:
    """Content-Type 必须 application/json，仅允许附加 charset=utf-8（§5.1）。"""
    if not ctype:
        return False
    parts = [p.strip() for p in ctype.split(";")]
    if not parts or parts[0].lower() != "application/json":
        return False
    for p in parts[1:]:
        if not p:
            continue
        if "=" not in p:
            return False
        k, v = p.split("=", 1)
        if k.strip().lower() != "charset" or v.strip().lower() != "utf-8":
            return False
    return True


class SimServer(ThreadingHTTPServer):
    """承载一个 SimSession 的 HTTP 服务器（仅监听回环地址）。"""

    daemon_threads = True

    def __init__(self, session: SimSession, host: str = "127.0.0.1", port: int = 2026, verbose: bool = False):
        super().__init__((host, port), SimHTTPRequestHandler)
        self.session = session
        self.verbose = verbose


def serve(case: Case, robot_id: str, host="127.0.0.1", port=2026, verbose=False) -> SimServer:
    """便捷入口：创建会话并返回 SimServer（调用者负责 serve_forever/shutdown）。"""
    session = SimSession(case, robot_id)
    return SimServer(session, host=host, port=port, verbose=verbose)
