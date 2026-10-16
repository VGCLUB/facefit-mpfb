import json, socket

class BlenderClient:
    def __init__(self, host="127.0.0.1", port=9876, timeout=600):
        self.addr, self.timeout, self.sock, self.f = (host, port), timeout, None, None

    def _conn(self):
        if self.sock is None:
            self.sock = socket.create_connection(self.addr, timeout=self.timeout)
            self.f = self.sock.makefile("rwb")
        return self.f

    def call(self, cmd, **args):
        f = self._conn()
        f.write((json.dumps({"cmd": cmd, "args": args}) + "\n").encode()); f.flush()
        line = f.readline()
        if not line:
            self.sock = None; raise ConnectionError("Blender closed the connection")
        r = json.loads(line)
        if not r["ok"]:
            raise RuntimeError(r["error"] + "\n" + r.get("trace", ""))
        return r["result"]

    def shutdown(self):
        f = self._conn(); f.write(b"__shutdown__\n"); f.flush(); self.sock.close(); self.sock = None
