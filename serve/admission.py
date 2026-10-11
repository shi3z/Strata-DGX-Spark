"""Atomic HTTP admission and owner-scoped draining, including preprocessing."""
import threading
import uuid


class AdmissionGate:
    def __init__(self):
        self.lock = threading.Lock()
        self.token = None
        self.active = 0

    def enter(self):
        with self.lock:
            if self.token is not None:
                return False
            self.active += 1
            return True

    def leave(self):
        with self.lock:
            self.active -= 1

    def snapshot(self):
        with self.lock:
            return {'draining': self.token is not None, 'active_requests': self.active}

    def drain(self):
        with self.lock:
            if self.token is not None:
                raise ValueError('another drain is already active')
            self.token = uuid.uuid4().hex
            return {'token': self.token, 'draining': True, 'active_requests': self.active}

    def resume(self, token):
        with self.lock:
            if not self.token or token != self.token:
                raise ValueError('drain owner token does not match')
            self.token = None
            return {'draining': False, 'active_requests': self.active}
