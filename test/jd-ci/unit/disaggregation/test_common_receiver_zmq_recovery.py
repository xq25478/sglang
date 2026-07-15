import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import zmq

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVReceiver
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestCommonKVReceiverZMQRecovery(CustomTestCase):
    def test_connect_enables_reconnect_and_bounds_send_wait(self):
        context = zmq.Context()
        socket_cache = {}
        socket_locks = {}
        try:
            with (
                patch.object(CommonKVReceiver, "_ctx", context),
                patch.object(CommonKVReceiver, "_socket_cache", socket_cache),
                patch.object(CommonKVReceiver, "_socket_locks", socket_locks),
            ):
                sock, _ = CommonKVReceiver._connect("tcp://127.0.0.1:1")

                self.assertEqual(sock.getsockopt(zmq.RECONNECT_IVL), 100)
                self.assertEqual(sock.getsockopt(zmq.RECONNECT_IVL_MAX), 1000)
                self.assertEqual(sock.getsockopt(zmq.SNDTIMEO), 5000)
                self.assertEqual(sock.getsockopt(zmq.LINGER), 0)
        finally:
            for sock in socket_cache.values():
                sock.close(linger=0)
            context.term()

    def test_send_timeout_discards_stale_socket_and_fails_request(self):
        receiver = CommonKVReceiver.__new__(CommonKVReceiver)
        receiver.bootstrap_room = 42
        receiver.conclude_state = None
        receiver.kv_mgr = SimpleNamespace(
            record_failure=MagicMock(),
            update_status=MagicMock(),
        )

        sock = MagicMock()
        sock.send_multipart.side_effect = zmq.Again()
        lock = threading.Lock()
        bootstrap_info = {"rank_ip": "127.0.0.1", "rank_port": 12345}

        with (
            patch.object(
                receiver,
                "_connect_to_bootstrap_server",
                return_value=(sock, lock),
            ),
            patch.object(receiver, "disconnect_endpoint") as disconnect_endpoint,
        ):
            sent = receiver._send_multipart_to_bootstrap_server(
                bootstrap_info, [b"metadata"]
            )

        self.assertFalse(sent)
        disconnect_endpoint.assert_called_once_with("tcp://127.0.0.1:12345")
        receiver.kv_mgr.record_failure.assert_called_once()
        receiver.kv_mgr.update_status.assert_called_once_with(42, KVPoll.Failed)
        self.assertEqual(receiver.conclude_state, KVPoll.Failed)

    def test_failed_registration_is_not_cached(self):
        receiver = CommonKVReceiver.__new__(CommonKVReceiver)
        receiver.bootstrap_addr = "127.0.0.1:30000"
        receiver.bootstrap_room = 42
        receiver.prefill_dp_rank = 0
        receiver.target_cp_ranks = [0]
        receiver.target_tp_rank = 0
        receiver.target_tp_ranks = [0]
        receiver.target_pp_ranks = [0]
        receiver.conclude_state = None
        receiver.kv_mgr = SimpleNamespace(
            connection_pool={},
            is_mla_backend=False,
            record_failure=MagicMock(),
            update_status=MagicMock(),
        )
        receiver._get_bootstrap_info_from_server = MagicMock(
            return_value={"rank_ip": "127.0.0.1", "rank_port": 12345}
        )

        def fail_registration():
            receiver.conclude_state = KVPoll.Failed

        receiver._register_kv_args = fail_registration

        receiver._setup_bootstrap_infos()

        self.assertEqual(receiver.kv_mgr.connection_pool, {})


if __name__ == "__main__":
    unittest.main()
