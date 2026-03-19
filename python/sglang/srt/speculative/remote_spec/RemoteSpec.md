# RemoteSpec

D/T disaggregation and Pearl implementation.


## Args
To enable EAGLE speculative decoding the following parameters are relevant:
| Parameter | Description | Default |
|--------------------|--------------------|--------------------|
|--speculative-algorithm | Speculative algorithm. | `REMOTE` |
|--remote-speculative-role | The role of the server. | `None` |
|--remote-speculative-max-batch-size | The threshold at which the server is considered overloaded. When the running bsz exceeds this value, it is deemed overloaded. The Draft server stops processing draft requests, and the target stops sending draft requests. | 32 |
|--remote-speculative-reject-interval | The interval to resend draft requests to draft server when recvieve reject singal from draft server. | 5000 |
|--remote-speculative-no-draft-ratio | The ratio of requests with no draft tokens to the total batch size. If the ratio is larger than this value, the server will only decode one token. | 0.5 |
|--remote-speculative-retry-fail-ratio | Minimum ratio of pre-verify requests to batch size required to trigger a retry. | 0.5 |
|--remote-speculative-retry-min-count | Minimum number of failed requests required to trigger a retry. | 4 |
|--remote-speculative-zmq-addr | ZMQ address for remote speculative decoding | `None` |
|--remote-speculative-zmq-port | ZMQ port | `None` |
| REMOTE_SPEC_RECV_TIMEOUT_MS | The maximum time (ms) the target waits for the draft response. | 200 |
| REMOTE_SPEC_FAILURE_THRESHOLD | The threshold of consecutive rounds without receiving draft responses that triggers the target to stop sending draft requests. | 30 |
| REMOTE_SPEC_COOLDOWN_ROUNDS | The number of rounds for the target to resume sending draft requests after a cold start. | 100 |
| REMOTE_SPEC_DEBUG | Enable ZMQ-related logging. | 0 |



