from sglang.srt.server_args import ServerArgs
from typing import Optional
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.speculative.eagle_info import EagleVerifyInput


class RemoteSpecWorker:
    '''
    RemoteSpecWorker is used in the target end of remote spec.
    It is responsible for verifying.
    '''
    def __init__(self,
                 server_args: ServerArgs,
                 gpu_id: int,
                 tp_rank: int,
                 dp_rank: Optional[int],
                 moe_ep_rank: int,
                 nccl_port: int,
                 target_worker: TpModelWorker,
                 ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank


    def forward_batch_generation(self, batch: ScheduleBatch, **kwargs) -> GenerationBatchResult:
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            logits_output, next_token_ids, seq_lens_cpu = self.forward_target_extend(
                batch
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=next_token_ids,
                num_accepted_tokens=0,
                can_run_cuda_graph=False,
            )
        else:
            spec_info = self.construct_spec_info(batch)
            logits_output, verify_output, model_worker_batch, can_run_cuda_graph = (
                self.verify(batch, spec_info)
            )
            return GenerationBatchResult(
                logits_output=logits_output,
                next_token_ids=verify_output.verified_id,
                num_accepted_tokens=sum(verify_output.accept_length_per_req_cpu),
                accept_length_per_req_cpu=verify_output.accept_length_per_req_cpu,
                can_run_cuda_graph=can_run_cuda_graph,
            )
        
    def construct_spec_info(self, batch: ScheduleBatch) -> EagleVerifyInput:
        pass