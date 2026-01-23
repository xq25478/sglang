from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.managers.schedule_batch import Req


class RemoteSpecKVRollbacker:
    '''
    RemoteSpecKVRollbacker is used in the draft end of remote spec.
    It is responsible for rolling back the KV cache for the draft requests.
    '''
    def __init__(self,
                 token_to_kv_pool_allocator: TokenToKVPoolAllocator,
                 req_to_token_pool: ReqToTokenPool,
                 tree_cache: BasePrefixCache,
                 page_size: int = 1,
        ) -> None:
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.req_to_token_pool = req_to_token_pool
        self.tree_cache = tree_cache
        self.page_size = page_size

    
    def rollback(self, req: Req, fork_point: int) -> None:
        pass


    def release_all_kv_for_finished_req(self, req: Req) -> None:
        pass


    def release_all_kv_for_reprefill_req(self, req: Req) -> None:
        pass



    