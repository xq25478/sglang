# SGLang HiCache 技术文档（L1 / L2 / L3 三级 KV 缓存）

> 版本基线：本仓库 `codex/dspark-layersplit` 分支。文中所有 `file:line` 引用均可跳转核对。官方英文文档见 `docs/advanced_features/hicache_design.md` 与 `hicache_best_practices.md`；本文补充源码级的数据结构、线程模型与控制流细节。

---

## 1. 系统概览

### 1.1 为什么需要 HiCache

LLM 推理的 prefill 阶段要把输入序列转成 KV cache；当多个请求共享前缀时，这段 KV cache 完全相同，缓存复用即可避免重复计算。SGLang 的 RadixAttention 已利用 GPU 显存做前缀缓存，但容量受限于单卡显存。HiCache 借鉴 CPU 三级缓存结构，把前缀 KV 扩展到两级更大的介质：

| 层级 | 介质 | 角色 | 共享范围 |
|---|---|---|---|
| **L1** | GPU 显存 | 计算直接使用 | 实例私有 |
| **L2** | 主机内存（pinned DRAM） | 本地扩展池，默认 ≥2× 显存 | 实例私有 |
| **L3** | 分布式存储（Mooncake / HF3FS / NIXL / AIBrix / EIC / 文件 …） | 全局共享池 | 集群内所有实例 |

L1 命中零拷贝；L2 命中需要 H2D 拷贝（逐层流水、与计算重叠）；L3 命中需要「存储读 → L2」再「L2 → L1」两步。本质是**用 I/O 时间换 prefill 计算时间**，在多轮对话、多文档 QA、agent 长上下文等前缀复用密集的场景收益显著。

### 1.2 元数据组织：HiRadixTree

HiCache 在 RadixTree 之上扩展出 HiRadixTree，核心类 `HiRadixCache(RadixCache)`（[hiradix_cache.py:76](python/sglang/srt/mem_cache/hiradix_cache.py:76)）。树中每个节点代表一段连续 token 的 KV，并用两组字段记录 KV 位于哪些层级——

- `node.value`（GPU slot 索引）非空 → 该段 KV 在 **L1**；
- `node.host_value`（host slot 索引）非空 → 该段 KV 在 **L2**；
- **L3 没有任何树上元数据**——访问时对 token 做逐页哈希，调用存储后端 `batch_exists` 实时查询（避免维护全局元数据的开销，见 [hicache_design.md](docs/advanced_features/hicache_design.md) 设计章节）。

派生属性：`node.evicted = (value is None)`（[:250](python/sglang/srt/mem_cache/radix_cache.py:250)）表示该节点已被驱逐出 L1（只剩 L2 副本），`node.backuped = (host_value is not None)`（[:254](python/sglang/srt/mem_cache/radix_cache.py:254)）表示 L2 上持有该段 KV。`HiRadixCache` 持有 `token_to_kv_pool_host`（L2 宿主池）与 `cache_controller`（`HiCacheController`，负责全部异步传输），且**只支持 MHA / MLA / DSA / MiniMax 稀疏注意力（MSA）四类设备池**（[hiradix_cache.py:84-109](python/sglang/srt/mem_cache/hiradix_cache.py:84)）；混合线性注意（Mamba 等）走 `HiMambaRadixCache`，SWA 走 SWA 专用 radix 变体。

### 1.3 请求生命周期速览

新请求 → `match_prefix` 在树上同时匹配出 L1 前缀长度与 L2 前缀长度 → L2 部分由 `HiCacheController` 分配 GPU slot 并逐层 H2D 加载（layer pipeline，prefill 边算边传）→ L1/L2 均未覆盖的尾巴按页哈希查 L3 → L3 命中量超过阈值则在后台 prefetch 到 L2（命中才进入下一轮调度）→ prefill 完成/驱逐时按写策略把新 KV 写回 L2，可选继续写 L3（只写缺的页，实现跨实例去重）。

---

## 2. L1：GPU 设备内存池

### 2.1 类层次与布局

设备端 KV 池定义在 [memory_pool.py](python/sglang/srt/mem_cache/memory_pool.py)：

| 池类 | 适用 | KV 布局 |
|---|---|---|
| `MHATokenToKVPool` | 标准 MHA/GQA/MoE | 每层独立的 `k_buffer[layer]` / `v_buffer[layer]`，形状 `[pool_size, head_num, head_dim]` |
| `MLATokenToKVPool` | DeepSeek 系列 MLA | 每层单个融合 `kv_buffer[layer]`（`kv_lora_rank + qk_rope_head_dim`） |
| `DSATokenToKVPool` | DeepSeek V4 稀疏注意力 | 复合池（MLA + 索引器） |
| `SWATokenToKVPool` | 滑窗注意力 | 「所有层全量池 + 滑窗层小池」双池（不在 HiRadixCache 支持范围） |
| `HybridLinearKVPool` | 全注意力 + 线性/Mamba | 复合 `full_kv_pool` + 线性池（HiCache 由 `HiMambaRadixCache` 对接） |

选择由 `model_runner_kv_cache_mixin.py` 和 `registry.py` 按注意力类型、`page_size`、`disaggregation_mode` 决定。

### 2.2 分配器（[allocator/](python/sglang/srt/mem_cache/allocator/)）

- **`TokenToKVPoolAllocator`（page_size=1）**：`free_pages`/`release_pages` 双队列模式——`free()` 把页先放入 release，整批操作结束统一 `merge` 回 free；保证同一次调度内的 page 选举稳定。
- **`PagedTokenToKVPoolAllocator`（page_size>1）**：按页分配，第 0 页保留；同上双队列，但 release 合并时还要做「**free_group 按前缀连续整组提交**」——保证已释放的页在原 radix 树/上级视图里依然构成连续整页，`HiCache` 的执行依赖该语义。

### 2.3 HiCache 复用的基础接口

- `get_kv_buffer(indices)` / `get_flat_data(indices)` / `transfer_flat_data(dst, src_indices)` 等，是所有设备↔宿主拷贝最终落地的入口（宿主池在内部循环调用）。
- 物理布局对全注意力模型是逐层分离的 `k_buffer`/`v_buffer` 列表（layer-first），这决定了 L2↔L1 的拷贝只有「逐层整块」才是高效的；L2 的物理布局（§3.4）决定传输是「逐层」还是「页内跨层聚合」。

### 2.4 L1 的驱逐

`HiRadixCache.evict(EvictParams)`（[hiradix_cache.py:1059](python/sglang/srt/mem_cache/hiradix_cache.py:1059)）在写策略间分流：

- **`_evict_write_through`**（[:1083](python/sglang/srt/mem_cache/hiradix_cache.py:1083)）：节点本就已经备份在 L2（`node.backuped`），直接 detach 并释放 GPU slot（`_evict_backuped`）→ `_detach_backuped`（[:1139](python/sglang/srt/mem_cache/hiradix_cache.py:1139)）；没被备份的（如 hit 数未达选择性阈值）走 `_evict_regular`（[:1158](python/sglang/srt/mem_cache/hiradix_cache.py:1158)）直接释放。
- **`_evict_write_back`**（[:1103](python/sglang/srt/mem_cache/hiradix_cache.py:1103)）：为「**待废弃**」路径（注释原文 "this path will be deprecated in the future"），仅当 `write_policy == "write_back"` 时启用。先对所有已备份节点走 `_evict_backuped`；未备份节点逐叶 `write_backup(write_back=True)`（host 分配失败且 `evict_host` 补不上即直接跳过），期间 `evict_device(device_indices)` 先保留（`_detach_backuped` 仅清 `node.value`），等全部 staging 完成后 `flush_staged()` 调 `cache_controller.evict_device` 释放真正的 GPU slot；`evict_device` 这一步前必须 `writing_check(write_back=True)` **阻塞等回写结束**。整个 staging 失败时用 `_drop_subtree_no_host`（[:1168](python/sglang/srt/mem_cache/hiradix_cache.py:1168)）丢弃无 host 的子树。

`evict_host(num_tokens)`（[:1209](python/sglang/srt/mem_cache/hiradix_cache.py:1209)）对 L2 的驱逐：从 `evictable_host_leaves` 集合中收集不在 L1、也无进行中的 write-back/prefetch/load 引用的 host 叶子，归还 `free_slots`。

---

## 3. L2：主机内存池

### 3.1 类层次

宿主池接口 `HostKVCache` 在 [pool_host/base.py:79](python/sglang/srt/mem_cache/pool_host/base.py:79)，唯一一份实现的类位于 [memory_pool_host.py](python/sglang/srt/mem_cache/memory_pool_host.py)（约 3600 行）：

| 类 | 用途 |
|---|---|
| `MHATokenToKVPoolHost`（[:89](python/sglang/srt/mem_cache/memory_pool_host.py:89)） | 标准 MHA 宿主池；`get_mha_host_pool_cls`（[:1249](python/sglang/srt/mem_cache/memory_pool_host.py:1249)）会按设备池 K/V head_dim 是否对称，返回该类或 `AsymmetricMHATokenToKVPoolHost`（[:947](python/sglang/srt/mem_cache/memory_pool_host.py:947)） |
| `MHATokenToKOnlyPoolHost`（[:637](python/sglang/srt/mem_cache/memory_pool_host.py:637)） | MiniMax 稀疏注意力的 K-only 宿主池 |
| `MLATokenToKVPoolHost`（[:1259](python/sglang/srt/mem_cache/memory_pool_host.py:1259)） | MLA 模型宿主池（带 `HiSparseHostPoolMixin`） |
| `MambaPoolHost`（[:1706](python/sglang/srt/mem_cache/memory_pool_host.py:1706)） | 混合线性注意力的 Mamba state 宿主池 |
| `DeepSeekV4PagedHostPool` / `DeepSeekV4StateHostPool`（[:2365](python/sglang/srt/mem_cache/memory_pool_host.py:2365) / [:2739](python/sglang/srt/mem_cache/memory_pool_host.py:2739)） | DeepSeek V4 DSA 的分页宿主池与 state 宿主池 |
| `DSAIndexerPoolHost`（[:3269](python/sglang/srt/mem_cache/memory_pool_host.py:3269)） | DSA 索引器宿主池 |
| `HostPoolGroup` / `PoolEntry`（[:3129](python/sglang/srt/mem_cache/memory_pool_host.py:3129) / [:3109](python/sglang/srt/mem_cache/memory_pool_host.py:3109)） | 混合池管理（多个子宿主池组成逻辑宿主池） |

分配与释放（[base.py:241-268](python/sglang/srt/mem_cache/pool_host/base.py:241)）在 `RLock` 同步下用 `free_slots` 张量切片完成；`alloc(need_size)` 断言 `need_size % page_size == 0`，并用 `slot_used` 位图检测 double-alloc/double-free。

> 说明：目录 `pool_host/` 是新的包宿主（`base.py / common.py / hisparse.py`），但 HiRadixCache 当前实例化的池类实现（`MHATokenToKVPoolHost` 等）位于 `memory_pool_host.py`。

### 3.2 容量换算

宿主容量由两个 arg 决定（**每个 rank 独立申请**）：

- `--hicache-ratio R`（默认 2.0）→ `size = device_pool.size * R`；
- `--hicache-size S`（默认 0）→ `size = S * 1e9 / size_per_token`，覆盖 ratio。

size 再向上对齐到 `page_size`（`page_num = size // page_size + 1`，[base.py:109-110](python/sglang/srt/mem_cache/pool_host/base.py:109)）；`size_per_token` 为每 token 全部层 KV 字节数之和（如 MHA 需乘 2 倍 k/v，MLA 只一份）。宿主内存申请使用 `cudaHostRegister`/`_cuda_host_unregister`（[common.py](python/sglang/srt/mem_cache/pool_host/common.py) 中的 `get_allocator_from_storage`），启动时会校验可用宿主内存（预留 `HICACHE_HOST_MEMORY_RESERVE_BYTES = 10 GiB` 给系统，[base.py:25, 123-137](python/sglang/srt/mem_cache/pool_host/base.py:25)）。

PP 下还有 `sync_fixed_hicache_size`（[base.py:28](python/sglang/srt/mem_cache/pool_host/base.py:28)）在 pp_group 内取最小 token 容量，保证各 stage 使用相同 host size。

### 3.3 宿主内存布局（`--hicache-mem-layout`）

| 布局 | 内存组织 | L2→L1 拷贝粒度 | 与 L3 交互 |
|---|---|---|---|
| `layer_first` | 与 GPU 相同的逐层 buffer | 整层一次拷贝 | 非零拷贝（经 `get_data_page`/`set_from_flat_data_page` 转换） |
| `page_first`（默认） | 按页连续 | 整页逐层拷贝 | 零拷贝 |
| `page_first_direct` | 页内再按层聚合（每页 = layer × token × head × dim 连续 blob） | **页-层级**整块拷贝 | 零拷贝 |
| `page_first_kv_split` | page_first 变体，K/V 分块 | 逐层 | 供 DSA 等（见 [memory_pool_host.py:1348](python/sglang/srt/mem_cache/memory_pool_host.py:1348)） |
| `page_head` | page_first 且按 head 分片 | 逐层 | mooncake 支持下可用 `tp_lcm_size` 做异构 TP key 拆分 |

合法化流程在 `ServerArgs._handle_hicache`（[server_args.py:5506-5598](python/sglang/srt/server_args.py:5506)）：

- `page_first_direct + kernel` → 强制改为 `direct` I/O；
- `page_first + direct` → 改写为 `page_first_direct`；
- ROCm 上 `page_first + kernel` → 退化为 `layer_first`（JIT 写回内核是 CUDA-only）；
- mooncake + `layer_first` → 改写为 `page_first_direct`（direct）或 `page_first`（kernel）。

**选型**：`kernel` I/O + `page_first` 是社区推荐基准（零拷贝 + 基于 JIT 的 GPU 辅助拷贝内核，自测可达 cudaMemcpyAsync 的 ~3×）；必须 `direct` 时配 `page_first_direct`。

### 3.4 传输关键 API

宿主侧抽象（[base.py:174-211](python/sglang/srt/mem_cache/pool_host/base.py:174)）：

- `load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend)` / `backup_from_device_all_layer(...)`：H2D/D2H 的主力 — `HiCacheController` 在 `load_stream` 上逐层调用，在 `write_stream` 上一次性备份全部层。
- `get_data_page(index, flat)` / `set_from_flat_data_page(index, data_page)` / `get_dummy_flat_data_page()`：L3 **非零拷贝**通道（`HiCacheStorage.batch_get/batch_set` 老接口的翻译层），把宿主一页（或 flat 化）取出/写回。
- `is_stride_page_aligned(4096)`：页步长 4KiB 对齐探测，mooncake/hf3fs/O_DIRECT 类后端用此判定能否直读（[base.py:213](python/sglang/srt/mem_cache/pool_host/base.py:213)；默认 False → copy 模式）。

**空白区一致性**：页内尾部 padding slot 的 KV 可能没有值，HiRadixCache 在写 L2/L3 前对 padding 位置做一致化（fill），保证两个实例对「同一前缀」写出的字节完全一致——这是 L3 按内容哈希去重的正确性前提。

### 3.5 传输调度与同步（HiCacheController）

调度器每个 TP rank 持有一个 `HiCacheController`（[managers/cache_controller.py:210](python/sglang/srt/managers/cache_controller.py:210)），两条专用 CUDA stream：`write_stream`（D2H）与 `load_stream`（H2D）。

- **CacheOperation**（[:106](python/sglang/srt/managers/cache_controller.py:106)）：持有 `host_indices`/`device_indices`/`node_ids`/priority。`start_writing` / `start_loading` 会把 `write_queue` / `load_queue` 中所有 op **合并为一个 CacheOperation** 一次下发（`merge_ops`，[:128](python/sglang/srt/managers/cache_controller.py:128)）。
- **ack 队列**：`ack_write_queue` / `ack_load_queue` 元素为 `HiCacheAck(start_event, finish_event, node_ids)`（[:147](python/sglang/srt/managers/cache_controller.py:147)）——由 `HiRadixCache.writing_check()` / `loading_check()` 在调度 loop 中轮询完成。多 rank 下「finish 数量不一致」会造成 NCCL 集合动作错位，故 `writing_check` 走 `pp_rank==0` 判断 + `_all_reduce(MIN)` 决定本回合是否有完成的 op（[hiradix_cache.py:944-966](python/sglang/srt/mem_cache/hiradix_cache.py:944)，注释明确说明这是为防 TP>1 死锁的全员强 collective）。
- **逐层加载流水**：`LayerDoneCounter`（[cache_controller.py:75](python/sglang/srt/managers/cache_controller.py:75)）持有 3 组 `LayerLoadingEvent` 环形槽（适配 overlap 模式），producer 每完成一层 `complete(i)` 记录 event，attention forward 内 `wait_until(layer)` 等待第 i 层——**第 N 层边算边拷第 N+1 层**的 compute-transfer overlap（[:779-797](python/sglang/srt/managers/cache_controller.py:779)）。
- **索引整理**：`move_indices`（[:744](python/sglang/srt/managers/cache_controller.py:744)）按 `io_backend` 调整索引——`kernel` 时把 host_indices 移上 GPU；`direct + layer_first` 时 host_indices 排序并同步重排 device_indices；`direct + page_first_direct` 时 device_indices 移到 CPU。
- **页首布局 JIT 捷径**（[:688-697](python/sglang/srt/managers/cache_controller.py:688)）：`kernel + page_first` 且 `can_use_write_back_jit` 时跳过 `move_indices`，直接把 CPU 端副索引给 JIT 内核。
- **容量背压**：`prefetch_capacity_limit = int(0.5 * mem_pool_host.size)`（[:474](python/sglang/srt/managers/cache_controller.py:474)）保证 prefetch 至多占用宿主池一半，其余留给回写暂存路径，防 prefetch 自己把 L2 打满。

### 3.6 L2→L3 写回的去重

`write_backup_storage(node)`（[hiradix_cache.py:861](python/sglang/srt/mem_cache/hiradix_cache.py:861)）在节点被 ack 后触发：

1. 取 `node.hash_value`（每页链式 SHA‑256）——`_concat_split_chain`（[:888](python/sglang/srt/mem_cache/hiradix_cache.py:888)）会先把分裂开的多段 hash 串接成连续链；
2. 调 `storage_backend.batch_exists(...)` 查 L3 已有页；
3. **只对缺失的页**调用 `write_storage(host_indices, token_ids, hash_value, prefix_keys)` 进入 backup_queue（[cache_controller.py:1073](python/sglang/srt/managers/cache_controller.py:1073)）。

`batch_exists` 的 contract 是「返回**前缀连续**命中页数」（[hicache_storage.py:301](python/sglang/srt/mem_cache/hicache_storage.py:301)），保证后续页的哈希链依赖前缀不被截断。跨实例共享下，第二个实例只会写入它的第一个未命中页起始的尾部——这就是集群内 dedup 的来源。

---

## 4. L3：分布式存储后端

### 4.1 抽象接口与配置

基类 `HiCacheStorage`（[hicache_storage.py:142](python/sglang/srt/mem_cache/hicache_storage.py:142)）暴露三代接口：

- 老接口 `get/set/batch_get/batch_set`（张量数据）；
- 零拷贝 v1 `batch_get_v1/batch_set_v1(keys, host_indices, extra_info)`（[:212](python/sglang/srt/mem_cache/hicache_storage.py:212)）——传宿主槽位，后端直接读写；
- v2 `batch_exists_v2`（[:157](python/sglang/srt/mem_cache/hicache_storage.py:157)）以及 per‑pool 的 `batch_get_v2/batch_set_v2`（`PoolTransfer`，用于 draft KV 等副池）。

`HiCacheStorageConfig`（[:27](python/sglang/srt/mem_cache/hicache_storage.py:27)）由 `HiCacheController._generate_storage_config`（[cache_controller.py:570-633](python/sglang/srt/managers/cache_controller.py:570)）生成，携带 `tp_rank/tp_size/pp_*/attn_cp_*`、`is_mla_model`、`is_page_first_layout`、`tp_lcm_size`、`cp_cache_layer_split`、`extra_config`。

### 4.2 Key：逐页链式哈希

- 计算入口 `get_hash_str`（[utils.py:106](python/sglang/srt/mem_cache/utils.py:106)）→ `get_native_hash(token_ids, prior_digest, page_size)`：逐页取 `page_token_ids` 的 SHA‑256，**父页 digest 作为下页哈希盐**（只要前缀任一字节不同，就会雪崩出不同 key）；
- 树节点侧 `compute_node_hash_values`（[utils.py:122](python/sglang/srt/mem_cache/utils.py:122)）把父节点的最后一段哈希作为 `prior_hash` 继续链式推导；
- 后端侧 `_get_component_key` 会把 `model_name`/`is_mla_model`/`tp_rank`（MLA 例外）等信息拼入 key 前缀（如 mooncake 的 `:key:tp_rank`），bool 前缀由 `HiCacheStorageConfig` 控制。

`HiCacheStorageExtraInfo`（[:45](python/sglang/srt/mem_cache/hicache_storage.py:45)）携带 `prefix_keys`（本次已确认命中的前缀链）给需要前缀感知统计的后端（eic / hf3fs 等）；由 `hicache_storage_pass_prefix_keys=true` 打开。

### 4.3 内建后端一览

以 `backend_factory.py` 的 `register_backend(...)` 为准（[:194-239](python/sglang/srt/mem_cache/storage/backend_factory.py:194)）：

| 名称 | 类 / 位置 | 说明 |
|---|---|---|
| `file` | `HiCacheFile`（[hicache_storage.py:321](python/sglang/srt/mem_cache/hicache_storage.py:321)） | 本地文件演示；目录下有 `storage/file/lru_file_evictor.py` 提供 LRU 驱逐器（注意 `file_lru` **不是**注册的 backend 名） |
| `mooncake` | `MooncakeStore`（[storage/mooncake_store/mooncake_store.py](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py)） | RDMA + 多网卡；要求 page_first/page_first_direct/page_head；MLA 模型仅 rank 0 真写；`page_head` + `tp_lcm_size` 支持异构 TP 复用 |
| `hf3fs` | `HiCacheHF3FS`（[storage/hf3fs/storage_hf3fs.py](python/sglang/srt/mem_cache/storage/hf3fs/storage_hf3fs.py)） | Fire-Flyer File System，O_DIRECT/大页约束，直读 path |
| `nixl` | `HiCacheNixl`（[storage/nixl/hicache_nixl.py](python/sglang/srt/mem_cache/storage/nixl/hicache_nixl.py)） | NVIDIA Inference Xfer Library，对接 3FS/GDS/S3 等插件，支持 P2P 模式 |
| `aibrix` | `AibrixKVCacheStorage`（[storage/aibrix_kvcache/](python/sglang/srt/mem_cache/storage/aibrix_kvcache/)） | AIBrix 生产级级联 KV Offloading |
| `eic` | `EICStorage`（[storage/eic/eic_storage.py](python/sglang/srt/mem_cache/storage/eic/eic_storage.py)） | 企业级内存计算里的弹性 in-context / RAG 后端 |
| `simm` | `HiCacheSiMM`（[storage/simm/hicache_simm.py](python/sglang/srt/mem_cache/storage/simm/hicache_simm.py)） | SGLang-native 内存池；仅支持 page_first/page_first_direct 布局 |
| `mori` | `UMBPStore`（[storage/umbp/umbp_store.py](python/sglang/srt/mem_cache/storage/umbp/umbp_store.py)） | UMBP 统一缓存池存储 |
| `dynamic` | 运行时加载 | `--hicache-storage-backend-extra-config '{"backend_name":..., "module_path":..., "class_name":...", "interface_v1":0/1}'`，由 `StorageBackendFactory._load_backend_class` 动态 import 并要求 `issubclass(HiCacheStorage)`（[backend_factory.py:16-42](python/sglang/srt/mem_cache/storage/backend_factory.py:16)） |

零拷贝选择：`HiCacheController.attach_storage_backend` 中对 `["hf3fs", "mooncake", "eic", "nixl", "simm", "mori"]`（以及 dynamic+interface_v1=1）注册 `_page_get_zero_copy`/`_page_set_zero_copy`（[cache_controller.py:486-494](python/sglang/srt/managers/cache_controller.py:486)）；其余走 `_generic_page_get/_generic_page_set`（经 `get_data_page`/`set_from_flat_data_page` 中转）。

⚠️ **`STORAGE_BATCH_SIZE = 128`**（[hicache_storage.py:22](python/sglang/srt/mem_cache/hicache_storage.py:22)）：prefetch/backup 内部按 128 页一批走。

### 4.4 Prefetch 线程模型

`HiCacheController` 在 storage attach 时启动常驻线程（[cache_controller.py:345-367](python/sglang/srt/managers/cache_controller.py:345)）：

```
prefetch_thread_func    消费 prefetch_queue（调度器 put），prefetch_buffer 中转
  ├─ _storage_hit_query：按页哈希 → batch_exists（128页/批），链式 prefix_keys
  ├─ _all_reduce_prefetch_groups(MIN) 得命中页数
  ├─ 命中 < prefetch_threshold(≥page_size) → 撤销 + 释放全部预分配 host 槽
  └─ 否则 hash_value 截断到命中页，入 prefetch_buffer
       未命中尾部 host 槽入 host_mem_release_queue
prefetch_io_aux_thread  消费 prefetch_buffer
  ├─ _page_transfer：每批 128 页调 page_get_func，operation.increment 计数
  ├─ 已完成页 prefix_keys 追加；失败或 mark_terminate → break
  └─ 尾部未完成 host 槽 → host_mem_release_queue
backup_thread_func      消费 backup_queue
  ├─ _page_backup：page_set_func（MLA 且 rank>0 时 controller 的 backup_skip=True 直接跳过）
  ├─ draft KV best-effort piggyback（_draft_page_set）
  └─ 完成 → ack_backup_queue（调度器后续 free 对应 host 槽）
```

**同步组**：prefetch/backup 集合操作用独立的 gloo group（`_create_prefetch_sync_groups`，[cache_controller.py:309-331](python/sglang/srt/managers/cache_controller.py:309)）——优先 attn_cp_group/attn_tp_group，其次 tp_group；避免与主 NCCL 冲突。这一层 group 在 attach 时创建、detach 时销毁。
**容量背压**：`prefetch_tokens_occupied`（由调度器在 `prefetch_from_storage` 中累加）达 `0.5 × host_pool.size` 即 `prefetch_rate_limited()` 返回 True，新 prefetch 被拒（[cache_controller.py:992-1000](python/sglang/srt/managers/cache_controller.py:992)）。

### 4.5 Prefetch 三策略

`--hicache-storage-prefetch-policy` ∈ {`best_effort`, `wait_complete`, `timeout`}（默认 `timeout`），实现在 `HiRadixCache.can_terminate_prefetch(operation)`（[hiradix_cache.py:1447](python/sglang/srt/mem_cache/hiradix_cache.py:1447)）：

- `best_effort`：随着命中页增长随时「够用即走」；
- `wait_complete`：必须等到 `operation.hash_value` 的页全部 increment 到；
- `timeout`：走 `_prefetch_timeout_check_linear_func`（[:1441](python/sglang/srt/mem_cache/hiradix_cache.py:1441)）——`prefetch_timeout_base + prefetch_timeout_per_ki_token × num_token_to_fetch / 1024`，两个系数由 `--hicache-storage-backend-extra-config` 提供。

达到条件后 `terminate_prefetch(req_id)` 标记 op、返回最终的 `completed_tokens`，由其「把已加载部分归根到 `_insert_helper_host` 插入树、释放未达成页」。

### 4.6 Speculative draft KV

`HiCacheController.set_draft_kv_pool(draft_device_pool, draft_host_pool)`（[cache_controller.py:826](python/sglang/srt/managers/cache_controller.py:826)）注册投机解码草稿模型的池：所有 D2H/H2D 传输自动追加一次草稿对应拷贝；L3 前缀加 `.draft`（generic，[:1144](python/sglang/srt/managers/cache_controller.py:1144)）或走 v2 `PoolName.DRAFT`（mooncake，[:1122-1142](python/sglang/srt/managers/cache_controller.py:1122)）。
**已知受限后端**：`hf3fs/eic/nixl/simm` 的 draft L3 被显式禁用（warn，[:866-872](python/sglang/srt/managers/cache_controller.py:866)）；mooncake 在 `should_split_heads=True` 时也禁用 draft v2（[:852-857](python/sglang/srt/managers/cache_controller.py:852)）。

---

## 5. 写策略与一致性

### 5.1 三种写策略（`--hicache-write-policy`）

| 策略 | L1→L2 何时写 | 关键代码 |
|---|---|---|
| `write_through`（server_args 默认，`write_through_threshold = 1`） | 每次请求**写入树**时（`cache_finished_req` / `cache_unfinished_req` 的 insert 路径） | `RadixCache._insert_helper` 对每个走过的节点调 `self._inc_hit_count(node)`（[:741-755](python/sglang/srt/mem_cache/radix_cache.py:741)），HiRadixCache 重写它（[:922-931](python/sglang/srt/mem_cache/hiradix_cache.py:922)）：`hit_count ≥ 1` 即触发 `write_backup`（[:784](python/sglang/srt/mem_cache/hiradix_cache.py:784)）；污染传制（`node.parent.backuped`）不满足时直接跳过 |
| `write_through_selective`（阈值 = 2） | 同上 | 唯一差别是 `_inc_hit_count` 里 `hit_count ≥ 2` 才触发备份；轻易复用才不写，单次前缀不买账 |
| `write_back` | 只在 L1 evict 时（`、`write_policy == "write_back"` 由 `cache_controller.write_policy` 判断） | `_evict_write_back`（[:1103](python/sglang/srt/mem_cache/hiradix_cache.py:1103)）+ `writing_check(write_back=True)` 阻塞（[:933-942](python/sglang/srt/mem_cache/hiradix_cache.py:933)）；代码内明确标注将废弃 |

⚠️ **`hit_count` 在 insert（`cache_finished_req`/`cache_unfinished_req` → `_insert_helper`）时递增**（[radix_cache.py:741, 745, 755](python/sglang/srt/mem_cache/radix_cache.py:741)），而不是在 `match_prefix` 路径。`match_prefix` 只对 L2 命中长度进行计数，不会累加节点热度——因此「热度阈值」在**同一前缀被第二个同类请求「写」进树**（如 prefill 完成后的 insert，当前 chunk 除外）时生效。

### 5.2 关键不变量

- **页对齐写入**：写 L2/L3 前 `key.page_aligned(page_size)` 对齐（[hiradix_cache.py:1567](python/sglang/srt/mem_cache/hiradix_cache.py:1567)），未对齐的尾段先以 `insert` 挂 LRU `protect` 防驱逐。
- **分级连续链**：`last_host_node`（`:1578-1583`）保证「L1 的前缀尾巴」和「L2 前缀头」在树上连续；`match_prefix` 序列在 key 上先 `page_aligned`（[:1567](python/sglang/srt/mem_cache/hiradix_cache.py:1567)）+ `_match_prefix_helper` 按页匹配（[:1697](python/sglang/srt/mem_cache/hiradix_cache.py:1697)），保证返回的 host/device index 都按页整段对齐。

> 术语小注：`value` 字段在 `node.value` / `last_node.value` 之间语义不同——`node.value` 存的是**该 node 自己的 GPU slot 段**，而 `match_prefix` 返回的 `device_indices` 是沿链收集后 `torch.cat` 的完整 L1 前缀。
- **在途保护**：`_track_write_through_node`（[:817](python/sglang/srt/mem_cache/hiradix_cache.py:817)）在 `write_backup` 成功后把 `(node, backup_len, [node])` 存入 `ongoing_write_through[node.id]`；node 中途被 `_split_node` 分裂时 `_replace_pending_write_through_node`（[:821](python/sglang/srt/mem_cache/hiradix_cache.py:821)）会把该 id 转移给新 node；`HiCacheAck` 在 `writing_check` 中完成时 `_finish_write_through_ack`（[:849](python/sglang/srt/mem_cache/hiradix_cache.py:849)）record CPU medium 事件、调 `write_backup_storage`（在 enable_storage 下）、并 `dec_lock_ref`。
- **全员集合通信**：`writing_check`/`loading_check` 无「短路跳过」——注释直接说明：即使一侧无 in-flight op，也要跟 `all_reduce(MIN)`，否则 TP>1 下 NCCL op 序列错位死锁（[:944-951](python/sglang/srt/mem_cache/hiradix_cache.py:944) 的注释）。

---

## 6. 预取与匹配端到端

`match_prefix`（[hiradix_cache.py:1561-1592](python/sglang/srt/mem_cache/hiradix_cache.py:1561)）序列：

```
match_prefix(params)
├── key.page_aligned(page_size)
├── _match_prefix_helper(root, key)                              # [hiradix_cache.py:1697]
│     while 按页前进：
│     • 完整匹配 child → not child.evicted 时 append(child.value)  # GPU 段累积
│     • child 中途命中 → _split_node 分裂，分裂出新段同时收 GPU 值
│     • child.evicted（host_value）不进 value 名单——host_hit_length 在返回后另外在 match_prefix 内最后累计
├── value 张量 cat → device_indices（仅含**未驱逐** segment 的 GPU slot）
└── MatchResult(device_indices, last_device_node=首个未驱逐节点, last_host_node=向上找可备份到的节点,
                best_match_node=last_host_node，host_hit_length=沿 last_node 向上的 evicted 祖先总和)
```

L3 在 `prefetch_from_storage(req_id, last_host_node, new_input_tokens, last_hash, prefix_keys)`（[:1594-1651](python/sglang/srt/mem_cache/hiradix_cache.py:1594)）发起：

- 调度器在 `get_new_batch_prefill` 阶段（scheduler `match_prefix` 之前，[scheduler.py:2509](python/sglang/srt/managers/scheduler.py:2509) 一带）对 `last_host_node` 还在等待队列的请求发起。
- 由 `prefetch_threshold`（`extra_config.prefetch_threshold`，默认 256，`max(threshold, page_size)` 生效于 [cache_controller.py:472](python/sglang/srt/managers/cache_controller.py:472)）+ `prefetch_capacity_limit` 双重限流。
- `ongoing_prefetch[req_id]` 记录；调度器下一轮（scheduler.py:3096 附近）调 `check_prefetch_progress(req_id)`（[hiradix_cache.py:1488](python/sglang/srt/mem_cache/hiradix_cache.py:1488)）——只有 `can_terminate_prefetch` 返回 True 才 conclude，然后 `_insert_helper_host`（[:1653](python/sglang/srt/mem_cache/hiradix_cache.py:1653)）把实际拉到的页挂进树，释放未达成的 host 槽。该函数注释明确（[:1492-1493](python/sglang/srt/mem_cache/hiradix_cache.py:1492)）：**best-effort prefetch 在「请求出队」时终止**——也就是 best_effort 策略下的实际语义是写满调度可用即停，而不是空闲拉遍。

---

## 7. 分布式：TP / PP / cp / dp

- **TP（或 dp-atten 下的 attn_tp/attn_cp 组）**：prefetch 同步用独立 gloo group；MLA/DeepSeekV4 压缩 MLA（`DeepSeekV4TokenToKVPool`）rank 去重——`HiCacheController.attach_storage_backend` 里 `backup_skip = is_rank_replicated and rank != 0`（[cache_controller.py:448-459](python/sglang/srt/managers/cache_controller.py:448)）；`cp_cache_layer_split` 开 `attn_cp` 时把主 rank 取 `attn_tp_rank == 0` 当主 writer。
- **`tp_lcm_size`（跨 TP 复用）**：MHA + mooncake + `page_head` 布局 + `tp_lcm_size > tp_size` 时把 head 按 `tp_lcm_size` 拆分，允许 tp=4/tp=8 双集群共享同一 L3 命名空间（[hicache_best_practices.md](docs/advanced_features/hicache_best_practices.md)；判定 [cache_controller.py:601-613](python/sglang/srt/managers/cache_controller.py:601)）。
- **PP**：HiRadixCache 只在 **PP rank 0** 做「已完成写操作的 `finish_event.query()`」；完成数 `finish_count` 经 `_all_reduce(MIN)` 同步到 pp 所有 rank，再集体消费 ack（[:950-966](python/sglang/srt/mem_cache/hiradix_cache.py:950)）。宿主池 `host_size` 也通过 `sync_fixed_hicache_size` 在 pp 组内取 MIN（[base.py:28-67](python/sglang/srt/mem_cache/pool_host/base.py:28)）。
- **cp_cache_layer_split**：DSA 的层切组件，`is_cp_cache_layer_split_pool`（`mem_cache/cp_cache_layer_split.py`）被 storage config 携带（[cache_controller.py:599, 631](python/sglang/srt/managers/cache_controller.py:599)），供后端处理 CP-attention 的层 export layout。
- **PD disaggregation**：prefill 节点上 HiCache 原生跑；decode 节点用 `--disaggregation-decode-enable-offload-kvcache` 异步把 decode 输出 KV 回写 L2/L3（官方文档部署模式）。

---

## 8. 运行时管理与指标

### 8.1 runtime attach/detach

HTTP API：`GET / PUT / DELETE /hicache/storage-backend`（详见 [docs/advanced_features/hicache_storage_runtime_attach_detach.md](docs/advanced_features/hicache_storage_runtime_attach_detach.md)；入口 `scheduler.is_fully_idle()` 强 idle 门禁）。控制路径 HTTP Server → TokenizerManager → Scheduler → `HiRadixCache.attach_storage_backend`（[hiradix_cache.py:361](python/sglang/srt/mem_cache/hiradix_cache.py:361) / `detach_storage_backend` [:479](python/sglang/srt/mem_cache/hiradix_cache.py:479)）→ `HiCacheController` 对称接口；detach 不会清 L3 上已有数据。attach 失败会自动 stop 已启动的线程并清理 process group（[cache_controller.py:501-524](python/sglang/srt/managers/cache_controller.py:501)）。dp>1 时要求全部 DP rank 各自 idle，成功需要全部成功。

### 8.2 指标

- `enable_storage_metrics=true` 时 HiRadixCache 通过 `StorageMetricsCollector` 暴露 `prefetch_hit_tokens`、`prefetch_miss_tokens`、`backup_tokens`、b/w 直方图等到 Prometheus，标签带 rank/tp_size/pp_rank 等（[hicache_storage.py](python/sglang/srt/mem_cache/hicache_storage.py) 头部 import 的 `storage.metrics_collector`，eic 等后端自定义扩展）。
- 调度侧 `--enable-metrics --enable-cache-report` 另提供 `cache_finished_req` 的 cached_token/total 计数。

---

## 9. 常见限制与调优注意点

- **host pool 必须大于 device pool**，否则 `HostKVCache` 会打 warning（[base.py:114-121](python/sglang/srt/mem_cache/pool_host/base.py:114)）；本地宿主可用内存强制保留 10GiB。
- **prefetch/backup 按 `STORAGE_BATCH_SIZE = 128` 页一批**处理（[hicache_storage.py:22](python/sglang/srt/mem_cache/hicache_storage.py:22)）；前一批失败会触发 `mark_terminate` 停止该 op 的后续页——批大小对总延迟有直接影响。
- **prefetch_threshold 下限是 `page_size`**（[cache_controller.py:472](python/sglang/srt/managers/cache_controller.py:472)），页比 256 大时原阈值自动失效。
- **prefetch_capacity_limit = 0.5 × host pool**：高并发 prefetch 即使阈值满足也可能被限流（[hiradix_cache.py:1613](python/sglang/srt/mem_cache/hiradix_cache.py:1613) 的 `prefetch_rate_limited()`）。
- **写穿透阈值硬编码 1/2**：`write_through_selective` 与 `write_through` 的差异只是「`hit_count≥2` 才备份」vs「≥1」，没有额外 CLI 参数可调（[:203-205](python/sglang/srt/mem_cache/hiradix_cache.py:203) 的 todo 注释承认）。
- **页首布局直通路径的 JIT 内核对 element_size 有 128B 对齐约束**（MLA/MHA 的 `can_use_write_back_jit` 检查），不满足时自动降级到 `move_indices` 通用路径（[memory_pool_host.py:196, 370](python/sglang/srt/mem_cache/memory_pool_host.py:196) / [cache_controller.py:688-697](python/sglang/srt/managers/cache_controller.py:688)）。
- **L3 key 使用链式页哈希**：上游「前缀相同」对 key 链是充要条件；system prompt 哪怕末尾换一个 token，也会让后续所有页哈希雪崩。
- **HiRadixCache 只支持 MHA/MLA/DSA/MSA**；SWA 与 Mamba/hybrid 请走 `SWARadixCache*` 或 `HiMambaRadixCache`。
- **`HiCacheController.reset()`（[cache_controller.py:635](python/sglang/srt/managers/cache_controller.py:635)）重启线程但保留 `enable_storage`**：`prefetch/backup` 队列被清空，`prefetch_tokens_occupied` 归零，`storage_stop_event` 清除并重新拉起线程——它跟在 `_stop_storage_threads` 后的幂等设计不同，调用前置依赖调度器自身 idle。

---

## 10. 快速验证

```bash
# 1) 基础 HiCache（无 L3）
python -m sglang.launch_server \
  --model-path <model> --tp 8 --page-size 64 \
  --enable-hierarchical-cache --hicache-ratio 2 \
  --enable-metrics --enable-cache-report

# 2) 检查启动日志中的参数规范化结果（按 _resolve_layout_io_compatibility 改写后再确认真正生效的 layout / io_backend）

# 3) 单测（居于 radix tree 的 hicache 流程）
pytest test/srt/mem_cache/test_hiradix_cache.py test/srt/mem_cache/test_hiradix_cache_mla.py

# 4) 运行时切 storage（无需重启）
curl -X PUT http://127.0.0.1:30000/hicache/storage-backend \
  -H 'Content-Type: application/json' \
  -d '{"hicache_storage_backend": "file", "hicache_storage_backend_extra_config_json": "{\"prefetch_threshold\": 512}"}'

# 5) PD 部署 decode 侧 async offload
#    decode 节点额外加 --disaggregation-decode-enable-offload-kvcache，详见 hicache_best_practices.md
```

---

## 附录 A：HiCache 相关启动参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--enable-hierarchical-cache` | off | 启用 HiCache |
| `--hicache-ratio` | 2.0 | 宿主池 / 设备池 token 数比 |
| `--hicache-size` | 0 | 宿主池 GB（每 rank），>0 覆盖 ratio |
| `--hicache-write-policy` | `write_through` | `write_back` / `write_through` / `write_through_selective` |
| `--hicache-io-backend` | `kernel` | `direct` / `kernel` / `kernel_ascend` |
| `--hicache-mem-layout` | `page_first` | `layer_first` / `page_first` / `page_first_direct` / `page_first_kv_split` / `page_head` |
| `--hicache-storage-backend` | None | `file` / `mooncake` / `hf3fs` / `nixl` / `aibrix` / `eic` / `simm` / `mori` / `dynamic`（注意 `file_lru` 未作为 backend 注册；文件后端的 LRU 驱逐在 `storage/file/lru_file_evictor.py`） |
| `--hicache-storage-prefetch-policy` | `timeout` | `best_effort` / `wait_complete` / `timeout` |
| `--hicache-storage-backend-extra-config` | None | JSON 或 `@file.(toml/json/yaml)`；常用键：`prefetch_threshold`（默认 256）/ `prefetch_timeout_base` / `prefetch_timeout_per_ki_token` / `tp_lcm_size` / `hicache_storage_pass_prefix_keys` / dynamic 后端的 `backend_name`/`module_path`/`class_name`/`interface_v1` |
| 配套 `--page-size` | 1 | 页大小，>1 时分配、传输、L3 key 全部按页对齐 |
| 配套 `--disaggregation-decode-enable-offload-kvcache` | off | PD decode 节点异步回写 KV |

完整列表以 [server_args.py 的 Hierarchical cache 段，:2028-2094](python/sglang/srt/server_args.py:2028) 为准。

## 附录 B：关键源码入口速查

| 主题 | 位置 |
|---|---|
| Radix tree 三级协调 | [hiradix_cache.py](python/sglang/srt/mem_cache/hiradix_cache.py) |
| 传输调度 / 控制器 | [managers/cache_controller.py](python/sglang/srt/managers/cache_controller.py) |
| L1 池 | [memory_pool.py](python/sglang/srt/mem_cache/memory_pool.py) + [allocator/](python/sglang/srt/mem_cache/allocator/) |
| L2 宿主池接口 | [pool_host/base.py](python/sglang/srt/mem_cache/pool_host/base.py)（抽象 + alloc/free + 容量校验） |
| L2 宿主池实现 | [memory_pool_host.py](python/sglang/srt/mem_cache/memory_pool_host.py) |
| L3 抽象 / 配置 / HiCacheFile | [hicache_storage.py](python/sglang/srt/mem_cache/hicache_storage.py) |
| L3 后端工厂 | [storage/backend_factory.py](python/sglang/srt/mem_cache/storage/backend_factory.py) |
| L3 各后端 | [storage/](python/sglang/srt/mem_cache/storage/) 下 `mooncake_store/ hf3fs/ nixl/ aibrix_kvcache/ eic/ simm/ umbp/ file/ lmcache/` |
| 页哈希 | [mem_cache/utils.py `get_hash_str`，:106](python/sglang/srt/mem_cache/utils.py:106) |
| 参数合法化 | [server_args.py `_handle_hicache`，:5506](python/sglang/srt/server_args.py:5506) |
| Runtime attach/detach | [docs/advanced_features/hicache_storage_runtime_attach_detach.md](docs/advanced_features/hicache_storage_runtime_attach_detach.md) |
| 官方设计与最佳实践 | [docs/advanced_features/hicache_design.md](docs/advanced_features/hicache_design.md)，[docs/advanced_features/hicache_best_practices.md](docs/advanced_features/hicache_best_practices.md) |
