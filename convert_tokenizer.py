from transformers.integrations.tiktoken import convert_tiktoken_to_fast
from tiktoken import get_encoding
from transformers import AutoTokenizer, PreTrainedTokenizerFast
def convert_kimi_to_fast(original_tokenizer, save_path):
    # 1. 准备所有特殊标记
    def get_token_content(token):
        """统一处理普通字符串和AddedToken对象"""
        if hasattr(token, 'content'):  # 处理AddedToken对象
            return token.content
        return str(token)

    # 1. 提取原始tokenizer的特殊标记映射
    original_special_mapping = {
        token: idx
        for idx, token in original_tokenizer.added_tokens_decoder.items()
        if isinstance(token, str) or hasattr(token, 'content')
    }

    # 2. 按原始ID顺序组织特殊标记
    special_tokens_in_order = [
        original_tokenizer.added_tokens_decoder[idx]
        for idx in sorted(original_tokenizer.added_tokens_decoder.keys())
    ]

    # 2. 执行转换
    converter = TikTokenConverter(
        vocab_file=original_tokenizer.vocab_file,
        pattern=original_tokenizer.model._pat_str,
        **original_special_mapping
    )
    tokenizer_obj = converter.converted()

    # 3. 创建Fast Tokenizer
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_obj,
        **special_tokens_in_order,
        legacy=True,
        add_bos_token=False,  # 禁用 BOS
        add_eos_token=False   # 禁用 EOS
    )

    # 4. 处理可能遗漏的added_tokens
    added_tokens = {}
    for idx, token in original_tokenizer.added_tokens_decoder.items():
        if isinstance(token, dict):
            added_tokens[idx] = token["content"]
        else:
            added_tokens[idx] = get_token_content(token)

    missing_tokens = [
        token for token in added_tokens.values()
        if token not in fast_tokenizer.get_added_vocab()
    ]

    if missing_tokens:
        # 检查哪些是特殊标记
        special_to_add = [
            t for t in missing_tokens
            if t in special_tokens_map.values() or
            any(t in get_token_content(spec) for spec in original_tokenizer.special_tokens_map.values())
        ]

        if special_to_add:
            fast_tokenizer.add_special_tokens({"additional_special_tokens": special_to_add})

        # 添加普通token
        normal_to_add = [t for t in missing_tokens if t not in special_to_add]
        if normal_to_add:
            fast_tokenizer.add_tokens(normal_to_add)

    # 5. 保存
    fast_tokenizer.save_pretrained(save_path)

    return fast_tokenizer
tokenizer = AutoTokenizer.from_pretrained("/models/model",trust_remote_code=True)
print(f"tokenizer {tokenizer.model}")
print(f"tokenizer name {tokenizer.model.name}")
from transformers.convert_slow_tokenizer import TikTokenConverter

tokenizer_con = TikTokenConverter(
        vocab_file="/models/model/tiktoken.model", pattern=tokenizer.model._pat_str, additional_special_tokens=tokenizer.model._special_tokens
    ).converted()
tokenizer_con.save("/models/model/fast/tokenizer.json")
fast_tokenizer = convert_kimi_to_fast(tokenizer, "/models/model/fast/")

tokenizer_fast = AutoTokenizer.from_pretrained("/models/model/fast/")
print("tokenizer_fast")
print(tokenizer_fast.tokenize("Hello world!"))
print("tokenizer")
print(tokenizer.tokenize("Hello world!"))
