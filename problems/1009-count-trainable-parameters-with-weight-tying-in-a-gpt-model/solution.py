def count_gpt_parameters(cfg: dict) -> list:
    """Return [total_params, tied_params] for the GPT-style model.

    cfg keys:
        vocab_size, context_length, emb_dim, n_heads, n_layers,
        drop_rate, qkv_bias
    """
    vocab_size = cfg["vocab_size"]
    context_length = cfg["context_length"]
    emb_dim = cfg["emb_dim"]
    n_heads = cfg["n_heads"]
    n_layers = cfg["n_layers"]
    drop_rate = cfg["drop_rate"]
    qkv_bias = cfg["qkv_bias"]

    emb_dim_sq = emb_dim ** 2

    token_e = vocab_size * emb_dim
    positional_e = context_length * emb_dim
    transformer_e = (13 if qkv_bias else 10) * emb_dim + 12 * emb_dim_sq

    total = token_e + positional_e + n_layers * transformer_e + 2 * emb_dim

    return [total + token_e, total]    