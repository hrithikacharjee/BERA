class BERAConfig:
    """
    Base configuration for the BERA transformer encoder.
    Update values later if needed.
    """
    vocab_size: int = 32000
    max_position_embeddings: int = 512
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_attention_heads: int = 12
    num_hidden_layers: int = 22   # our custom 22-layer encoder
    dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    model_type: str = "bera-transformer"
