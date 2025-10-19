from transformer import TransformerLM

def estimate_flops(vocab_size: int,
                   context_length: int,
                   d_model: int,
                   num_layers: int,
                   num_heads: int,
                   d_ff: int):
    return TransformerLM(vocab_size=vocab_size,
                         context_length=context_length,
                         d_model=d_model,
                         num_layers=num_layers,
                         num_heads=num_heads,
                         rope_theta=0.1,
                         d_ff=d_ff).flops(context_length)

if __name__ == '__main__':
    '''
    GPT-2 XL
    vocab_size : 50,257
    context_length : 1,024
    num_layers : 48
    d_model : 1,600
    num_heads : 25
    d_ff : 6,400
    '''
    gpt2_xl_flops = estimate_flops(vocab_size=50257,
                                   context_length=1024,
                                   d_model=1600,
                                   num_layers=48,
                                   num_heads=25,
                                   d_ff=6400)
    formatted = "{:e}".format(gpt2_xl_flops)
    print(f'GPT-2 XL flops: {gpt2_xl_flops} or {formatted}')
