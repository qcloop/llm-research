import torch


def compute():
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]  
    )

    query = inputs[1]  # 2nd input token is the query

    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(
            x_i, query
        )  # dot product (transpose not necessary here since they are 1-dim vectors)

    print(f"Attention scores {attn_scores_2}")

    attn_scores_tmp = attn_scores_2 / attn_scores_2.sum()
    print(f"Normalized attention scores {attn_scores_tmp}")
    print(f"Normalized attention scores add up to {attn_scores_tmp.sum()}")
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print(f"Normalized attention scores via softmax {attn_weights_2_naive}")
    print(
        f"Normalized attention scores via softmax add up to {attn_weights_2_naive.sum()}"
    )
    attn_weights_2_built_in = torch.softmax(attn_scores_2, dim=0)
    print(f"Normalized attention scores via torch.softmax {attn_weights_2_built_in}")
    print(
        f"Normalized attention scores via torch.softmax add up to {attn_weights_2_built_in.sum()}"
    )

    # context vector for query x^2
    query = inputs[1] # 2nd input token is the query
    context_vec_2 = torch.zeros(query.shape)
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2_built_in[i]*x_i
    print(
        f"Context vector z^2 for x^2 {context_vec_2}"
    )
    

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)
