from typing import Optional
import math
import random

import torch
import triton

from flash_attn_interface import flash_attn_varlen_func


def get_attn_bias(s_q, s_k, causal, window, topk_index: Optional[torch.Tensor] = None):
    attn_bias = torch.zeros(s_q, s_k, dtype=torch.float32)
    if causal:
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if window > 0:
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q - window)
        attn_bias.masked_fill_(temp_mask, float("-inf"))
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q + window - 1)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if topk_index is not None:
        attn_mask = torch.zeros((s_q, s_k), dtype=torch.bool)
        mask = (topk_index >= 0) & (topk_index < s_k)
        rows = torch.arange(s_q).unsqueeze(1).expand_as(topk_index)
        attn_mask[rows[mask], topk_index[mask]] = True
        attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
    return attn_bias


def calc_diff(x, y):
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    denominator = (x * x + y * y).sum()
    if denominator < 1e-12:
        return 0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim.item(), RMSE


def assert_close(x, y, name=""):
    x, y = x.double(), y.double()
    diff = calc_diff(x, y)
    amax = (x - y).abs().max()
    print(f"{name} diff: {diff}, amax: {amax}")
    # assert diff[0] < 1e-2


def timer(func, name, total_attn_compute, total_q, total_k):
    t = triton.testing.do_bench(func)
    FLOPS = h * (d + dv) * total_attn_compute * 2 * (3.5 if has_bwd else 1)
    BYTES = (total_q * h * d + total_k * h_k * (d + dv)) * (torch.finfo(dtype).bits / 8)
    print(f"{t} ms, {FLOPS / 10 ** 9 / t} TFLOP/s, {BYTES / 10 ** 6 / t} GB/s, name: {name}")
    return t


def scaled_dot_product_attention(query, key, value, attn_bias, h, h_k, return_max_logits) -> torch.Tensor:
    key = key.repeat_interleave(h // h_k, dim=1)
    value = value.repeat_interleave(h // h_k, dim=1)
    attn_weight = query @ key.transpose(-2, -1) / math.sqrt(query.size(-1))
    attn_weight += attn_bias
    if return_max_logits:
        if attn_weight.shape[-1] > 0:
            max_logits = attn_weight.max(dim=-1, keepdim=True)[0]
        else:
            b, q, _ = attn_weight.shape
            max_logits = torch.full((b, q, 1), float('-inf'), dtype=attn_weight.dtype)
    else:
        max_logits = None
    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32)
    return attn_weight.to(query.dtype) @ value, max_logits


def test_flash_attention(b, mean_sq, mean_sk, varlen, h, h_k, d, dv, causal, topk, return_max_logits, window_size):
    print(f"{b=}, {mean_sq=}, {mean_sk=}, {varlen=}, {h=}, {h_k=}, {d=}, {dv=}, {causal=}, {topk=}, {return_max_logits=}, {window_size=}")

    torch.manual_seed(0)
    random.seed(0)

    seqlens_q = torch.full((b,), mean_sq, dtype=torch.int32)
    seqlens_k = torch.full((b,), mean_sk, dtype=torch.int32)
    if varlen:
        for i in range(b):
            seqlens_q[i] = max(random.normalvariate(mean_sq, mean_sq / 2), 1)
        for i in range(b):
            seqlens_k[i] = max(random.normalvariate(mean_sk, mean_sk / 2), seqlens_q[i].item() - 50)
    seqlens_q[1] = 1  # Test the corner case that seqlen_q = 1 & seqlen_k = 0
    seqlens_k[1] = 0
    cu_seqlens_q = torch.cumsum(torch.nn.functional.pad(seqlens_q, (1, 0)), 0, dtype=torch.int32)
    cu_seqlens_k = torch.cumsum(torch.nn.functional.pad(seqlens_k, (1, 0)), 0, dtype=torch.int32)
    total_q = seqlens_q.sum().item()
    total_k = seqlens_k.sum().item()
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    total_attn_compute = sum([(get_attn_bias(seqlens_q[i].item(), seqlens_k[i].item(), causal, window) == 0).sum().item() for i in range(b)])

    q = torch.randn(total_q, h, d)
    k = torch.randn(total_k, h_k, d)
    v = torch.randn(total_k, h_k, dv)
    grad_out = torch.randn(total_q, h, dv)
    topk_index = torch.randint(0, total_k, size=(total_q, topk)) if topk > 0 else None

    q1 = q.clone().requires_grad_()
    k1 = k.clone().requires_grad_()
    v1 = v.clone().requires_grad_()

    q2 = q.clone().requires_grad_()
    k2 = k.clone().requires_grad_()
    v2 = v.clone().requires_grad_()


    def flash_attn(provider="FA3"):
        q1.grad = k1.grad = v1.grad = None
        kwargs = {}
        kwargs["deterministic"] = deterministic
        if causal:
            kwargs["causal"] = causal
        if window != 0:
            kwargs["window_size"] = window_size
        kwargs["topk_index"] = topk_index
        kwargs["return_max_logits"] = return_max_logits
        assert provider == "FA3 varlen"
        return flash_attn_varlen_func(q1, k1, v1, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)

    def torch_attn():
        q2.grad = k2.grad = v2.grad = None
        out = []
        if return_max_logits:
            max_logits = []
        for i in range(b):
            OUT_TENSORS = scaled_dot_product_attention(
                q2[cu_seqlens_q[i].item(): cu_seqlens_q[i + 1].item()].float().transpose(-3, -2),
                k2[cu_seqlens_k[i].item(): cu_seqlens_k[i + 1].item()].float().transpose(-3, -2),
                v2[cu_seqlens_k[i].item(): cu_seqlens_k[i + 1].item()].float().transpose(-3, -2),
                attn_bias=get_attn_bias(
                    seqlens_q[i].item(), seqlens_k[i].item(),
                    causal, window,
                    None if topk_index is None else (topk_index[cu_seqlens_q[i].item(): cu_seqlens_q[i + 1].item(), :] - cu_seqlens_k[i])),
                h=h, h_k=h_k,
                return_max_logits=return_max_logits,
            )
            if return_max_logits:
                OUT, MAX_LOGITS = OUT_TENSORS
            else:
                OUT, _ = OUT_TENSORS
            out.append(OUT.transpose(-3, -2))
            if return_max_logits:
                max_logits.append(MAX_LOGITS.transpose(-2, -1))
        out = torch.cat(out)
        if return_max_logits:
            max_logits = torch.cat(max_logits, dim=-1)
            return out, max_logits
        return (out,)

    def fn(provider="FA3"):
        flash_attn(provider)[0].backward(grad_out) if has_bwd else flash_attn(provider)[0]

    # with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    #     fn("FA3 varlen")
    # print(prof.key_averages().table(sort_by="cuda_time_total", max_name_column_width=120))

    full_output_flash_attn = flash_attn("FA3 varlen")
    full_output_torch_attn = torch_attn()

    out_flash_attn = full_output_flash_attn[0]
    out_torch_attn = full_output_torch_attn[0]

    # print(seqlens_q, seqlens_k)
    # print(out_flash_attn.isnan().nonzero(as_tuple=False))
    # print(out_flash_attn[3476])

    assert_close(out_flash_attn, torch.nan_to_num(out_torch_attn, nan=0.0), "out")

    if return_max_logits:
        out_max_logits = full_output_flash_attn[2]
        out_max_logits_ref = full_output_torch_attn[1]
        # TODO: support inf compare in max_logits
        assert_close(out_max_logits, torch.nan_to_num(out_max_logits_ref, nan=0.0), "max_logits")

    if has_bwd:
        out_flash_attn.backward(grad_out)
        out_torch_attn.backward(grad_out)
        assert_close(q1.grad, q2.grad, "dq")
        assert_close(k1.grad, k2.grad, "dk")
        assert_close(v1.grad, v2.grad, "dv")

    timer(lambda: fn("FA3 varlen"), "FA3 varlen", total_attn_compute, total_q, total_k)
    # timer(lambda: fn("FA3 varlen"), "FA3 varlen", total_attn_compute, total_q, total_k)


if __name__ == "__main__":
    b = 2
    mean_sq = 4096
    mean_sk = 4096
    h = 128
    h_k = 128
    d = 192
    dv = 128
    window = 0
    has_bwd = False
    deterministic = False
    return_max_logits = False
    dtype = torch.bfloat16
    default_dtype = torch.bfloat16
    torch.set_default_dtype(default_dtype)
    device = torch.device("cuda:4")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)

    for causal in [False, True]:
        for varlen in [False, True]:
            for topk in [0, 512]:
                if window > 0:
                    window_size = (window - 1, 0) if causal else (window - 1, window - 1)
                else:
                    window_size = (-1, -1)
                test_flash_attention(b, mean_sq, mean_sk, varlen, h, h_k, d, dv, causal, topk, return_max_logits, window_size)
