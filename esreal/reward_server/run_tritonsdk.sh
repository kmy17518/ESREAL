docker run -it --rm --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:23.09-py3-sdk bash
# watch -n0.1 nvidia-smi
# # Concurrency: 16, throughput: 26.1396 infer/sec, latency 735367 usec
# # Concurrency: 8, throughput: 21.5933 infer/sec, latency 413884 usec (8 from batch size)
# perf_analyzer -m eva_clip -b 1 --concurrency-range 8 --percentile=95

# # concurrency 8: batch_size 8
# perf_analyzer -m reward_model -b 1 --concurrency-range 8:16:8 --percentile=95 --shape TOKENIZED_PROMPT:333

# cd /workspace/model_repository/open_clip
# # v0: Concurrency: 8, throughput: 98.7008 infer/sec, latency 109297 usec (8 from reward_models)
# # v1: Concurrency: 8, throughput: 58.0995 infer/sec, latency 141851 usec (8 from reward_models)
# # v1 + infer batching : Concurrency: 8, throughput: 73.4279 infer/sec, latency 151822 usec
# # v1 + infer batching + 1-4 input: Concurrency: 8, throughput: 53.3766 infer/sec, latency 195702 usec
# # v1 + infer/ragged batching + 1-4 input: Concurrency: 8, throughput: 55.5433 infer/sec, latency 215294 usec
# # v1 + infer/ragged batching: Concurrency: 8, throughput: 63.5465 infer/sec, latency 160515 usec
# #     + jit: Concurrency: 8, throughput: 71.5265 infer/sec, latency 149482 usec
# perf_analyzer -m open_clip -b 1 --concurrency-range 8 --percentile=95 --shape INPUT:1234567