import torch
import numpy as np
from elr_lrt.model import ELRLRTModel
import time
import psutil
import cpuinfo
from datasets import load_dataset
import threading

# Load small datasets from Hugging Face
def load_gsm8k_data():
    """Load the first 10 samples from GSM8K dataset."""
    dataset = load_dataset("gsm8k", "main", split="train[:10]")
    inputs = [
        torch.tensor(list(sample["question"].encode("utf-8")), dtype=torch.long)
        for sample in dataset
    ]
    targets = [
        torch.tensor(list(sample["answer"].encode("utf-8")), dtype=torch.long)
        for sample in dataset
    ]
    return list(zip(inputs, targets))

def load_boolq_data():
    """Load the first 10 samples from BoolQ dataset."""
    dataset = load_dataset("boolq", split="train[:10]")
    inputs = [
        torch.tensor(list(sample["question"].encode("utf-8")), dtype=torch.long)
        for sample in dataset
    ]
    targets = [
        torch.tensor(list(str(sample["answer"]).encode("utf-8")), dtype=torch.long)
        for sample in dataset
    ]
    return list(zip(inputs, targets))

# Datasets dictionary using GSM8K and BoolQ
datasets = {
    "gsm8k": load_gsm8k_data(),
    "boolq": load_boolq_data(),
}

def sample_memory_usage(process, interval, stop_event, samples):
    """Thread function: sample the process' RSS memory (in MB) periodically."""
    while not stop_event.is_set():
        samples.append(process.memory_info().rss / (1024**2))
        time.sleep(interval)

def run_model_with_metrics(model, input_data):
    """
    Run the forward pass of the model while collecting additional debug metrics:
      - thinking_effectiveness: average ratio showing the relative contribution of the latent (CLR) component.
      - patch_compression_ratio: number of patches divided by the length of input_data.
    """
    # Dynamic Byte Patching
    patches = model.patch_input(input_data)
    patch_compression_ratio = len(patches) / len(input_data) if len(input_data) > 0 else 0

    # Encode patches and process through the encoder
    z = model.encode_patches(patches)
    encoded = model.encoder(z.unsqueeze(0))  # shape: [1, num_patches, d_model]
    h_t = encoded.squeeze(0)  # shape: [num_patches, d_model]

    outputs = []
    effective_thinking_total = 0.0  # Accumulate effective CLR usage per patch
    for t, patch in enumerate(patches):
        # Compute the gating value as in the model
        gate = model.gate_function(h_t[t])
        # If gate is active, compute the relative latent contribution
        if gate > 0:
            latent_component = model.alpha * h_t[t]
            # Ensure token is created on the same device as h_t[t]
            token_embedding = model.byte_embedding(torch.tensor(patch[-1]).to(h_t[t].device))
            latent_norm = torch.norm(latent_component)
            token_norm = torch.norm((1 - model.alpha) * token_embedding)
            effective = latent_norm / (latent_norm + token_norm + 1e-8)
            e_t = latent_component + (1 - model.alpha) * token_embedding
        else:
            effective = 0.0
            e_t = model.byte_embedding(torch.tensor(patch[-1]).to(h_t[t].device))
        effective_thinking_total += effective
        decoded = model.decoder(e_t.unsqueeze(0).unsqueeze(0), encoded)
        output_t = model.output_layer(decoded.squeeze(0))
        outputs.append(output_t)

    outputs = torch.stack(outputs).squeeze(1)
    thinking_effectiveness = effective_thinking_total / len(patches) if len(patches) > 0 else 0

    debug_metrics = {
        "thinking_effectiveness": thinking_effectiveness,
        "patch_compression_ratio": patch_compression_ratio,
    }
    return outputs, debug_metrics

def run_inference_with_metrics(model, input_data, device):
    """
    Run inference while measuring:
      - Latency (s)
      - Peak memory usage (MB)
      - CPU usage (%)
      - Energy proxy (latency * CPU usage)
      - Throughput (tokens processed per second)
      - Additional debug metrics (thinking effectiveness, patch compression ratio)
    """
    process = psutil.Process()
    baseline_mem = process.memory_info().rss / (1024**2)

    # Use torch.cuda memory counters on GPU; sample memory usage on CPU via a thread.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    else:
        samples = []
        stop_event = threading.Event()
        sampling_thread = threading.Thread(target=sample_memory_usage, args=(process, 0.01, stop_event, samples))
        sampling_thread.start()

    start_time = time.time()
    with torch.no_grad():
        if device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                output, debug_metrics = run_model_with_metrics(model, input_data)
        else:
            output, debug_metrics = run_model_with_metrics(model, input_data)
    end_time = time.time()

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        memory_usage = peak_mem
    else:
        stop_event.set()
        sampling_thread.join()
        peak_mem = max(samples) if samples else baseline_mem
        memory_usage = peak_mem - baseline_mem

    latency = end_time - start_time
    cpu_usage = psutil.cpu_percent(interval=latency)
    energy = latency * cpu_usage  # simplified energy proxy
    throughput = len(input_data) / latency if latency > 0 else 0

    debug_metrics["latency"] = latency
    debug_metrics["memory_usage_MB"] = memory_usage
    debug_metrics["cpu_usage_percent"] = cpu_usage
    debug_metrics["energy_proxy"] = energy
    debug_metrics["throughput_tokens_per_sec"] = throughput

    return output, debug_metrics

def measure_efficiency(model, input_data, device):
    """Wrapper to run inference and return all measured metrics."""
    output, metrics = run_inference_with_metrics(model, input_data, device)
    return metrics

def test_elr_lrt():
    """Test the ELR-LRT model with RL fine-tuning and detailed performance metrics."""
    model = ELRLRTModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Running tests with {cpuinfo.get_cpu_info()['brand_raw']}")
    for name, data in datasets.items():
        sample_input, sample_target = data[0]  # Use the first sample from each dataset
        sample_input = sample_input.to(device)
        sample_target = sample_target.to(device)

        print(f"==================== Testing on {name} ====================")
        metrics_before = measure_efficiency(model, sample_input, device)
        print(f"{name} Before RL Fine-tuning:")
        print(f"  Latency: {metrics_before['latency']:.4f}s")
        print(f"  Memory Usage: {metrics_before['memory_usage_MB']:.2f} MB")
        print(f"  CPU Usage: {metrics_before['cpu_usage_percent']:.2f}%")
        print(f"  Energy (proxy): {metrics_before['energy_proxy']:.4f}")
        print(f"  Throughput (tokens/sec): {metrics_before['throughput_tokens_per_sec']:.2f}")
        print(f"  Thinking Effectiveness: {metrics_before['thinking_effectiveness']:.4f}")
        print(f"  Patch Compression Ratio: {metrics_before['patch_compression_ratio']:.4f}")
        print("--------------------------------------------------------------")

        # Perform RL fine-tuning using a reduced number of iterations for testing.
        model.rl_finetune(sample_input, sample_target, num_iterations=2)
        print(f"RL fine-tuning completed for {name}")

        metrics_after = measure_efficiency(model, sample_input, device)
        print(f"{name} After RL Fine-tuning:")
        print(f"  Latency: {metrics_after['latency']:.4f}s")
        print(f"  Memory Usage: {metrics_after['memory_usage_MB']:.2f} MB")
        print(f"  CPU Usage: {metrics_after['cpu_usage_percent']:.2f}%")
        print(f"  Energy (proxy): {metrics_after['energy_proxy']:.4f}")
        print(f"  Throughput (tokens/sec): {metrics_after['throughput_tokens_per_sec']:.2f}")
        print(f"  Thinking Effectiveness: {metrics_after['thinking_effectiveness']:.4f}")
        print(f"  Patch Compression Ratio: {metrics_after['patch_compression_ratio']:.4f}")
        print("==============================================================\n")

if __name__ == "__main__":
    test_elr_lrt()
