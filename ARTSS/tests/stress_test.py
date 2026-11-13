#!/usr/bin/env python3
"""
stress_test.py - Concurrency and load-shedding stress test for ARTSS.
"""
from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path
import requests

URL = "http://127.0.0.1:8000/predict?gradcam=false"

def get_test_image_bytes() -> bytes:
    # Try finding an IDRiD or APTOS image, otherwise fallback to a dummy small JPEG
    for path in Path("ARTSS").rglob("*.jpg"):
        if path.is_file():
            return path.read_bytes()
    for path in Path("ARTSS").rglob("*.png"):
        if path.is_file():
            return path.read_bytes()
    raise RuntimeError("No fundus image found in workspace to run stress test.")

def send_request(idx: int, img_bytes: bytes) -> tuple[int, float]:
    try:
        files = {"file": (f"scan_{idx}.jpg", img_bytes, "image/jpeg")}
        t0 = time.perf_counter()
        resp = requests.post(URL, files=files, timeout=15)
        latency_ms = (time.perf_counter() - t0) * 1000
        return resp.status_code, latency_ms
    except Exception as exc:
        return 500, 0.0

def main():
    print("Preparing stress test payload...")
    img_bytes = get_test_image_bytes()

    total_requests = 30
    concurrency = 15
    print(f"Firing {total_requests} requests with concurrency={concurrency}...")
    print(f"Note: ARTSS max_queue is set to 8. Expect HTTP 503 responses once queue fills up.")

    t_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request, i, img_bytes) for i in range(total_requests)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    total_time = time.perf_counter() - t_start

    status_counts: dict[int, int] = {}
    latencies: list[float] = []
    for status, lat in results:
        status_counts[status] = status_counts.get(status, 0) + 1
        if status == 200:
            latencies.append(lat)

    print("\n" + "=" * 40)
    print("ARTSS STRESS TEST RESULTS")
    print("=" * 40)
    print(f"Total time elapsed: {total_time:.2f} seconds")
    print(f"HTTP Status Breakdown: {status_counts}")
    if latencies:
        print(f"Successful (200 OK) Count: {len(latencies)}")
        print(f"  - Min Latency: {min(latencies):.1f} ms")
        print(f"  - Max Latency: {max(latencies):.1f} ms")
        print(f"  - Avg Latency: {sum(latencies) / len(latencies):.1f} ms")
    if 503 in status_counts:
        print(f"Load-shedding active: {status_counts[503]} requests safely rejected with HTTP 503.")

if __name__ == "__main__":
    main()