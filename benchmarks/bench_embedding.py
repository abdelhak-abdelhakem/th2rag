"""
Benchmark: individual encode() vs encode_batch() on a realistic chunk set.

Simulates 105 chunks from the NIST CSF PDF (1 MB, the reference document
used during development). Run from the project root:

    PYTHONPATH=src uv run python benchmarks/bench_embedding.py

Output example:
    chunks     : 105
    individual : 4.21 s  (105 calls to encode())
    batch      : 0.87 s  (1 call to encode_batch())
    speedup    : 4.8x
"""

import sys
import time

sys.path.insert(0, "src")

from th2rag.rag.embeddings.sentence_transformer import SentenceTransformerEmbedding

# Realistic chunk texts sampled from NIST CSF-style content (105 items).
# Each ~200-400 chars, similar to what HybridTextChunker produces from a PDF.
_BASE_CHUNKS = [
    "The organization identifies and manages cybersecurity risk to systems, assets, data, and capabilities.",
    "Asset management ensures that data, personnel, devices, systems, and facilities are identified and managed consistent with their relative importance to business objectives.",
    "Access control policies restrict user access to only the minimum required to perform their job functions.",
    "Awareness and training programs ensure that all users are informed and adequately trained to perform cybersecurity-related duties.",
    "Data security protections maintain confidentiality, integrity, and availability of information assets.",
    "Information protection processes and procedures are maintained to manage protection of systems and assets.",
    "Protective technology is managed to ensure the security and resilience of systems and assets.",
    "Anomalies and events are detected in a timely manner and their potential impact is understood.",
    "Security continuous monitoring provides the organization with cybersecurity event visibility.",
    "Detection processes are tested to ensure awareness of anomalous events.",
    "Response planning processes and procedures are executed and maintained to enable timely response.",
    "Communications during incident response ensure that all personnel are aware of roles and responsibilities.",
    "Analysis of detected cybersecurity events ensures adequate response and supports recovery activities.",
    "Mitigation activities are performed to prevent expansion and resolve detected cybersecurity incidents.",
    "Improvements to the organization's security posture are implemented based on lessons learned.",
    "Recovery planning processes and procedures are executed and maintained to ensure timely restoration.",
    "Improvements to recovery planning and processes are incorporated based on lessons learned.",
    "Backups of information are conducted, maintained, and tested periodically.",
    "Recovery strategies incorporate public relations and reputation repair.",
    "The organization coordinates restoration activities with internal and external parties.",
    "Cybersecurity risk is identified, assessed, and managed as part of organizational risk management.",
    "The organization establishes a risk management strategy, including risk tolerance and assumptions.",
    "The organization understands the cybersecurity risk to organizational operations and assets.",
    "The organization identifies and prioritizes their critical infrastructure and the functions they perform.",
    "Suppliers and third-party partners of information systems are identified and monitored.",
    "Supply chain risk management processes are established, implemented, assessed, and improved.",
    "Contracts with suppliers and third-party partners include cybersecurity requirements.",
    "Supplier audits and assessments are conducted to verify security compliance.",
    "The organization establishes and maintains a cyber supply chain risk management program.",
    "Third-party service providers are required to implement appropriate safeguards.",
]

N_CHUNKS = 105


def make_chunks(n: int) -> list[str]:
    chunks = []
    for i in range(n):
        base = _BASE_CHUNKS[i % len(_BASE_CHUNKS)]
        # vary slightly so they're not identical
        chunks.append(f"[Chunk {i + 1}] {base}")
    return chunks


def main():
    print("Loading embedding model (gtr-t5-large)...")
    embedder = SentenceTransformerEmbedding()
    dim = embedder.get_embedding_dimension()
    print(f"Model ready. Embedding dimension: {dim}\n")

    chunks = make_chunks(N_CHUNKS)
    print(f"chunks     : {len(chunks)}")

    # --- individual encode() ---
    t0 = time.perf_counter()
    individual_vectors = [embedder.encode(t) for t in chunks]
    individual_time = time.perf_counter() - t0
    print(f"individual : {individual_time:.3f} s  ({len(chunks)} calls to encode())")

    # --- encode_batch() ---
    t0 = time.perf_counter()
    batch_vectors = embedder.encode_batch(chunks)
    batch_time = time.perf_counter() - t0
    print(f"batch      : {batch_time:.3f} s  (1 call to encode_batch())")

    speedup = individual_time / batch_time if batch_time > 0 else float("inf")
    print(f"speedup    : {speedup:.1f}x")

    # Sanity check: vectors match
    mismatches = 0
    for i, (iv, bv) in enumerate(zip(individual_vectors, batch_vectors)):
        if any(abs(a - b) > 1e-4 for a, b in zip(iv, bv)):
            mismatches += 1
    print(f"\nvector match: {'OK' if mismatches == 0 else f'{mismatches} mismatches'}")


if __name__ == "__main__":
    main()
