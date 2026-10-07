# Research log: what exists, what it means for tinyforge

Compiled 2026-10-07 from web search summaries and a few fetched pages. **Nothing here has been reproduced by us.** Every
number is a claim made by its source; papers were not read in full. Treat "Verify" items as open work, not facts. IDs and
dates come from the search results as returned.

Legend: **Act** = changes what we build next. **Watch** = relevant later. **Skip** = decided against (with reason).

## 1. Training a model that does not fit on a small GPU (our core goal)

| Finding | Claim (source) | Effect on tinyforge | Status |
|---|---|---|---|
| **Soup CLI exact layer streaming**: frozen NF4 base in RAM/NVMe, one decoder layer in VRAM at a time, only LoRA resident | Llama-3.1-8B on RTX 3050 Laptop 4 GB: 3.32 GB peak, 119.6 tok/s; Qwen2.5-3B 1.76 GB, 264 tok/s; bit-exact vs resident run; Apache-2.0, ~8.3k stars, single maintainer. BETA; the project itself says 4 GB numbers predate a correctness fix and re-measurement is pending (issue #361); previous silent bugs (adapter no-op load, wrong NF4 gradients) | The exact feature we planned to build. **Act:** benchmark it on this laptop in an isolated venv; then integrate as a backend behind JobSpec or build our own | Verify |
| **Horizon-LM** (arXiv 2602.04816): host RAM is the parameter store, GPU is a transient cache; layer templates; 3 CUDA streams with double buffering; block-wise recompute | 12.2x vs ZeRO-3 CPU offload on A100 (14B); built for huge hosts (up to 1.5 TB RAM), may not scale down to a laptop (PCIe-bound) | Design reference for our own streaming fallback: pack each layer into one contiguous host buffer, pinned double buffers, explicit recompute | Act (fallback) |
| **ZeRO-Infinity / FlexGen / LoHan** | CPU+NVMe offload engines; FlexGen 4-bit weights+KV for single-GPU inference; LoHan fine-tunes up to 100B on a consumer GPU with NVMe offload | Confirms the ladder shape and that NVMe throughput is the new limit; planner must know disk speed | Watch |
| **RoundPipe** (arXiv 2604.27085) | Needs several consumer GPUs (8x RTX 4090 tested), 1.48-2.16x over baselines, open-source | Multi-GPU only, not our single-laptop case | Watch |
| **MegaTrain** (arXiv 2604.05091) | Streams layers; 120B on one H200 with 1.5 TB host RAM | Same idea at datacentre scale | Watch |

## 2. Memory techniques in the same family as what we built

- **Chunked / fused cross-entropy.** Liger Kernel FLCE (Triton) and Apple Cut Cross-Entropy (ICLR 2025, arXiv 2411.09009)
  avoid materialising logits; Apple claims 24 GB -> 1 MB loss memory on Gemma 2 2B. Unsloth: chunk size adapts to free VRAM.
  Our `memory.py` is the simple, exact version (tested equal to HF loss and gradients). **Act:** measure it (running);
  adaptive chunk size is a cheap upgrade; Liger/CCE kernels are the next step if the simple one is not enough.
- **Smarter gradient checkpointing and async activation offload to RAM** (Unsloth: ~30% less memory for ~2% time, claimed;
  torchtune has activation offloading). **Act:** add as a ladder rung and measure on PCIe 4 GB laptop.
- **Padding-free packing with varlen attention** (HF DataCollatorWithFlattening / TRL padding_free; ~1.9x throughput
  claimed). Silent-bug risk if packing leaks across examples. **Act:** after the ladder; add a regression test for
  cross-example attention.
- **8-bit / paged optimizers (bitsandbytes), GaLore, LISA, BAdam.** Only matter for full-parameter training, since LoRA
  optimizer state is tiny. LISA claims 10-35% MT-Bench over LoRA at LoRA-like memory. GaLore has periodic SVD stalls and
  brittle hyperparameters. **Watch:** a "full fine-tune of small models" mode later.
- **MeZO** (forward-only, inference-level memory; claims comparable to backprop on several tasks, slow convergence).
  Last-resort rung. **Skip for now.**
- **Layer-wise local-loss training.** Hurts quality. **Skip** (already decided in OBJECTIVES K).

## 3. Adapter and recipe knowledge that can raise quality for free

- **"LoRA Without Regret" (Thinking Machines) + related:** LoRA matches full fine-tuning when applied to all linear layers
  (attention-only underperforms; we already use `all-linear`), best LoRA LR ~10x the full-FT LR, optimal LR roughly
  rank-independent under 1/r scaling, tune LR first. A 2026 paper gives an LR formula by hidden size (less than 0.5% regret
  claimed). **Act:** planner defaults for LR/rank with a short LR probe; keep it measured, not assumed.
- **Variants available through PEFT as flags:** DoRA, rsLoRA, PiSSA, LoftQ. QDoRA claims to beat QLoRA. **Act (cheap):**
  expose as options, run one controlled comparison on our SQL task before recommending any.
- **Batch size confounds LoRA comparisons** (arXiv 2602.09492): fix it when we compare methods.

## 4. Beyond supervised fine-tuning

- **GRPO / RLVR on small models.** Needs no value network; Unsloth claims GRPO in ~5 GB. The reward must be verifiable.
  **Our execution-accuracy harness is exactly such a reward for text-to-SQL.** **Act (high leverage, later):** a GRPO
  recipe for the SQL task with execution reward, measured against the SFT baseline. Memory for generations (group size)
  is the cost; arXiv 2609.39321 studies GRPO dynamics at 1.5B-7B.
- **Preference methods** (DPO, ORPO, KTO) via TRL. Watch.
- **Distillation / synthetic data:** a 7B tuned on ~2k distilled examples can beat an untuned 70B on a narrow task (claim).
  Teacher traces with explanations beat answers only (Orca). Watch; licence of teacher outputs matters.
- **Text-to-SQL targets:** SLM-SQL (arXiv 2507.22478) reports 0.5B and 1.5B models far above our 43.9% execution accuracy,
  but on BIRD/Spider, a different and harder-setup benchmark. **Verify the exact figures before citing**: earlier notes in
  OBJECTIVES (56.87% / 67.08% BIRD EX) disagree with a search summary (73.5% / 79.06%); do not quote either yet.

## 5. Data

- Quality beats quantity (LIMA: ~1,000 curated examples); selection methods (LESS, IFD) and dedup. Our pipeline already
  dedups and flags PII. **Act:** surface data-quality scores; add the silent-bug checks below.
- **Silent data bugs to audit in `ft_data.py`:** hand-built chat templates instead of `apply_chat_template`, wrong EOS
  handling, packing without proper masking. **Act (next, no GPU needed).**
- **Privacy:** LoRA memorizes less than full fine-tuning (claimed); DP-LoRA / DP-FedLoRA exist. Watch (enterprise).

## 6. Quantization and serving (the "deploy it" half)

- Claimed quality vs BF16: FP8 ~99%, AWQ INT4 ~94-96%, GPTQ INT4 ~93-95%, GGUF Q5_K_M ~95-97%, Q2_K ~80-85%. Our measured
  step 3: Q8_0 ok, Q4_K_M -5 points exact match. **Act (small):** imatrix calibration for Q4_K_M; offer AWQ/GPTQ export.
  FP8/NVFP4 need Hopper/Blackwell: not our laptop, Watch.
- **Serving:** vLLM and SGLang converge on prefix caching, speculative decoding, multi-LoRA, prefill/decode disaggregation.
  llama.cpp remains the laptop default. Benchmark under concurrent load (1/8/32 clients) per OBJECTIVES Q6. Watch.

## 7. Evaluation integrity (our stated moat)

- Contamination inflates scores; LLM judges show self-preference bias; single-run comparisons hide variance. We already
  report Wilson CIs and a self-match sanity check for the SQL harness. **Act:** add contamination check (train vs eval
  overlap), paired tests for base-vs-tuned, and keep rule "no claim without raw JSON". lm-evaluation-harness for general
  regression checks.

## 8. Hardware trends that change the planner

- Unified-memory machines (Apple M-series up to ~800 GB/s, AMD Strix Halo ~256 GB/s with 128 GB, NVIDIA DGX Spark 128 GB)
  make "offload" mean something different: no PCIe copy, but lower bandwidth than HBM. The planner must model memory tiers
  and bandwidth, not just VRAM. **Act (design):** represent hardware as tiers (VRAM, shared RAM, NVMe) with measured
  bandwidth. Non-NVIDIA backends (MLX, ROCm, Vulkan) are Watch until the NVIDIA ladder is proven.

## 9. Decentralised training (long-term "anyone, anywhere")

- DiLoCo (communicate every 100-500 steps, ~500x less bandwidth), Hivemind, Petals, Pluralis; a 7.5B OLMo-style model was
  trained across the internet for 3+ weeks (claim). Fits "many small GPUs" but is research-grade. **Watch.**

## 10. Regulation (affects model cards and enterprise adoption)

- EU AI Act GPAI obligations applied from 2 Aug 2025; a fine-tuner becomes a provider only for a significant modification
  (indicative threshold: more than one third of the original training compute, ~3.3e22 FLOP). Almost every LoRA run is far
  below that, but downstream users still need documentation. Our generated model card is a start. **Act (small):** add a
  training-data summary section and compute used (FLOP estimate). Not legal advice.

## 11. Competitive landscape

- **Unsloth Studio** (launched 2026-03-17): local no-code UI, data recipes (PDF/CSV/DOCX to datasets), model arena, claims
  2x faster / 70% less VRAM. **LLaMA-Factory:** 100+ models, web UI, many methods. **Axolotl:** flexible YAML, multi-GPU.
  **torchtune:** PyTorch-native, activation offload. We should not clone them. Differentiators worth keeping: hardware-aware
  planner with honest ETA/RAM, eval gates that refuse to ship a model that is not better than base, fleet/ops layer
  (agent, Helm, monitoring), and published negative results.

## Decision backlog from this research (ordered)

1. Finish measuring chunked CE on a 1.5B model (running). 2. Benchmark Soup on this laptop. 3. Audit chat-template/EOS/mask
handling. 4. Activation offload + adaptive chunk rungs. 5. Integrate or build layer streaming. 6. Planner: tiers, LR/rank
defaults, ETA/RAM. 7. GRPO with execution reward on SQL. 8. imatrix/AWQ export; contamination + paired tests.

## Sources
Soup: https://github.com/MakazhanAlpamys/Soup , https://themenonlab.blog/blog/soup-fine-tune-8b-llm-4gb-laptop-gpu ;
Horizon-LM https://arxiv.org/abs/2602.04816 ; LoHan https://arxiv.org/abs/2403.06504 ; MegaTrain https://arxiv.org/abs/2604.05091 ;
RoundPipe https://arxiv.org/abs/2604.27085 ; ZeRO-Infinity https://arxiv.org/abs/2104.07857 ; FlexGen https://arxiv.org/abs/2303.06865 ;
Liger Kernel https://github.com/linkedin/Liger-Kernel ; Cut Cross-Entropy https://arxiv.org/abs/2411.09009 ;
Unsloth https://unsloth.ai/docs/blog/500k-context-length-fine-tuning ; LISA https://arxiv.org/abs/2403.17919 ;
GaLore https://proceedings.mlr.press/v235/zhao24s.html ; MeZO https://arxiv.org/abs/2305.17333 ;
LoRA Without Regret https://thinkingmachines.ai/blog/lora/ ; LR matters https://arxiv.org/abs/2602.04998 ;
Batch size bias https://arxiv.org/abs/2602.09492 ; LoRA variants study https://arxiv.org/abs/2601.22708 ;
GRPO small models https://arxiv.org/abs/2609.39321 ; SLM-SQL https://arxiv.org/abs/2507.22478 ;
Decentralised survey https://arxiv.org/abs/2503.11023 ; data selection survey https://arxiv.org/abs/2402.05123 ;
DP-FedLoRA https://arxiv.org/abs/2509.09097 ; LoRA DP by design https://arxiv.org/abs/2409.17538 ;
EU AI Act GPAI roles https://cms.law/en/swe/legal-updates/general-purpose-ai-models-obligations-and-roles-for-providers-and-downstream-modifiers-under-the-eu-ai-act ;
Fine-tuning pitfalls https://machinelearningmastery.com/5-problems-encountered-fine-tuning-llms-with-solutions/ ;
Unsloth Studio https://rits.shanghai.nyu.edu/ai/unsloth-studio-open-source-no-code-ui-for-local-llm-training-and-inference/ ;
Quantization comparison https://computingforgeeks.com/gguf-vs-awq-vs-gptq/ .
