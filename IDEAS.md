# 1) Hyperbolic (non-Euclidean) embeddings for hierarchies

Hyperbolic spaces model tree-like structure with low distortion, so they’re strong for label/topic/doc taxonomies. Recent work adds simple regularizers you can drop into standard encoders:

* **Poincaré/Lorentz hyperbolic embeddings** for hierarchical structure (foundational). ([arXiv][1])
* **HypStructure (NeurIPS 2024)**: a hyperbolic tree-based representation loss + centering loss that you add alongside your task loss to make representations hierarchy-aware. Code is public. ([NeurIPS Proceedings][2])

**When to use:** you have an explicit tree/DAG (taxonomy, label hierarchy, site map) and want global + per-node geometry that respects parent→child relations.

# 2) Hierarchical contrastive learning (local↔global)

Contrastive objectives at multiple granularities (segments/sentences vs full sequence/doc) give you a clean “dual-layer” setup:

* **HiCL (EMNLP Findings 2023):** splits sequences into segments, learns local (segment) and global (sequence) representations with joint contrastive objectives—faster and more robust than sequence-only CSE. ([arXiv][3])
* **HiLight (2024):** “hierarchy-aware light global model” + **hierarchical local contrastive** learning; addresses common collapse issues in recursive regularization. ([arXiv][4])

**When to use:** you want sentence/paragraph vectors that compose into strong doc vectors, and you don’t have (or don’t fully trust) a labeled hierarchy.

# 3) Multi-vector / late-interaction retrieval (token-level + doc-level)

For retrieval over hierarchical corpora, multi-vector models give you a natural dual layer: token/phrase vectors for precise matching plus a pooled doc vector:

* **ColBERT-style** late interaction remains state of the art for recall, with new work to **shrink storage**: token-pooling/clustering and **ConstBERT**/**constant-space** variants fix the footprint while keeping accuracy high. ([arXiv][5])

**When to use:** hierarchical sites/wikis/manuals where you need both fine-grain passage hits (local) and section/document ranking (global).

# 4) Label-aware and graph-aware encoders

If you have a label tree, making the model *see* it helps both layers:

* **Dual Prompt Tuning (ACL Findings 2024):** contrastive learning **per hierarchy level** with prompts that expose positives/negatives among sibling labels. ([ACL Anthology][6])
* **Hybrid hierarchical text classification** with the decoder **pre-populated by all label embeddings** to expose global structure during learning. ([SpringerLink][7])
* **Graph attention / GAT variants** for label graphs combine structural and semantic signals (e.g., HE-HMTC lineage). ([ScienceDirect][8])

# 5) Topic/modeling under hyperbolic geometry

For hierarchical topic trees, pairing contrastive signals with hyperbolic projection preserves tree structure in the embedding space. ([ACL Anthology][9])

---

## A practical “dual-layer” recipe (works well today)

1. **Encoder:** start with a strong text encoder (e.g., RoBERTa/E5-family) that outputs token embeddings.
2. **Local layer:** produce **sentence/segment vectors** via mean-pool or [CLS]; train with **local contrastive** (segment↔segment positives within a doc; hard negatives across docs). (HiCL pattern.) ([arXiv][3])
3. **Global layer:** aggregate segments with **attention pooling** (or small transformer) into a **doc/section vector**; add **global contrastive** (doc↔doc) and, if you have a taxonomy, add **hyperbolic regularization** (HypStructure) on the global vectors. ([NeurIPS Proceedings][2])
4. **Optional retrieval head:** keep **multi-vector outputs** (a small set of learned token/phrase centroids per doc) for late interaction; apply **token-pooling**/constant-space tricks to cap storage. ([arXiv][5])
5. **Hierarchy losses:** add **sibling separation** (contrast siblings at each level) and **ancestor consistency** (parent closer than non-ancestors); DPT-style prompts can generate level-wise pairs without heavy labeling. ([ACL Anthology][6])
6. **Geometry:** if your hierarchy is real (taxonomy), **project global vectors to hyperbolic space** for distance computations and use Möbius ops during training; keep **Euclidean locals** if you want simpler token/segment ops. ([arXiv][1])

---

## How to choose

* **You have a real tree/DAG** (ontology, product catalog): add **HypStructure** on the **global** embeddings; consider **Lorentz/Poincaré** distances at inference. ([NeurIPS Proceedings][2])
* **You need high-recall retrieval** across long manuals/wikis: use a **multi-vector/late-interaction** head + **token-pooling** to control index size. ([arXiv][5])
* **You lack clean labels but have long docs:** train **HiCL-style** (local+global contrastive) and optionally discover topic trees in **hyperbolic** space post-hoc. ([arXiv][3])
* **You’re doing hierarchical classification:** use **Dual Prompt Tuning** or **hybrid encoder-decoder with label embeddings** to expose level-wise structure. ([ACL Anthology][6])


[1]: https://arxiv.org/abs/1705.08039?utm_source=chatgpt.com "Poincaré Embeddings for Learning Hierarchical Representations"
[2]: https://proceedings.neurips.cc/paper_files/paper/2024/hash/a5d2da376bab7624b3caeb9f78fcaa2f-Abstract-Conference.html?utm_source=chatgpt.com "Learning Structured Representations with Hyperbolic Embeddings"
[3]: https://arxiv.org/abs/2310.09720?utm_source=chatgpt.com "HiCL: Hierarchical Contrastive Learning of Unsupervised Sentence Embeddings"
[4]: https://arxiv.org/html/2408.05786?utm_source=chatgpt.com "HiLight: A Hierarchy-aware Light Global Model with Hierarchical Local ..."
[5]: https://arxiv.org/abs/2409.14683?utm_source=chatgpt.com "Reducing the Footprint of Multi-Vector Retrieval with Minimal ..."
[6]: https://aclanthology.org/2024.findings-acl.723.pdf?utm_source=chatgpt.com "Dual Prompt Tuning based Contrastive Learning for Hierarchical Text ..."
[7]: https://link.springer.com/content/pdf/10.1007/978-3-031-88708-6_26.pdf?pdf=inline+link&utm_source=chatgpt.com "Decoding the Hierarchy: A Hybrid Approach to Hierarchical ... - Springer"
[8]: https://www.sciencedirect.com/science/article/pii/S0957417423027690?utm_source=chatgpt.com "Improve label embedding quality through global sensitive GAT for ..."
[9]: https://aclanthology.org/2024.lrec-main.712.pdf?utm_source=chatgpt.com "Hierarchical Topic Modeling via Contrastive Learning and Hyperbolic ..."
