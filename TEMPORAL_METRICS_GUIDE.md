# Temporal Retrieval Metrics Guide

This guide explains the metrics computed during temporal embedding experiments.

## Within-Group Metrics

These metrics evaluate how well embeddings preserve temporal relationships among facts in the same group.

### `within_group_temporal_order_correlation`
- **Range**: -1.0 to 1.0
- **Interpretation**:
  - **Positive values** = GOOD: Facts are retrieved in temporal order (earlier facts rank higher when querying earlier facts)
    - +0.7 to +1.0 = Strong temporal ordering preserved
    - +0.3 to +0.7 = Moderate temporal signal
    - +0.1 to +0.3 = Weak temporal signal
  - **Near 0** (-0.1 to +0.1) = Embeddings have no temporal signal (typical for paraphrased facts)
  - **Negative values** = BAD: Temporal order is inverted (earlier facts are less similar to earlier facts)
- **Measured by**: Spearman or Kendall rank correlation between temporal position and retrieval ranking
- **How it works**: For each fact, we rank other facts by similarity and compare that ranking to their chronological order. Positive correlation means chronological order is preserved.

### `within_group_nearest_fact_mrr`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Mean Reciprocal Rank for finding the temporally nearest fact
  - 1.0 = Nearest fact always ranked #1
  - 0.5 = Nearest fact typically ranked #2
  - 0.33 = Nearest fact typically ranked #3
  - 0.17 = Nearest fact typically ranked #6
- **Measures**: How quickly the model finds facts that are temporally adjacent (smallest time difference)

## Cross-Group Metrics

These metrics evaluate how well embeddings discriminate between different fact groups.

### `cross_group_mrr`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Mean Reciprocal Rank for finding the first same-group fact when retrieving from the entire corpus
  - 1.0 = Same-group facts always rank highest
  - 0.5 = First same-group fact typically at rank #2
  - Lower values = Other groups' facts interfere

### `cross_group_recall@k`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: What fraction of same-group facts appear in top-k results
  - At k=5: What % of the 6 other group members appear in top 5?
  - Example: 0.67 means 4 out of 6 same-group facts appear in top 5

### `cross_group_precision@k`
- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: What fraction of top-k results are from the same group
  - At k=5: Out of top 5 results, how many are from the correct group?
  - Example: 0.8 means 4 out of 5 top results are from the same group

### `cross_group_purity@k`
- **Range**: 0.0 to 1.0 (higher is better)
- **Same as precision@k**: Measures group homogeneity in top-k results
- **Higher is better**: Indicates strong group discrimination

## Temporal Drift Metrics

These metrics analyze how semantic similarity changes over time.

### `temporal_drift_correlation`
- **Range**: -1.0 to 1.0
- **Interpretation**:
  - **Negative values**: Similarity decreases as temporal distance increases (expected for time-varying facts)
    - -0.3 to -0.7 = Strong temporal drift captured
    - -0.1 to -0.3 = Moderate drift
  - **Near 0**: No temporal drift detected (typical for paraphrased facts with same semantic content)
  - **Positive values**: Unusual - similarity increases with time (may indicate stylistic patterns)
- **Measured by**: Spearman correlation between temporal distance (seconds) and semantic similarity

### `avg_similarity_time_bin_N`
- **Range**: 0.0 to 1.0
- **Interpretation**: Average similarity for fact pairs in different time ranges
  - Lower bin numbers = Facts closer in time
  - Higher bin numbers = Facts farther apart in time
  - Expected pattern: Similarity should decrease across bins (if temporal drift exists)

### `avg_within_group_similarity`
- **Range**: 0.0 to 1.0
- **Interpretation**: Overall average similarity between facts in the same group
  - Baseline for comparison with cross-group similarity
  - High values (0.8+) indicate facts are semantically very similar

## Example Results Interpretation

### Scenario 1: Paraphrased Facts (Your Current Dataset)
```
Model: hierarchical_concat
├─ within_group_temporal_order_correlation: 0.06
│  → Near zero: No temporal ordering (expected for paraphrases)
├─ within_group_nearest_fact_mrr: 0.36
│  → Temporally nearest fact typically ranks around #3
├─ cross_group_mrr: 0.85
│  → Excellent: Same-group facts rank very high
├─ cross_group_purity@5: 0.72
│  → Good: 72% of top-5 results are from correct group
└─ temporal_drift_correlation: -0.08
   → Near zero: Minimal drift (paraphrases have consistent semantics)
```

**Interpretation**: This model is excellent at discriminating between different fact groups (high cross_group metrics) but doesn't capture temporal ordering within groups (temporal_order_correlation ≈ 0). This makes sense because your facts are paraphrases of the same event—they're semantically identical regardless of timestamp.

### Scenario 2: Evolving Content (Hypothetical)
```
Model: hierarchical_concat
├─ within_group_temporal_order_correlation: 0.45
│  → Moderate temporal signal preserved
├─ within_group_nearest_fact_mrr: 0.68
│  → Temporally nearest fact typically ranks #1-2
├─ cross_group_mrr: 0.92
│  → Excellent group discrimination
├─ cross_group_purity@5: 0.88
│  → Strong group purity
└─ temporal_drift_correlation: -0.52
   → Strong temporal drift detected
```

**Interpretation**: This model captures both temporal progression and group discrimination. Would occur if facts evolved over time (e.g., "1M views" → "50M views" → "viral sensation").

## What Makes a Good Temporal Embedding Model?

### For Your Use Case (Human Queries → Retrieve Correct Facts)

**Most important metrics:**
1. **High** `cross_group_mrr` (0.7+) → Finds the right fact group
2. **High** `cross_group_purity@k` (0.6+) → Top results are from correct group
3. **High** `within_group_nearest_fact_mrr` (0.4+) → Returns the most relevant variation

**Less important for paraphrased data:**
- `within_group_temporal_order_correlation` → Expected to be near 0 for paraphrases
- `temporal_drift_correlation` → Expected to be near 0 for paraphrases

### For Time-Evolving Content (If You Had It)

**Strong performance indicators:**
1. **Positive** `within_group_temporal_order_correlation` (0.3+) → Preserves chronology
2. **High** `within_group_nearest_fact_mrr` (0.5+) → Finds adjacent facts quickly
3. **High** `cross_group_mrr` and `cross_group_purity@k` → Discriminates between groups
4. **Negative** `temporal_drift_correlation` (-0.3 or lower) → Captures time-varying semantics

## Understanding Your Results

Your current results (~0.06 temporal_order_correlation, ~0.36 nearest_fact_mrr) are **expected and reasonable** because:

1. **Your facts are paraphrases**: All 7 variations in each group describe the same event with the same details
2. **No content evolution**: Unlike real temporal data (where "1M views" becomes "50M views"), your facts don't change semantically over time
3. **Temporal position is arbitrary**: The timestamp order of paraphrases doesn't convey meaningful information

**What's working well:**
- Cross-group metrics show the models CAN distinguish between different fact groups
- This means for your use case (human query → retrieve fact), the models will find the right group
- Within a group, you'll get a semantically correct fact (even if not in temporal order)

**To capture temporal ordering**, you would need facts where:
- Content changes over time (numbers, events, sentiment)
- Earlier facts are contextually different from later facts
- Temporal position carries semantic meaning

## Timestamp Augmentation

### What Is It?

Timestamp augmentation includes the timestamp directly in the text before embedding. Instead of embedding just the fact text, you embed:

```
"TikTok challenge went viral... [Hour 0]"
"TikTok challenge went viral... [Hour 1]"
"TikTok challenge went viral... [Hour 6]"
```

### Configuration

Set `append_timestamp_to_text: true` in your dataset config:

```yaml
datasets:
- type: temporal
  name: temporal_facts_social
  file_path: datasets/temporal_facts_social_20251018_055003.jsonl
  text_column: instruction
  timestamp_column: metadata.timestamp
  append_timestamp_to_text: true  # Include timestamps in embeddings
  timestamp_format: relative  # iso, human, unix, or relative
```

### Timestamp Formats

1. **iso**: Full ISO timestamp
   ```
   "TikTok challenge... [2025-01-01T00:00:00]"
   ```

2. **human**: Human-readable format
   ```
   "TikTok challenge... [January 1, 2025 at 12:00 AM]"
   ```

3. **unix**: Unix timestamp
   ```
   "TikTok challenge... [1735689600]"
   ```

4. **relative**: Hours from first fact in group
   ```
   "TikTok challenge... [Hour 0]"
   "TikTok challenge... [Hour 1]"
   "TikTok challenge... [Hour 6]"
   ```

### Expected Impact

**With timestamp augmentation (append_timestamp_to_text: true):**

✅ **Improved temporal ordering**
- `within_group_temporal_order_correlation` should increase (0.3-0.7 expected)
- Model can now distinguish facts by their timestamp markers
- Works even for paraphrased content

⚠️ **Potential trade-offs**
- May reduce semantic similarity across time periods
- Cross-group metrics might change (could improve or worsen)
- Embedding space includes artificial temporal markers

**Without timestamp augmentation (default: false):**

✅ **Pure semantic embeddings**
- Focuses on content similarity
- Better for content-based retrieval
- Natural similarity patterns

⚠️ **Limited temporal signal**
- `within_group_temporal_order_correlation` ≈ 0 for paraphrases
- Cannot distinguish chronological order from content alone

### When to Use Timestamp Augmentation

**Use timestamp augmentation when:**
- Testing if temporal markers improve retrieval
- Facts are paraphrases (same semantics, different times)
- You want to explicitly capture chronological order
- Comparing temporal-aware vs content-only embeddings

**Don't use timestamp augmentation when:**
- Content naturally evolves over time (already has temporal signal)
- You want pure semantic similarity
- Timestamps are arbitrary or meaningless
- Production use where timestamps wouldn't be available

### Example Comparison

Run both experiments to compare:

```bash
# Without timestamps (pure semantic)
python cli.py run config/experiments/temporal_experiment.yaml

# With timestamps (temporal-aware)
python cli.py run config/experiments/temporal_experiment_with_timestamps.yaml
```

**Expected results:**
- **Without timestamps**: `temporal_order_correlation` ≈ 0.06 (no signal)
- **With timestamps**: `temporal_order_correlation` ≈ 0.5-0.8 (strong signal)

This demonstrates whether adding timestamps helps the model learn temporal ordering for your use case.
