"""Eight synthetic tasks for attention failure mode discovery.

Each generator returns (sequences, masks, task_info):
  sequences: (N, L) int32
  masks: dict of (N, L) bool — mutually exclusive token categories
  task_info: dict with vocab_size, critical_mask, description
"""

import numpy as np

SEQ_LEN = 256


def associative_recall(num_seqs, seq_len=SEQ_LEN, seed=0):
    """Key-value recall: k1 v1 k2 v2 ... <sep> ki -> vi.

    Keys: tokens 0-7, Values: tokens 8-15, Separator: 16, Query repeats a key.
    8-12 pairs, then separator + query key, rest is padding.
    Critical: predict value at the position after the query key.
    """
    rng = np.random.RandomState(seed)
    vocab = 17  # 0-7 keys, 8-15 values, 16 separator
    seqs = np.zeros((num_seqs, seq_len), dtype=np.int32)
    masks = {"context": np.zeros((num_seqs, seq_len), dtype=bool),
             "query": np.zeros((num_seqs, seq_len), dtype=bool),
             "padding": np.zeros((num_seqs, seq_len), dtype=bool)}

    for i in range(num_seqs):
        n_pairs = rng.randint(6, 9)  # max 8 keys available (0-7)
        keys = rng.choice(8, size=n_pairs, replace=False)
        values = rng.randint(8, 16, size=n_pairs)

        pos = 0
        for j in range(n_pairs):
            if pos + 1 >= seq_len:
                break
            seqs[i, pos] = keys[j]
            seqs[i, pos + 1] = values[j]
            masks["context"][i, pos] = True
            masks["context"][i, pos + 1] = True
            pos += 2

        if pos + 2 < seq_len:
            seqs[i, pos] = 16  # separator
            masks["context"][i, pos] = True
            pos += 1

            query_idx = rng.randint(0, n_pairs)
            seqs[i, pos] = keys[query_idx]
            masks["context"][i, pos] = True
            pos += 1

            seqs[i, pos] = values[query_idx]  # target
            masks["query"][i, pos] = True
            pos += 1

        # Fill rest with padding pattern
        for j in range(pos, seq_len):
            seqs[i, j] = 16
            masks["padding"][i, j] = True

    return seqs, masks, {"vocab_size": vocab, "critical_mask": "query",
                          "description": "Associative recall (k-v pairs)"}


def selective_copy(num_seqs, seq_len=SEQ_LEN, seed=0):
    """Copy target tokens, ignore distractors.

    Targets: tokens 0-7, Distractors: tokens 8-13 (but some match target IDs contextually).
    Marker: 14. After marker, reproduce targets in order.
    """
    rng = np.random.RandomState(seed)
    vocab = 15
    seqs = np.zeros((num_seqs, seq_len), dtype=np.int32)
    masks = {"input": np.zeros((num_seqs, seq_len), dtype=bool),
             "copy": np.zeros((num_seqs, seq_len), dtype=bool),
             "padding": np.zeros((num_seqs, seq_len), dtype=bool)}

    for i in range(num_seqs):
        n_targets = rng.randint(10, 20)
        targets = rng.randint(0, 8, size=n_targets)

        pos = 0
        # Input phase: targets interleaved with distractors
        for j in range(n_targets):
            if pos >= seq_len - n_targets - 2:
                break
            seqs[i, pos] = targets[j]
            masks["input"][i, pos] = True
            pos += 1
            # 1-3 distractors
            n_dist = rng.randint(1, 4)
            for _ in range(n_dist):
                if pos >= seq_len - n_targets - 2:
                    break
                seqs[i, pos] = rng.randint(8, 14)
                masks["input"][i, pos] = True
                pos += 1

        # Marker
        if pos < seq_len:
            seqs[i, pos] = 14
            masks["input"][i, pos] = True
            pos += 1

        # Copy phase: reproduce targets
        actual_targets = min(n_targets, seq_len - pos)
        for j in range(actual_targets):
            if pos >= seq_len:
                break
            seqs[i, pos] = targets[j]
            masks["copy"][i, pos] = True
            pos += 1

        for j in range(pos, seq_len):
            seqs[i, j] = 14
            masks["padding"][i, j] = True

    return seqs, masks, {"vocab_size": vocab, "critical_mask": "copy",
                          "description": "Selective copy past distractors"}


def dual_stream(num_seqs, seq_len=SEQ_LEN, seed=0):
    """Two interleaved periodic streams: A (period 3) and B (period 4).

    Stream A: tokens 0-2, Stream B: tokens 3-6. Interleaved: A1 B1 A2 B2 ...
    """
    rng = np.random.RandomState(seed)
    vocab = 7
    seqs = np.zeros((num_seqs, seq_len), dtype=np.int32)
    masks = {"stream_a": np.zeros((num_seqs, seq_len), dtype=bool),
             "stream_b": np.zeros((num_seqs, seq_len), dtype=bool)}

    pattern_a = [0, 1, 2]
    pattern_b = [3, 4, 5, 6]

    for i in range(num_seqs):
        phase_a = rng.randint(0, 3)
        phase_b = rng.randint(0, 4)
        for pos in range(seq_len):
            if pos % 2 == 0:
                seqs[i, pos] = pattern_a[phase_a % 3]
                masks["stream_a"][i, pos] = True
                phase_a += 1
            else:
                seqs[i, pos] = pattern_b[phase_b % 4]
                masks["stream_b"][i, pos] = True
                phase_b += 1

    return seqs, masks, {"vocab_size": vocab, "critical_mask": "stream_a",
                          "description": "Dual-stream interleaved tracking"}


def nested_periodicity(num_seqs, seq_len=SEQ_LEN, seed=0):
    """Three nested periodic scales.

    Fast: period 3 (tokens from a 3-token pattern).
    Medium: every 12 tokens (4 fast cycles), shift to a new fast pattern.
    Slow: every 36 tokens (3 medium cycles), shift to a new set of medium patterns.
    Vocab 0-11. Critical: accuracy on phase-change positions.
    """
    rng = np.random.RandomState(seed)
    vocab = 12
    seqs = np.zeros((num_seqs, seq_len), dtype=np.int32)
    masks = {"stable": np.zeros((num_seqs, seq_len), dtype=bool),
             "phase_change": np.zeros((num_seqs, seq_len), dtype=bool)}

    for i in range(num_seqs):
        all_patterns = list(range(vocab))
        rng.shuffle(all_patterns)
        # 4 fast patterns of 3 tokens each
        fast_patterns = [all_patterns[j * 3:(j + 1) * 3] for j in range(4)]

        slow_phase = 0
        med_phase = 0
        fast_phase = 0

        for pos in range(seq_len):
            slow_idx = (pos // 36) % 2
            med_idx = ((pos % 36) // 12)
            pattern_idx = (slow_idx * 2 + med_idx) % 4
            pattern = fast_patterns[pattern_idx]
            seqs[i, pos] = pattern[pos % 3]

            # Phase change: first token of new medium or slow cycle
            if pos > 0 and (pos % 12 == 0):
                masks["phase_change"][i, pos] = True
            else:
                masks["stable"][i, pos] = True

    return seqs, masks, {"vocab_size": vocab, "critical_mask": "phase_change",
                          "description": "Nested periodicity (3 scales)"}


def sparse_needle(num_seqs, seq_len=SEQ_LEN, seed=0):
    """Retrieve a single needle token from early in the sequence.

    Needle: one of tokens 0-7, placed at a random position in first 20.
    Filler: repeating pattern of tokens 8-12.
    Query signal: token 13 at position seq_len-2.
    Target: needle token at position seq_len-1.
    """
    rng = np.random.RandomState(seed)
    vocab = 14
    seqs = np.zeros((num_seqs, seq_len), dtype=np.int32)
    masks = {"filler": np.zeros((num_seqs, seq_len), dtype=bool),
             "retrieval": np.zeros((num_seqs, seq_len), dtype=bool)}

    filler_pattern = [8, 9, 10, 11, 12]

    for i in range(num_seqs):
        needle_pos = rng.randint(1, 20)
        needle_val = rng.randint(0, 8)

        for pos in range(seq_len):
            if pos == needle_pos:
                seqs[i, pos] = needle_val
                masks["filler"][i, pos] = True  # still in filler region
            elif pos == seq_len - 2:
                seqs[i, pos] = 13  # query signal
                masks["filler"][i, pos] = True
            elif pos == seq_len - 1:
                seqs[i, pos] = needle_val  # target
                masks["retrieval"][i, pos] = True
            else:
                seqs[i, pos] = filler_pattern[pos % 5]
                masks["filler"][i, pos] = True

    return seqs, masks, {"vocab_size": vocab, "critical_mask": "retrieval",
                          "description": "Sparse needle retrieval"}


def pattern_confounders(num_seqs, seq_len=SEQ_LEN, seed=0):
    """Original pattern, then confounder pattern, then resume original.

    Original: period-4 from tokens 0-3. Confounder: period-4 from tokens 4-7.
    Resume signal: token 8. After resume, must predict original pattern.
    """
    rng = np.random.RandomState(seed)
    vocab = 9
    seqs = np.zeros((num_seqs, seq_len), dtype=np.int32)
    masks = {"original": np.zeros((num_seqs, seq_len), dtype=bool),
             "confounder": np.zeros((num_seqs, seq_len), dtype=bool),
             "resume": np.zeros((num_seqs, seq_len), dtype=bool),
             "padding": np.zeros((num_seqs, seq_len), dtype=bool)}

    for i in range(num_seqs):
        orig = [0, 1, 2, 3]
        conf = [4, 5, 6, 7]
        # Shuffle both patterns
        rng.shuffle(orig)
        rng.shuffle(conf)

        pos = 0
        # Original pattern: 30-50 tokens
        orig_len = rng.randint(30, 51)
        phase = 0
        for _ in range(min(orig_len, seq_len)):
            seqs[i, pos] = orig[phase % 4]
            masks["original"][i, pos] = True
            phase += 1
            pos += 1

        # Confounder: 10-15 tokens
        conf_len = rng.randint(10, 16)
        conf_phase = 0
        for _ in range(min(conf_len, seq_len - pos)):
            seqs[i, pos] = conf[conf_phase % 4]
            masks["confounder"][i, pos] = True
            conf_phase += 1
            pos += 1

        # Resume signal
        if pos < seq_len:
            seqs[i, pos] = 8
            masks["confounder"][i, pos] = True
            pos += 1

        # Resume original — first 3 are critical
        resume_phase = phase  # continue from where original left off
        for j in range(min(seq_len - pos, 40)):
            if pos >= seq_len:
                break
            seqs[i, pos] = orig[resume_phase % 4]
            if j < 3:
                masks["resume"][i, pos] = True
            else:
                masks["original"][i, pos] = True
            resume_phase += 1
            pos += 1

        for j in range(pos, seq_len):
            seqs[i, j] = 8
            masks["padding"][i, j] = True

    return seqs, masks, {"vocab_size": vocab, "critical_mask": "resume",
                          "description": "Pattern recovery after confounders"}


def mode_interference(num_seqs, seq_len=SEQ_LEN, seed=0):
    """Two modes with overlapping tokens, alternating with separator.

    Mode A: [0, 1, 2, 3] period 4. Mode B: [2, 3, 4, 5] period 4.
    Tokens 2 and 3 overlap — predicting them requires knowing the active mode.
    Separator: token 6.
    """
    rng = np.random.RandomState(seed)
    vocab = 7
    seqs = np.zeros((num_seqs, seq_len), dtype=np.int32)
    masks = {"unique": np.zeros((num_seqs, seq_len), dtype=bool),
             "shared": np.zeros((num_seqs, seq_len), dtype=bool),
             "separator": np.zeros((num_seqs, seq_len), dtype=bool)}

    mode_a = [0, 1, 2, 3]
    mode_b = [2, 3, 4, 5]
    shared_tokens = {2, 3}

    for i in range(num_seqs):
        mode = rng.randint(0, 2)  # start mode
        pos = 0
        phase = 0

        while pos < seq_len:
            segment_len = rng.randint(10, 16)
            pattern = mode_a if mode == 0 else mode_b

            for _ in range(segment_len):
                if pos >= seq_len:
                    break
                tok = pattern[phase % 4]
                seqs[i, pos] = tok
                if tok in shared_tokens:
                    masks["shared"][i, pos] = True
                else:
                    masks["unique"][i, pos] = True
                phase += 1
                pos += 1

            # Separator
            if pos < seq_len:
                seqs[i, pos] = 6
                masks["separator"][i, pos] = True
                pos += 1

            mode = 1 - mode
            phase = 0

    return seqs, masks, {"vocab_size": vocab, "critical_mask": "shared",
                          "description": "Mode interference (overlapping tokens)"}


def compositional_lookup(num_seqs, seq_len=SEQ_LEN, seed=0):
    """Two-hop lookup: pointer -> position -> answer.

    Position 0: pointer token (0-7) encoding a target position (ptr * 3 + 10).
    Target position: answer token (8-15).
    All other positions: filler (token 16).
    Last position: query signal (17), target is the answer token.
    Requires 2-hop composition — ceiling test for 1-layer models.
    """
    rng = np.random.RandomState(seed)
    vocab = 18
    seqs = np.zeros((num_seqs, seq_len), dtype=np.int32)
    masks = {"filler": np.zeros((num_seqs, seq_len), dtype=bool),
             "answer": np.zeros((num_seqs, seq_len), dtype=bool)}

    for i in range(num_seqs):
        ptr = rng.randint(0, 8)
        target_pos = ptr * 3 + 10  # positions 10, 13, 16, ..., 31
        answer = rng.randint(8, 16)

        # Fill with filler
        seqs[i, :] = 16
        masks["filler"][i, :] = True

        # Pointer at position 0
        seqs[i, 0] = ptr

        # Answer at target position
        if target_pos < seq_len:
            seqs[i, target_pos] = answer

        # Query at last position, answer is the target
        seqs[i, seq_len - 2] = 17  # query signal
        seqs[i, seq_len - 1] = answer
        masks["filler"][i, seq_len - 1] = False
        masks["answer"][i, seq_len - 1] = True

    return seqs, masks, {"vocab_size": vocab, "critical_mask": "answer",
                          "description": "Compositional 2-hop lookup (ceiling test)"}


# Registry
ALL_TASKS = {
    "associative_recall": associative_recall,
    "selective_copy": selective_copy,
    "dual_stream": dual_stream,
    "nested_periodicity": nested_periodicity,
    "sparse_needle": sparse_needle,
    "pattern_confounders": pattern_confounders,
    "mode_interference": mode_interference,
    "compositional_lookup": compositional_lookup,
}
