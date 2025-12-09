import torch
from typing import List, Any

def is_sublist(sub, full):
    n, m = len(sub), len(full)
    return any(full[i:i+n] == sub for i in range(m - n + 1))

# Function to remove segments of a list between a start pattern and an end pattern
def remove_pattern_ranges(seq: List[Any],
                        start_pat: List[Any],
                        end_pat: List[Any]) -> List[Any]:
    """Remove every [start_pat ... end_pat] slice (inclusive) from seq."""
    
    out: List[Any] = []
    i = 0
    n = len(seq)
    ls, le = len(start_pat), len(end_pat)

    while i < n:
        # Check if the start pattern matches at the current position
        if i + ls <= n and seq[i:i+ls] == start_pat:
            # Look for the first occurrence of the end pattern after the start pattern
            j = i + ls
            found_end = -1
            while j + le <= n:
                if seq[j:j+le] == end_pat:
                    found_end = j
                    break  # Stop when the end pattern is found
                j += 1

            # If the end pattern is found, skip the whole segment from start to end
            if found_end != -1:
                i = found_end + le  # Move the index past the end pattern
                continue  # Skip the current iteration and go to the next
            else:
                # If the end pattern is not found, keep the current element and move one step forward
                out.append(seq[i])
                i += 1
        else:
            # If the start pattern is not found, just append the current element
            out.append(seq[i])
            i += 1

    # Return the filtered list with the start-end pattern segments removed
    return out

def low_prob_token_masking(batch):
    response_mask = batch.batch["response_mask"]       # [N, T]
    old_log_prob = batch.batch["old_log_probs"]        # [N, T]
    response_action_region = batch.batch["response_action_region"]

    if "old_response_action_region" in batch.batch:
        old_response_action_region = batch.batch["old_response_action_region"]

        batch_size = batch.batch["response_mask"].shape[0]

        for gen_id in range(batch_size):
            # 256 means containing up to 100+ actions, enough for now
            for cnt in range(256):
                if old_response_action_region[gen_id, cnt * 2] < 0:
                    break
                loc_l = old_response_action_region[gen_id, cnt * 2    ].numpy()
                loc_r = old_response_action_region[gen_id, cnt * 2 + 1].numpy()
                loc_l_off_policy = response_action_region[gen_id, cnt * 2    ].numpy()
                loc_r_off_policy = response_action_region[gen_id, cnt * 2 + 1].numpy()
                old_values = old_log_prob[gen_id, loc_l:loc_r]
                tmp_min = torch.min(old_values) if loc_r - loc_l > 0 else 0
                
                # Disable the extremly low probs
                if tmp_min < -5:
                    response_mask[gen_id, loc_l_off_policy:loc_r_off_policy] = 0
    else:
        batch_size = batch.batch["response_mask"].shape[0]

        for gen_id in range(batch_size):
            # 256 means containing up to 100+ actions, enough for now
            for cnt in range(256):
                if response_action_region[gen_id, cnt * 2] < 0:
                    break
                loc_l = response_action_region[gen_id, cnt * 2    ].numpy()
                loc_r = response_action_region[gen_id, cnt * 2 + 1].numpy()
                old_values = old_log_prob[gen_id, loc_l:loc_r]
                tmp_min = torch.min(old_values) if loc_r - loc_l > 0 else 0
                
                # Disable the extremly low probs
                if tmp_min < -5:
                    response_mask[gen_id, loc_l:loc_r] = 0

    return batch