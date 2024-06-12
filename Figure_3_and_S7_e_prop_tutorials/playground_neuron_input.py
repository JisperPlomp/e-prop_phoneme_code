import numpy as np
import random

def generate_spike_pattern(phoneme_id, num_neurons):
    np.random.seed(phoneme_id)
    return np.random.rand(num_neurons)

def generate_phoneme_data_unique_patterns(batch_size, phoneme_sequences, word_mapping, num_neurons, cue_spacing):
    max_length = max(len(seq) for seq in phoneme_sequences)
    spk_data = np.zeros((batch_size, max_length * cue_spacing, num_neurons))
    in_nums = np.zeros((batch_size, max_length), dtype=np.int32)
    target_data = np.zeros((batch_size,), dtype=np.int32)

    for i in range(batch_size):
        seq_idx = random.randint(0, len(phoneme_sequences) - 1)
        seq = phoneme_sequences[seq_idx]
        for t, phoneme_id in enumerate(seq):
            pattern = generate_spike_pattern(phoneme_id, num_neurons)
            spk_data[i, t * cue_spacing:(t + 1) * cue_spacing] = pattern
        in_nums[i, :len(seq)] = seq
        target_data[i] = word_mapping[tuple(seq)]
    
    return spk_data, in_nums, target_data

# Example usage
batch_size = 4
phoneme_sequences = [
    [1, 2, 3,1],
    [3, 2, 1],
    [1, 3, 2]
]
word_mapping = {
    (1, 2, 3,1): 0,
    (3, 2, 1): 1,
    (1, 3, 2): 2
}
num_neurons = 10
cue_spacing = 50

spk_data, in_nums, target_data = generate_phoneme_data_unique_patterns(batch_size, phoneme_sequences, word_mapping, num_neurons, cue_spacing)

print("Spike Data Shape:", spk_data.shape)
print("Input Numbers Shape:", in_nums.shape)
print("Target Data Shape:", target_data.shape)
print(in_nums)

