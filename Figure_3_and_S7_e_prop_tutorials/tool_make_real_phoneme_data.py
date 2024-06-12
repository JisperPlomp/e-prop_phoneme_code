import nltk
from nltk.corpus import cmudict
import random
import json

# Ensure you have the CMU Pronouncing Dictionary
nltk.download('cmudict')

# Load the CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

# Function to select random words and process phoneme data
def create_phoneme_data(num_words):
    # Select a subset of random words
    selected_words = random.sample(list(cmu_dict.keys()), num_words)

    # Collect unique phonemes
    phoneme_set = set()
    word_to_phoneme_ids = {}
    phoneme_sequences = []

    for word in selected_words:
        for phoneme_seq in cmu_dict[word]:
            phoneme_set.update(phoneme_seq)
            word_to_phoneme_ids[word] = phoneme_seq
            phoneme_sequences.append(phoneme_seq)
            break

    # Limit to 30 unique phonemes (if more than 30 are collected)
    unique_phonemes = list(phoneme_set)

    # Create phoneme to ID mapping
    phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(unique_phonemes)}

    # Create sequences and word mapping
    phoneme_sequences_ids = [[phoneme_to_id[phoneme] for phoneme in seq] for seq in phoneme_sequences]
    word_mapping = {tuple(seq): idx for idx, seq in enumerate(phoneme_sequences_ids)}
    word_to_index = {word: idx for idx, word in enumerate(selected_words)}

    return phoneme_sequences_ids, word_mapping, word_to_index, phoneme_to_id, len(unique_phonemes)

# Set the number of words to select
num_words = 500

# Create the phoneme data
phoneme_sequences, word_mapping, word_to_index, phoneme_to_id, num_phonemes = create_phoneme_data(num_words)

# Define n_in based on the number of unique phonemes
neurons_per_phoneme = 1
n_in = neurons_per_phoneme * num_phonemes

# Save to a text file
data = {
    "phoneme_sequences": phoneme_sequences,
    "word_mapping": {str(k): v for k, v in word_mapping.items()},
    "word_to_index": word_to_index,
    "phoneme_to_id": phoneme_to_id,
    "n_in": n_in
}

with open("phoneme_data.json", "w") as f:
    json.dump(data, f, indent=4)

print("Phoneme data saved to phoneme_data.json")



