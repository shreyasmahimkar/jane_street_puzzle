# Jane Street Puzzle Solution: MD5 Reverse-Engineering

The reverse-engineering puzzle for the provided PyTorch model was successfully solved. Based on the clues in the story, the model was meticulously handcrafted with integer weights to execute the MD5 hashing algorithm over 32 blocks.

Here is a step-by-step breakdown of how the solution was obtained using the provided code.

## 1. Extracting the Target MD5 Hash
The story hinted that the second-to-last layer of the network compares the neural network's MD5 computation against a predefined reference hash. This was encoded directly into the layer's bias weights:

> "The second-to-last layer appeared to be three repetitions of the same weights, while its bias contained the same 16 bytes, but incremented by 1 each time... The bias on the last layer was -15, so the final neuron would only fire when v=x for all 16 bytes."

The script `code/extract_hash.py` was created to load `model_3_11.pt` and read these biases out of the PyTorch linear layers. By negating the bias block corresponding to `-x`, the 16 target bytes were extracted: `[199, 239, 101, 35, 60, 64, 170, 50, 194, 185, 172, 227, 117, 149, 250, 124]`

Converted to hexadecimal, the target MD5 hash to be reversed was: **`c7ef65233c40aa32c2b9ace37595fa7c`**

## 2. Brute-Forcing the Solution
The puzzle text provided a final, vital hint:

> "The puzzle creator had intentionally made the hash easy to brute force, leaving various small hints in the puzzle description and Python code that the solution was composed of two English words, lowercased, concatenated by a space... Once he had a big enough word list, he got the answer."

To crack this quickly, a high-performance C program (`code/brute_force.c`) was developed that:

- Downloaded an extensive dictionary of ~370,000 english words (much larger than the Top 10,000 reference).
- Used Apple's native GCD (`dispatch_apply`) to parallelize the search across multi-core processors.
- Computed the MD5 hash of every two-word combination (`word1 + " " + word2`) via `<CommonCrypto/CommonDigest.h>`.

Out of the ~136 billion possible combinations, the program successfully cracked the hash in under 5 minutes.

## The Final Solution
The hash matched the following two words:

> [!IMPORTANT]
> **bitter lesson**

This is a fitting answer, directly referring to "The Bitter Lesson", the famous AI essay by Rich Sutton stating that general methods leveraging computation are ultimately the most effective—which rings very true for this brute-force puzzle!
