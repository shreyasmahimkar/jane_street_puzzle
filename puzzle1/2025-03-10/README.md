# Neural Network MD5 Hash Puzzle Solution

> **Note:** If on Python version 3.11 or above, use `model_3_11.pt`, else use `model.pt`.

## 1. Extracting the Target MD5 Hash
Based on clues in the story regarding the bias weights of the final layers, we developed a script (`extract_hash.py`) to load the model and read the biases. By analyzing these values, we extracted the following 16 target bytes: `[199, 239, 101, 35, 60, 64, 170, 50, 194, 185, 172, 227, 117, 149, 250, 124]`. This corresponds to the hexadecimal MD5 hash: `c7ef65233c40aa32c2b9ace37595fa7c`.

## 2. Brute-Forcing the Solution
The puzzle description indicated the solution consisted of two lowercase English words separated by a space. To find the match, we wrote a high-performance C program (`brute_force.c`) that:

- Utilized an extensive dictionary of approximately 370,000 words.
- Parallelized the search using Apple’s GCD (`dispatch_apply`) for multi-core efficiency.
- Iterated through approximately 136 billion combinations.

## The Final Solution
The program successfully cracked the hash in under five minutes. The two-word solution is: **`bitter lesson`**.