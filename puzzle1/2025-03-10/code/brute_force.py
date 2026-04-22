import hashlib
import urllib.request
import os
import time
import itertools

TARGET_HASH = "c7ef65233c40aa32c2b9ace37595fa7c"
WORDLIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
WORDLIST_FILE = "words_alpha.txt"

def load_words():
    if not os.path.exists(WORDLIST_FILE):
        print("Downloading wordlist...")
        urllib.request.urlretrieve(WORDLIST_URL, WORDLIST_FILE)
    
    print("Loading words...")
    with open(WORDLIST_FILE, 'r') as f:
        # Keep only lowercased words as per puzzle
        words = [line.strip().lower() for line in f if line.strip()]
    
    # Remove duplicates
    words = list(set(words))
    print(f"Loaded {len(words)} unique words.")
    return words

def crack_hash(words):
    print(f"Starting brute force target {TARGET_HASH}...")
    start_time = time.time()
    
    total = len(words)
    # We are looking for "word1 word2"
    # To optimize, we can pre-hash halves if it's not simply MD5.
    # But for MD5, we just iterate.
    
    # Actually wait, doing len(words)^2 MD5 hashes in Python is slow.
    # len(words) is ~370,000. 370k^2 is ~136 billion hashes. Doing this in simple Python will take days.
    # We should use something faster, but maybe we can just write it in C, or parallelize,
    # or just use hashcat?
    pass

if __name__ == "__main__":
    words = load_words()
    crack_hash(words)
