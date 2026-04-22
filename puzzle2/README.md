# "I dropped a neural net" - Jane Street Puzzle

This repository chronicles the programmatic algorithms designed to resolve the [Jane Street "I dropped a neural net" puzzle](https://huggingface.co/spaces/jane-street/droppedaneuralnet). 

## The Challenge

An extremely valuable PyTorch trading model was "accidentally dropped" and disassembled into 97 unmarked `.pth` linear layers (`piece_0.pth` through `piece_96.pth`). Because they had lost their ordering, attempting to identically reconstruct the neural network back together poses a complex factorial assignment problem.

### Model Architecture
The neural network relies on a 48-layer Sequential Residual Network (ResNet) structure crafted from `Block` elements alongside a culminating `LastLayer` projection, targeting 48 input features extracted from a `historical_data.csv` holdout batch.

*   **Block (`[in_dim: 48, hidden_dim: 96]`)**: A customized component containing an input linear layer, ReLU activation, output linear layer, and a residual addition track. 
*   **LastLayer (`[in_dim: 48, out_dim: 1]`)**: The final prediction linear component formatting the unified scalar target `pred`. (`piece_85.pth`)

With $48!$ permutations for the input connections and $48!$ for the outbound connections, the mathematical landscape scales geometrically to $(48!)^2 \approx 10^{122}$ — larger than the amount of atoms in the observable universe. 

---

## Evolution of Iterative Solvers

The codebase traces through multiple experimental iterations. Ranging from primitive continuous gradients to structural validation sequences, below is an explanation of the purpose of each script phase and how the limitations uncovered paved the way for subsequent improvements.

### 1. Exploratory Analytics and Feature Extraction
**Files:** `inspect_pieces.py`, `test_shapes.py`, `test_inode_shapes.py`, `test_bias.py`, `test_bias_norms.py`, `test_sparsity.py`

Early efforts utilized localized scripts to process the 97 fragmented components, decoding their tensor bounds and classifying node identities (`inps` vs `outs`). Initial hypotheses investigated straightforward heuristic alignment patterns via OS metadata invariants (inode order), cross-referencing tensor sparsity ratios, and extracting contiguous $L^2$ bias norms. 
*   **Limitation:** It was computationally established that permutations were profoundly detached intentionally; basic sorting patterns, norm distributions, and file-based temporal hints offered no functional baseline for sequence discovery.

### 2. Gradient-Based Continuous Optimization
**Files:** `solver.py`, `solver2.py`, `solver3.py`

Since the mathematical architecture remained fixed, a differentiable soft-router pattern was implemented to sidestep raw combination trials. Utilizing PyTorch's native `.optim` frameworks, the matrices were structurally replaced with continuous Gumbel-Softmax distribution arrays executing simulated training over the entire 10,000 index datapoints.
*   **Improvement & Limitation:** While the Mean Squared Error (MSE) diminished significantly against the benchmark labels, deep continuous learning on unstructured factorials failed to isolate unambiguous boundaries. The extreme severity of neural saddle points created compounding non-convexity constraints restricting the network from achieving exact identity convergence.

### 3. Discrete Path-Finding Traversal
**Files:** `test_progressive.py`, `greedy_search.py`, `beam_search.py`

Moving away from continuous loss surfaces, rigid tree-search formulations were deployed to rebuild the sequence component by component. The search logic advanced sequentially—ranging from naive optimal-first combinations (`greedy_search.py`) to implementing width-oriented exploration pools preserving extensive active candidates (`beam_search.py`).
*   **Improvement & Limitation:** Tracking defined path structures improved absolute inference outputs compared to stochastic limits. Nevertheless, without knowledge of end-state permutations, delayed error compounding continuously disrupted localized search states causing ultimate tree-collapse at profound depths.

### 4. Stochastic Refinement Boundaries
**Files:** `hill_climb.py`, `test_parallel.py`

Leveraging the highest accuracy outputs salvaged from the preceding depth searches, robust simulated annealing / hill climbing methodologies were crafted. Random swap functions rotated intermediate blocks incrementally whenever local bounds showcased superior prediction alignment, while `test_parallel.py` simultaneously investigated decoupled residual independence models.
*   **Limitation:** While empirically valid, swap engines slowed to an exhaustive plateau across the search space showing that isolated brute force could not traverse out of highly-stabilized localized optimization valleys. 

### 5. Final Exact Analytical Constraint Solver
**File:** `solver_final.py`

Abandoning exhaustive search states and non-convex gradient techniques, the problem architecture was entirely abstracted into two exact operations exploiting the intrinsic analytical constraints found in properly trained residual parameters:
1.  **Bipartite Pairing Alignment**: Structural residual dependency requires "negative diagonal alignment". The algorithm isolates the mathematically perfect matching matrices mapping $W_{\text{out}} \times W_{\text{in}}$ cross-products and applies a Hungarian matching iteration across the trace to perfectly couple the 48 independent systems.
2.  **Norm Seeding & Bubble Repair**: Block sequences must be evaluated sequentially. Initial orientations were anchored by their relative bounded Frobenius Norm expansions ($\|W_{\text{out}}\|_F$) as depth proxies, followed up immediately by a programmatic bubble-repair sweep validating prediction arrays dynamically against the holdout labels.
*   **Final Result:** This execution flawlessly isolates the solution variables—breaking free from computational dead ends by achieving an uncorrupted validation trajectory of exactly `0.0` MSE.

---

## Execution
Run the isolated analytical extraction loop to parse indices dynamically.

```bash
# Ensure local virtual environment dependencies
pip install torch pandas scipy

# Execute the exact solver to generate the index representation
python solver_final.py
```

## The Final Solution
The calculated sequence of piece indices required to properly reconstruct the network is as follows:

`43,34,65,22,69,89,28,12,27,76,81,8,5,21,62,79,64,70,94,96,4,17,48,9,23,46,14,33,95,26,50,66,1,40,15,67,41,92,16,83,77,32,10,20,3,53,45,19,87,71,88,54,39,38,18,25,56,30,91,29,44,82,35,24,61,80,86,57,31,36,13,7,59,52,68,47,84,63,74,90,0,75,73,11,37,6,58,78,42,55,49,72,2,51,60,93,85`
