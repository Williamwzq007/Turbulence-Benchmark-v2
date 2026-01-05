# Reproduction Notes: Turbulence Benchmark (Question 34)

## 1. Paper Overview and Core Idea

This reproduction is based on *Turbulence: Benchmarking Robustness of Large Language Models under Parameter Perturbations*.  
These records are compiled through my hands-on replication of the code, with assistance from AI tools for structural organization and clarifying complex theoretical concepts.
The central idea of the paper is to move beyond single-run accuracy metrics and instead evaluate **robustness** of LLMs by introducing *controlled perturbations* in problem parameters and observing consistency of correctness across multiple runs.

Rather than asking “can the model solve this problem once?”, the benchmark asks:
- Does the model remain correct across *neighbourhoods* of the same problem?
- How sensitive is correctness to small changes in parameters?
- Does correctness degrade gracefully or collapse abruptly?

To capture this, the paper introduces metrics such as:
- **Accuracy Score (AS)** – average correctness across rounds
- **Correctness Potential Score (CPS)** – whether at least one correct solution exists
- **Consistent Correctness Score (CCS)** – whether correctness is preserved across all perturbations

This reproduction focuses on **Question 34**, a set-theoretic reasoning task with combinatorial structure, which is particularly sensitive to formatting and syntactic precision in LLM outputs.

---
## 2. Scope and Limitations of This Reproduction

This reproduction focuses on validating the execution logic and diagnostic intent of the
Turbulence benchmark, rather than reproducing the full quantitative claims of the original paper.

Accordingly, this study does not aim to match reported metric values at scale.
Instead, the goal is to verify whether the benchmark pipeline is able to surface robustness
failure modes under controlled parameter perturbations, even in a reduced experimental setting.

Within this scope, Question 34 was selected deliberately rather than at random.
This question involves set-theoretic reasoning with a combinatorial structure, where the number
of parameters grows rapidly with problem specification.
As a result, small perturbations can significantly increase both reasoning complexity and output
length, making the task particularly sensitive to robustness issues.

From a benchmark-design perspective, Question 34 serves as an effective stress test:
it simultaneously probes logical reasoning, output formatting stability, and the model’s ability
to maintain syntactic correctness under parameter-heavy specifications.
This aligns closely with the core objective of the Turbulence benchmark.

Due to hardware and API constraints, the number of perturbation rounds was reduced.
As a result, the reported metric values should be interpreted qualitatively rather than
statistically.
However, the central objective — observing robustness failure modes under controlled
perturbations — remains fully preserved.

---

## 3. Reproduction Environment and Constraints

### Hardware limitations
I conducted this reproduction on a personal laptop rather than a dedicated server.  
As a result:
- Running the full benchmark with the original number of rounds was computationally infeasible
- I reduced number_of_rounds to 2 as a concrete instantiation of the scope defined above,
enabling end-to-end execution under realistic hardware constraints.

This reduction preserves the *structure* of the benchmark while making it executable under realistic undergraduate constraints.

### API constraints
The original implementation supports multiple LLM backends.  
In practice:
- Several APIs were not accessible without paid plans
- I selected **Gemini Flash (latest)** as it was accessible and supported code generation

During execution, I encountered deprecation warnings related to `google.generativeai`, which reflect upstream API changes rather than implementation errors. These warnings were acknowledged but did not prevent execution.

---

## 4. Reproduction Process (Step-by-Step)

### Step 1: Initial setup and dependency resolution
- Installed required Python dependencies
- Resolved platform-specific issues (Windows paths, pytest plugins)
- Identified missing Graphviz dependency; resolved by installing system-level `dot.exe`


### Step 2: Running the benchmark pipeline
- Executed `main.py` with reduced rounds
- Verified that:
  - LLM responses were generated
  - Test cases were dynamically created
  - Pytest successfully evaluated generated code

At this stage, test suites for each neighbourhood instance were executed correctly, and pytest logs confirmed successful evaluation runs.

---

### Step 3: Metric computation debugging
During metric aggregation, I encountered multiple issues:
- Function signature mismatches (`calculate_accuracy`, `calculate_correctness_potential`, `calculate_consistent_correctness`)
- Inconsistent assumptions about arguments (`R`, `M`) being passed vs inferred
- Division-by-zero edge cases when data was empty

These were resolved by:
- Refactoring metric functions to infer dimensions directly from data
- Making function signatures robust to extra arguments
- Adding explicit safeguards against empty inputs

This step significantly deepened my understanding of how the benchmark aggregates correctness signals across rounds and parameters.

---

### Step 4: Interpreting failures in LLM outputs
Despite successful execution of the evaluation pipeline, some neighbourhood instances failed with errors such as: 

FAIL: '(' was never closed (<unknown>, line 1)

Upon inspection, these failures were traced to **LLM-generated code**, not the evaluation logic.  
For example, the model produced incomplete function signatures with hundreds of parameters, leading to syntactically invalid Python.

This behaviour is precisely what the benchmark aims to surface:
- Sensitivity to output length
- Failure under parameter-heavy specifications
- Brittleness in code generation under perturbation

---

## 5. Final Results and Observations

### What worked
- The full benchmark pipeline ran successfully end-to-end
- Test generation, execution, and aggregation all functioned correctly
- Metric computation produced valid outputs for most neighbourhoods

### What failed (and why it matters)
- Some instances failed due to malformed LLM outputs
- These failures were systematic rather than random
- They highlight robustness limitations rather than implementation flaws

From a learner’s perspective, this was a crucial insight:  
**robustness evaluation exposes failure modes that standard accuracy metrics completely miss.**

---

## 6. Reflections as a Learner

This reproduction required substantially more than “just running code”:
- I had to adapt the experimental design to realistic hardware constraints
- I debugged metric logic to align implementation with theoretical definitions
- I learned how evaluation pipelines must be robust not only to model errors, but also to malformed outputs

Most importantly, this project shifted my understanding of LLM evaluation from:
> *“Does the model get the right answer?”*  
to  
> *“Under what conditions does the model stop being reliable?”*

---

## 7. Reproducibility Notes

- Reduced rounds (`number_of_rounds = 2`) due to hardware limits
- Gemini Flash used due to API accessibility
- Raw LLM responses retained locally as evidence but not included due to size
- Aggregated metrics and pytest reports included for verification

## 8. Insights on Benchmark Design

Through this reproduction, I gained a deeper understanding of why turbulence-style evaluation
reveals failure modes that single-run benchmarks cannot capture.

In particular, correctness collapses observed in parameter-heavy neighbourhoods suggest that
LLM failures are often abrupt rather than gradual, highlighting the importance of consistency-based metrics such as CCS.


