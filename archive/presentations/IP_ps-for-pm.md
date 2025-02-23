---
marp: true
author: atharvas
---
# Neurosymbolic methods for Precision Medicine
## (Flowchart Generation)

---

# Overview
 - Intorduction to NEAR
 - Synthetic Dataset
    - Dataset | Grammar
    - Observations
 - PhysioNet Sepsis Dataset
    - Dataset | Grammar
    - Observations
 - Next Steps

---

# NEAR Overview

- Given:
    - A domain specific language
    - A dataset of input-output examples
- Algorithm:
    - Start with an empty program.
    - Replace all "holes" with a type-appropriate neural network.
    - Train on input-output examples and calculate loss.
    - The final loss and the structural cost becomes the "cost" of an edge.
    - Formulate as a weighted graph search problem and use graph search algorithms (like A*) to find better nodes.

---

# Next Steps
