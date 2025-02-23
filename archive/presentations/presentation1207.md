---
marp: true
author: atharvas
---

# Slot Attention with rule learning

---

# Agenda:
 - Slot attention decoder
 - DSL
 - (Extra) Biological fluid dynamics (https://arxiv.org/abs/1904.13013)

---

# Slot Attn Decoder

<center>

![e](processed_viz2_1207/sym_training.png)

</center>

---

# Changes

![e](processed_viz2_1207/dsp.png)

Changes:
 - Use smoothL1 loss instead of L1 loss.
 - Use the same loss function for matching and backprop.

<!-- 

<style> .container { display: flex; } .col { flex: 1; } </style>

<div class="container"> <div class="col">
Pipeline:

</div>
<div class="col">
Training curve:

</div>
</div> -->

---
# Slot Attn Decoder

Training Curve
![e](processed_viz2_1207/training_curve.png)



---

# Sample 0

<center>

![e](processed_viz2_1207/sample_0.png)

</center>

---

# Sample 1

<center>

![e](processed_viz2_1207/sample_1.png)

</center>

---

# Sample 2

<center>

![e](processed_viz2_1207/sample_2.png)

</center>

---

# DSL

Inputs:
```python
pos :: [int, int]
color :: int
shape :: {circ, sq, tri}
slot :: [color, shape, pos]
```



---

# (Extra) Fluid Dynamics
https://arxiv.org/abs/1904.13013

![e](processed_viz2_1207/bfd.png)
