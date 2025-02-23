---
marp: true
author: atharvas
---

# Programmatic Generation

---

# Recap

   The core problem is that the encoder/decoder are too flexible and are not allowing the rule controller to learn. Three potential ideas were suggested:

1. Increase the dataset complexity/capacity.

2. Increase the latent dimension based on some schedule.

3. This problem might require looking into task programming.


---




## Increase dataset complexity

Currently working on this.
Here is a task breakdown:

1. Create a  representation that is independent of number of objects
2. Train VAE on representation
3. Train modules on representation
4. Train controller on representation

This did not work.

```python
   controller output -> tensor([6, 6, 6, 6, 6, 6, 6, 6, 6, 6], device='cuda:0')
   actual action     -> tensor([24, 18,  1, 13,  2, 16, 19,  8,  4, 13], device='cuda:0')
```


---

## Latent dim scheduling

1. Read through `joint-vae` code and implement similar feature in current codebase [@TODO].

---

## Task programming


1. Read <https://arxiv.org/abs/2011.13917> and decide where/how task programming would be useful [@TODO].


---

# Slides from last week



---

# Diff. Rule learning Progress

## Overview

Given a history vector $H = \{(s_0, a_0), (s_1, a_1), \dots (s_T)\}$​​​ and a large set of neural transformation rules $\Phi = \{\phi : (\R^\text{latent dim} \rightarrow{} \R^\text{latent dim}) \}$​​​, we want to learn a finite subset of neural transformation rules  $\Phi_r \subset \Phi$​​​.

Still using the shapeworld dataset:

![shapeworld_eg](/mnt/C:/Users/atharvas/Desktop/shapeworld_eg.png)

---

## Recap

The model from last time:
$$
\begin{align*}
\texttt{def}~&\texttt{architecture($H$, $\Phi_r$):}\\
Z &\leftarrow \texttt{encoder}(H) && Z : \mathbb{R}^{|H|\times \text{latent dim}}\\
I &\leftarrow \texttt{controller}(Z)  && I : |\Phi_r|^{|H|}\\
Z_t &\leftarrow \Phi_r[I](Z)&& Z_t : \mathbb{R}^{|H|\times \text{latent dim}}\\
H_{r} &\leftarrow \texttt{decoder}(Z) && H_r : \mathbb{R}^{|H| \times \text{state dim}}
\end{align*}
$$
where:

* $H$ is the history vector  $\{(s_0, a_0), (s_1, a_1), \dots (s_{T-1}, a_{T-1})\}$

* $I$​ represents the index of the neural function selected by the controller.
* $Z_t$ is the latent state after it is transformed by the neural function.

---

The DARTS pseudocode remains unchanged other than $\alpha = \Phi_r$​​ :

![image-20210817195618967](C:/Users/atharvas/Desktop/image-20210817195618967.png)

---

## Methodology #1

The original dataset didn't have $\Phi$ available so I had to first generate $\Phi$ and then learn the $\texttt{controller}$. Training  
proceeded as follows:

---

1. Train the $\texttt{encoder}$​​​ and $\texttt{decoder}$​​​ module to generate and infer from a latent state $Z$.

   * Algorithm:
     $$
     \begin{align*}
     \texttt{def}~&\texttt{latent\_learner($H$, $\Phi_r$):}\\
     Z &\leftarrow \texttt{encoder}(H) && Z : \mathbb{R}^{|H|\times \text{latent dim}}\\
     H_{r}^\prime &\leftarrow \texttt{decoder}(S) && H_r^\prime : \mathbb{R}^{|H| \times \text{state dim}}
     \end{align*}
     $$
     Notes: The training objective is to reconstruct $H$ so $H_r^\prime$ is a reconstruction of $H$, not $H[1:]$. Used `MSELoss` and trained for 50 epochs. $|H_{\text{val}}| = \R^{200\times100\times\text{input dim}}$, $|H_{\text{train}}| = \R^{1000\times100\times\text{input dim}}$ , $|H_{\text{test}}| = \R^{10000\times10\times\text{input dim}}$​​.


---

2. Use learned $Z$ to learn the transition rules:

   * Algorithm:
     $$
     \begin{align*}
     \texttt{def}~&\texttt{fn\_learner($H$, $\Phi_r$):}\\
     Z &\leftarrow \texttt{encoder}(H) && Z : \mathbb{R}^{|H|\times \text{latent dim}}\\
     Z_t &\leftarrow \Phi_r[H_A](Z)&& Z_t : \mathbb{R}^{|H|\times \text{latent dim}}\\
     H_{r} &\leftarrow \texttt{decoder}(S) && H_r : \mathbb{R}^{|H| \times \text{state dim}}
     \end{align*}
     $$
     Notes: $H_A$​ is the action taken that transforms $H^i$​ to $H^{i+1}$​. The weights of the $\texttt{encoder}$​ and $\texttt{decoder}$ are frozen.​

---

3. Constructed $\Phi = \Phi_r + \text{random functions}$ and learned controller:
   $$
   \begin{align*}
   \texttt{def}~&\texttt{controller\_learner($H$, $\Phi_r$):}\\
   Z &\leftarrow \texttt{encoder}(H) && Z : \mathbb{R}^{|H|\times \text{latent dim}}\\
   I &\leftarrow \texttt{controller}(Z)  && I : |\Phi_r|^{|H|}\\
   Z_t &\leftarrow \Phi_r[I](Z)&& Z_t : \mathbb{R}^{|H|\times \text{latent dim}}\\
   H_{r} &\leftarrow \texttt{decoder}(S) && H_r : \mathbb{R}^{|H| \times \text{state dim}}
   \end{align*}
   $$

---

## Issues with Methodology #1

1. In step 1, the model failed to converge. The loss value differed if the dataset was shuffled with different seeds. To me, this signified that the model got stuck in a local minima.

2. In step 2, again, the model failed to converge past a certain loss value.

3. In step 3, the controller always selected one neural function regardless of the input.

   ```python
   controller output -> tensor([3, 3, 3, 3, 3, 3, 3, 3, 3, 3], device='cuda:0')
   actual action     -> tensor([24, 18,  1, 13,  2, 16, 19,  8,  4, 13], device='cuda:0')
   ```
---

## Methodology #2 - Use a modified VAE

I tried to learn a VAE on the data instead. Training  proceeded as follows:

---

1. Train the $\texttt{encoder}$​​​​, $\texttt{decoder}$, $\texttt{mu}$ and $\texttt{var}$ modules to generate and infer from a latent state $Z$​.

   * Algorithm:
     $$
     \begin{align*}
     \texttt{def}~&\texttt{latent\_learner($H$, $\Phi_r$):}\\
     X &\leftarrow \texttt{encoder}(H) && X : \mathbb{R}^{|H|\times \text{latent dim}}\\
     \mu &\leftarrow \texttt{mu}(X) && \mu : \mathbb{R}^{|H|\times \text{latent dim}}\\
     \sigma &\leftarrow \texttt{var}(X) && \sigma : \mathbb{R}^{|H|\times \text{latent dim}}\\
     Z &\leftarrow \texttt{sample}(\mu, \sigma) && Z : \mathbb{R}^{|H|\times \text{latent dim}}\\
     H_{r}^\prime &\leftarrow \texttt{decoder}(Z) && H_r^\prime : \mathbb{R}^{|H| \times \text{state dim}}
     \end{align*}
     $$
     Notes: Used `MSELoss` and KL  divergence for optimization.

---

2. Use learned $Z$ to learn the transition rules:

   * Algorithm:
     $$
     \begin{align*}
     \texttt{def}~&\texttt{fn\_learner($H$, $\Phi_r$):}\\
     X &\leftarrow \texttt{encoder}(H) && X : \mathbb{R}^{|H|\times \text{latent dim}}\\
     X_t &\leftarrow \Phi_r[H_A](X)&& X_t : \mathbb{R}^{|H|\times \text{latent dim}}\\
     \mu &\leftarrow \texttt{mu}(X_t) && \mu : \mathbb{R}^{|H|\times \text{latent dim}}\\
     \sigma &\leftarrow \texttt{var}(X_t) && \sigma : \mathbb{R}^{|H|\times \text{latent dim}}\\
     Z &\leftarrow \texttt{sample}(\mu, \sigma) && Z : \mathbb{R}^{|H|\times \text{latent dim}}\\
     H_{r}^\prime &\leftarrow \texttt{decoder}(Z) && H_r^\prime : \mathbb{R}^{|H| \times \text{state dim}}
     \end{align*}
     $$
     Notes: $H_A$​ is the action taken that transforms $H^i$​ to $H^{i+1}$​. The weights of the $\texttt{encoder}$​ and $\texttt{decoder}$ are frozen.​
---

3. Constructed $\Phi = \Phi_r + \text{random functions}$ and learned controller:
   $$
   \begin{align*}
   \texttt{def}~&\texttt{controller\_learner($H$, $\Phi_r$):}\\
   X &\leftarrow \texttt{encoder}(H) && X : \mathbb{R}^{|H|\times \text{latent dim}}\\
   I &\leftarrow \texttt{controller}(X)  && I : |\Phi_r|^{|H|}\\
   X_t &\leftarrow \Phi_r[I](X)&& X_t : \mathbb{R}^{|H|\times \text{latent dim}}\\
   \mu &\leftarrow \texttt{mu}(X_t) && \mu : \mathbb{R}^{|H|\times \text{latent dim}}\\
   \sigma &\leftarrow \texttt{var}(X_t) && \sigma : \mathbb{R}^{|H|\times \text{latent dim}}\\
   Z &\leftarrow \texttt{sample}(\mu, \sigma) && Z : \mathbb{R}^{|H|\times \text{latent dim}}\\
   H_{r}^\prime &\leftarrow \texttt{decoder}(Z) && H_r^\prime : \mathbb{R}^{|H| \times \text{state dim}}\end{align*}
   $$

---

## Issues with Methodology #2

1. Model converged successfully in Step 1.

2. Model converged successfully in Step 2. I verified the learned $\Phi_r$ using simple tests like:

   ```python
   fn_obj0_north(fn_obj0_south(X)) ?= fn_obj0_stay(X)
   fn_obj0_north(fn_obj0_west(X)) ?!= fn_obj0_stay(X)
   ...
   ```

3. However, In step 3, the controller always selected one neural function regardless of the input.

   ```python
   controller output -> tensor([4, 4, 4, 4, 4, 4, 4, 4, 4, 4], device='cuda:0')
   actual action     -> tensor([24, 18,  1, 13,  2, 16, 19,  8,  4, 13], device='cuda:0')
   ```

---
