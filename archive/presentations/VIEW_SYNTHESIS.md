---
marp: true
author: atharvas
auto-scaling: true
---

# Multi-Plane Program Induction with 3D Box Priors
Yikai Li*, Jiayuan Mao*, Xiuming Zhang, William T. Freeman, Joshua B. Tenenbaum, Noah Snavely, Jiajun Wu

---
<!-- 
## Overview
* Motivation
* Problem definition
* Algorithm
* Experiments
* Results
- Discussion: 
    - Does this approach solve the problem?
    - Significance for Computer Vision
    - Significance for Program Synthesis

--- -->

<style>
 .row {
  display: flex;
}

.column {
  flex: 90%;
}
</style>


<div class="row">
<div class="column">

## Motivation

- Task: Single Image View Synthesis
- Challenge: Learning scene decompositions is, generally, underspecified.
- Observation: Physical objects display a measure of uniformity.
- Hypothesis: Can we impose a structural prior to capture this uniformity?
- Use a "box prior"
    - Pinhole camera
    - $\exists \text{ 4 planes if inside else 2 planes}$
    - Each plane has repeating pattern.

</div>

<div class="column">

![image2](VIEW_SYNTHESIS_images/snapshot.jpg)

![image](https://production-media.paperswithcode.com/tasks/illlustration_nvs.001_Fp0QwJz.png)


</div>

</div>


---

## Sample Box Programs

<style>
img[alt~="center"] {
  display: block;
  margin: 0 auto;
}
</style>


![w:620 center](VIEW_SYNTHESIS_images/boxprior_ss.png)
![w:620 center](VIEW_SYNTHESIS_images/boxprior_dsl.png)

---

## Algorithm

![w:900 center](VIEW_SYNTHESIS_images/algorithm.png)

---

## Step 3 - Plane Recticiation

![w:720 center](VIEW_SYNTHESIS_images/boxprior_math.png)


---

### Step 4 - Fitness Ranking

![w:900 center](VIEW_SYNTHESIS_images/fitness.png)


---

## Evaluation

<div class="row">
<div class="column">

**Dataset**
- 44 Corridor images, 42 Buildings.
- Collected by Google Images
- Handmade mask for end of the corridor
- Handmade mask for building.

**Tasks**
- Plane Segmentation
- Image Inpainting
- Image Extrapolation
- View Synthesis

</div>
<div class="column">

![w:900 center](VIEW_SYNTHESIS_images/corridors.png)

</div>
</div>

---

### Plane Segmentation (Quantitiave Evaluation)

![center](VIEW_SYNTHESIS_images/mm2.png)
![center](VIEW_SYNTHESIS_images/mm1.png)

---

### Image Inpainting (Quantitiave Evaluation)

<div class="row">
<div class="column">


![w:600 center](VIEW_SYNTHESIS_images/mm4.png)

</div>
<div class="column">

![center](VIEW_SYNTHESIS_images/mm3.png)

</div>
</div>

---

### Image Extrapolation (Qualitative Evaluation)

![center](VIEW_SYNTHESIS_images/mm5.png)

Randomly select 15 images and ask 15 participants to rank outputs.
- `61%` BPI (This Paper)
- `16%` Context-Aware Scaling
- `2%` Kaskar et al.
- `16%` InGAN
- `5%` Huang'14 

---

### View Synthesis (Qualitative Evaluation)

![center](VIEW_SYNTHESIS_images/mm6.png)

Randomly select 20 images from 3 trajectories and ask 10 participants to rank outputs.
- `100%` prefer BPI on Trajectory 1
- `94%` prefer BPI on Trajectory 2
- `99.5%` prefer BPI on Trajectory 3


---

### Failure Cases

![center](VIEW_SYNTHESIS_images/mm7.png)


---

### Failure Cases (?)

![h:600 center](VIEW_SYNTHESIS_images/our.jpg)

<!-- 

## Problem Definition

- 
- For box priors: given an input image, we want to setment the image into different planes, estimate their surface normals relative to the camera and infer the regular patterns.

 -->


---

## Interesting papers

 - Perspective plane program induction framework
 - The unreasonable Effectiveness of Deep networks as a Perceptual Metric.
