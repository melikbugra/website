---
marp: true
theme: gaia
class: invert
paginate: true
style: |
  section {
    font-size: 28px;
  }
  h1 {
    font-size: 40px;
    color: #58a6ff;
  }
  code {
    background-color: #2d333b;
  }
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  table {
    font-size: 18px;
  }
  th, td {
    padding: 4px 8px;
  }
math: mathjax
---
<style>
img[alt~="center"] {
  position: absolute;
  top: 130px;
  right: 130px;
}
</style>

<style>
img[alt~="center-plot"] {
  position: absolute;
  top: 300px;
  right: 130px;
}
</style>

<style>
img[alt~="top-right"] {
  position: absolute;
  top: 120px;
  right: 30px;
}
</style>

<style>
.video-container {
  position: absolute;
  top: 180px;
  left: 50%;
  transform: translateX(-50%);
  width: 800px;
}
</style>

# Adversarial Reinforcement Learning

Melik Buğra Özçelik
Date: -

---

## Overview

- Introduction to Adversarial RL
- Key Concepts
- Related Work
- Proposed Idea/Method
- First Results
- Novelty Discussion

---

## Introduction

- What is Adversarial Reinforcement Learning?
- Adversary Types:
    1. Internal
      - Applying disturbance/perturbation to the actions
    2. External
      - Separate adversary agent takes the protagonist to a challenging state
      - Separate adversary agent creates the state (Procedural content generation)

---



## Key Concepts

- Continuous Maze (Toy Problem)
- Adversarial SAC
- Adversary Rewarding
- Reward Scaling by Protagonist Entropy
- Stirring (Target Problem)

<!-- Image showing continuous maze environment -->
![width:600px top-right](continuous_maze.png)

---

## Related Work

<!-- Put a link here  -->
- [Robust Adversarial Reinforcement Learning](https://www.notion.so/Robust-Adversarial-Reinforcement-Learning-15d60a9d769380429425e3ba15268804)
- [Deep Adversarial Reinforcement Learning for Object Disentangling](https://www.notion.so/Deep-Adversarial-Reinforcement-Learning-for-Object-Disentangling-14e60a9d76938030886ce9291fe70400)
- [Adversarial Reinforcement Learning for Procedural Content Generation](https://www.notion.so/Adversarial-Reinforcement-Learning-for-Procedural-Content-Generation-14e60a9d769380069203e486aabe59a2)
- [Robust Reinforcement Learning using Adversarial Populations](https://www.notion.so/Robust-Reinforcement-Learning-using-Adversarial-Populations-14e60a9d76938064a89be0bf9eaf20e5)
- [Deep Adversarial Reinforcement Learning With Noise Compensation by Autoencoder](https://www.notion.so/Deep-Adversarial-Reinforcement-Learning-With-Noise-Compensation-by-Autoencoder-14f60a9d769380b785dcfb91092140db)
- [Adversarial Skill Learning for Robust Manipulation](https://www.notion.so/Adversarial-Skill-Learning-for-Robust-Manipulation-15d60a9d76938052bda7e16338cfd4db)
- [(Entropy Usage)Explore and Control with Adversarial Surprise](https://www.notion.so/Explore-and-Control-with-Adversarial-Surprise-1c460a9d76938069bcb6d1c95e34ba3e)
---

## Main Training Algorithm
$$
\begin{aligned}
& \text{// Main training loop} \\
& \text{for } i = 1 \text{ to } N: \\
& \quad \text{// Train adversary} \\
& \quad \text{for } j = 1 \text{ to } K_a: \\
& \quad \quad \text{Let adversary act for $H_a$ steps and train}  \\
& \quad \quad \text{Let protagonist act (greedy policy) for $H_p$ steps} \\
\\
& \quad \text{// Train protagonist} \\
& \quad \text{for } j = 1 \text{ to } K_p: \\
& \quad \quad \text{Let adversary act (greedy policy) for $H_a$ steps} \\
& \quad \quad \text{Let protagonist act for $H_p$ steps and train}
\end{aligned}
$$

- where an episode truncates at ($H_a + H_p$)th step.
- N is increased by 50 for every 10 iterations
---

## Rewarding by Value from Protagonist

- Reward for each step is the value of that step from protagonist
$$
r_{\text{adv}}(s_t, s_{t+1}) = V_{\text{prt}}(s_{t+1})
$$

- Unstable training of adversary until the critic network of protanist is trained enough to make reasonable estimations
- Pre-training of protagonist improves the performance

---

## Lazy Rewarding by Protagonist Return

- Reward of the last transition is the return of the protagonist
- All the other transtions' rewards are 0
$$
r_{\text{adv}}(s_t, s_{t+1}) = 
\begin{cases} 
G_{\text{prt}} & \text{if } t = H_a-1 \\
0 & \text{otherwise}
\end{cases}
$$

- Sparse rewarding of adversary caused unstabilities

---

## Lazy Dense Rewarding by Protagonist Return

- Reward of each transition is the equal share from return of the protagonist, that return is divided by total time steps played by protagonist
$$
r_{\text{adv}}(s_t, s_{t+1}) = \frac{G_{\text{prt}}}{T_{\text{prt}}}
$$

- Usually gets stuck at local maximas

---

## Rewarding Scaling by Entropy

- At each time step, the reward of adversary is scaled by the entropy value of the protagonist, taking N examples

$$
\begin{aligned}
\bar{V} &= \frac{1}{N} \sum_{i=1}^{N} V_i \\
\bar{H} &= \frac{1}{N} \sum_{i=1}^{N} H_i \\
H_{factor} &= \text{clamp}(\bar{H}, 0.1, 5.0) \\
r_{adv} &= -\frac{\bar{V}}{\beta \cdot H_{factor}}
\end{aligned}
$$

- Where $\bar{V}$ is the mean estimated value, $\bar{H}$ is the mean entropy, $\beta$ is the entropy scaling coefficient
- Higher entropy (more uncertainty) reduces the impact of negative value, encouraging exploration

---

## Using the Change of the Value as a Reward

- Reward of each transition it the value change between the steps of the transition

$$
\begin{aligned}
v_{prev}, \_ &= \text{estimate\_v\_and\_entropy}(s_{prev}) \\
v_{curr}, entropy &= \text{estimate\_v\_and\_entropy}(s_{curr}) \\
\Delta v &= (v_{prev} - v_{curr}) \\
H_{factor} &= \text{clamp}(\bar{H}, 0.1, 5.0) \\
scale &= \frac{\beta}{\beta + H_{factor}} \\
r_{adv} &= \Delta v \cdot scale
\end{aligned}
$$

---

<!-- Code sample with syntax highlighting -->
```python
def get_adversary_reward(
    prev_state: np.ndarray,
    state: np.ndarray,
    protagonist: SAC,
    num_samples: int = 10,
    beta: float = 1.0,
):
    ### REST OF THE CODE
    s_prev = torch.tensor(prev_state, dtype=torch.float32, device=device).unsqueeze(0)
    s_curr = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    v_prev, _ = estimate_v_and_entropy(s_prev)
    v_curr, entropy = estimate_v_and_entropy(s_curr)

    # ΔV = V(s_prev) − V(s_curr)
    delta_v = (v_prev - v_curr).squeeze(0)

    entropy_factor = torch.clamp(entropy, min=0.1, max=5.0).squeeze(0)
    scale = beta / (beta + entropy_factor)

    scaled_reward = (delta_v * scale).cpu().item()
    return scaled_reward

```

---

## Evaluation

![width:1000px center](checkpoint5.gif)

---

## Evaluation

![width:1000px center](checkpoint10.gif)

---

## Evaluation

![width:1000px center](checkpoint25.gif)

---
## Evaluation

![width:1000px center](checkpoint45.gif)

---
## Evaluation

![width:1000px center](checkpoint130.gif)

---
## Evaluation

![width:1000px center](checkpoint185.gif)

---
## Episode Returns while Adversary Training

![width:1000px center-plot](adv_train.png)

---
## Episode Returns while Protagonist Training

![width:1000px center-plot](prt_train.png)

---

<!-- Split content -->
<div class="columns">
<div>

## Current Progress

- We have our own SAC (in the [rl-baselines](https://github.com/melikbugra?tab=repositories) library) implementation
- We have our own environment [ContinuousMaze-v0](https://github.com/melikbugra/continuous-maze-env)
- We have a couple of methods proposed
- Reward scaling by entropy and using the value change as reward seems to be novel to the literature

</div>
<div>

## Next Steps

- Applying one (or some) of the methods to the robot stirring problem
- Finalizing the literature review
- Depending on the novelty of the idea, choosing an appropriate conference
- Final trainings and preparing the paper

</div>
</div>

---

## Novel Idea 1: Adversary Rewarded by the Change in State-Value (ΔV)
$$
r_{\text{adv}}(s_{t-1}, s_t) = V(s_{t-1}) - V(s_t)
$$
- Explicitly defines the adversary’s immediate reward based on the difference in the protagonist's estimated state values between two consecutive states
- The adversary gets positively rewarded for actions that lead the protagonist from a higher-value state to a lower-value state
- Unlike cumulative-return-based or direct-value-based adversarial objectives, this explicitly rewards incremental "value degradation," potentially improving training stability and consistency of the adversary’s behavior

---

## Literature for idea 1: [Robust Adversarial Reinforcement Learning](https://arxiv.org/abs/1703.02702)

- Adversary explicitly tries to minimize protagonist’s cumulative return:
  - $\max_{\pi_p} \min_{\pi_a} \mathbb{E} \left[ \sum_{t=0}^{T} r(s_t, a_t^p, a_t^a) \right]$


  - **Similarity:**
    - Both ideas seek to reduce protagonist’s overall performance (value)
  - **Difference:**
    - Our approach uses explicit immediate ΔV per transition, RARL uses cumulative return
---

## Literature for idea 1: [Robust Deep Reinforcement Learning with Adversarial Attacks](https://arxiv.org/abs/1712.03632)

- Perturbations computed via gradients directly targeting protagonist’s Q-values, explicitly minimizing protagonist’s Q-value (similar intuition, different mechanism):
  - $\max_{\pi_p} \min_{\pi_a} \mathbb{E} \left[ \sum_{t=0}^{T} r(s_t, a_t^p, a_t^a) \right]$


  - **Similarity:**
    - Both ideas directly target protagonist's state values
  - **Difference:**
    - Our  method explicitly defines adversary's step-wise reward using ΔV, whereas this approach directly perturbs inputs to minimize Q-values (not explicit ΔV)
---
## Literature for idea 1: [Tactics of Adversarial Attack on Deep Reinforcement Learning Agents](https://arxiv.org/abs/1703.06748)

- Attacks exploit protagonist’s confident (low-entropy) action selection, forcing low-value outcomes:
  - $a_{\text{attack}}(s) = \arg\min_a \pi_p(a \mid s)$

  - **Similarity:**
    - Both ideas intuitively aim at decreasing protagonist’s value at confident moments
  - **Difference:**
    - No explicit immediate ΔV used; adversary picks worst action rather than explicitly rewarded by state-value differences
---
| Paper                                 | Reward formulation                                  | Immediate ΔV explicitly used?     | Similarity                                    | Difference                                       |                        |
| ------------------------------------- | --------------------------------------------------- | --------------------------------- | --------------------------------------------- | ------------------------------------------------ | ---------------------- |
| **RARL (Pinto et al.)**               | $\min_{\pi_a}\sum r(s,a)$ cumulative                | No                              | Both adversaries reduce protagonist’s returns | Uses cumulative returns, not immediate ΔV        |                        |
| **Gradient Attack (Pattanaik)**       | $\max_{\delta}-Q(s+\delta,a)$ direct Q-minimization | No                              | Targets protagonist’s Q-value directly        | Not immediate per-step ΔV reward                 |                        |
| **Strategically-timed Attacks (Lin)** | $\arg\min_a \pi_p(a \mid s)$,                              |  implicit worst-action choice | No                                          | Implicitly causes value drop at confident states | Not explicit ΔV reward |

---

## Novel Idea 2: Scaling Adversarial Reward by Protagonist’s Policy Entropy
$$
r_{\text{adv}}(s_{t-1}, s_t) = \Delta V(s_{t-1}, s_t) \cdot \frac{\beta}{\beta + H(\pi_{\text{p}}(\cdot \mid s_t))}
$$
- **Entropy** measures how uncertain the protagonist's policy is at state $s_t$
  - **Low entropy:** Protagonist is confident, adversarial reward increases
  - **High entropy:** Protagonist is uncertain, adversarial reward decreases
- This idea balances and stabilizes the training
---

## Literature for idea 2: [Tactics of Adversarial Attack on Deep Reinforcement Learning Agents](https://arxiv.org/abs/1703.06748)

- Instead of using entropy explicitly in the reward, they choose when to attack based on protagonist policy uncertainty:

  - **Similarity:**
    - Both ideas exploit protagonist's uncertainty/confidence
  - **Difference:**
    - They do not scale rewards explicitly by entropy; instead, their timing strategy implicitly leverages the same intuition (targeting low-entropy states)
---

## Literature for idea 2: [Deceptive Reinforcement Learning Under Adversarial Manipulations on Cost Signals](https://arxiv.org/abs/1906.10571)

<style scoped>
section {
  font-size: 22px;
}
</style>

- Instead of using entropy formula directly, they use Q value difference between actions for the same state:
  - $\text{trigger attack if } \max_{a, a'} \left| Q(s, a) - Q(s, a') \right| > \epsilon$

  - **Similarity:**
    - A large difference in Q-values (one action clearly better than others) means low entropy
    - Attacks are only triggered when protagonist is highly confident, so both out and their methods learns to drive the protagonist in to low entropy states
  - **Difference:**
    - No explicit entropy used; their "confidence measure" is Q-value spread rather than entropy
    - The adversary either fully triggers or doesn't (binary decision), while our idea smoothly scales reward by entropy
---

## Literature for idea 2: [Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design](https://arxiv.org/abs/2012.02096)
<style scoped>
section {
  font-size: 22px;
}
</style>
- An environment-designing adversary tries to maximize the *regret* of the protagonist compared to a second ("antagonist") agent:
  - $r_{adv} = \text{Regret} = V_{antagonist} - V_{protagonist}$

  - **Similarity:**
    - Both methods dynamically adjust adversarial pressure based on protagonist’s capability
    - "Regret" implicitly scales difficulty: if protagonist struggles significantly (high regret), adversary is rewarded greatly. This implicitly ensures the adversary chooses problems at the edge of protagonist ability (related to protagonist uncertainty/confidence)
  - **Difference:**
    - They use regret (performance gap) between two agents rather than entropy explicitly
    - Our method directly and explicitly utilizes entropy, giving clearer control
---

## Literature for idea 2: [Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula](https://arxiv.org/abs/2311.01642)
<style scoped>
section {
  font-size: 22px;
}
</style>
- They train the adversary with entropy regularization on the adversary’s policy itself:
  - $J(\pi) = \mathbb{E} \left[ \sum_t r_t - \tau H(\pi(\cdot \mid s_t)) \right]$
  - where $\tau$ is the temperature coefficient controlling the strength of the entropy term

  - **Similarity:**
    - Both methods explicitly use entropy, but QARL applies entropy to the adversary itself, controlling adversary "difficulty"
    - Both methods ensure adversarial pressure is adaptive: softer initially and increasing gradually as training progresses
  - **Difference:**
    - They scale by adversary's entropy, not protagonist’s
    - Our approach directly ties adversarial reward to protagonist’s confidence rather than controlling adversary’s policy softness
    
---

| Paper                    | Explicitly uses entropy?               | Whose entropy?                        | How entropy/confidence used?                                 | Similarity             | Difference             |
| ------------------------ | -------------------------------------- | ------------------------------------- | ------------------------------------------------------------ | --------------------------------------- | ----------------------------------------- |
| **Lin et al. (2017)**    | No (implicit)                        | Protagonist                           | Timing attacks when entropy low                              | Adversary impactful in confident states | No explicit scaling of reward             |
| **Huang & Zhu (2020)**   | No (implicit)                        | Protagonist (implicitly via Q-values) | Triggering attacks when large Q-gap                          | Adversary impactful in confident states | Binary trigger, no smooth entropy scaling |
| **Dennis et al. (2020)** | No (implicit via regret)             | Protagonist (implicitly via regret)   | Difficulty scales implicitly with protagonist ability        | Dynamic difficulty adaptation           | No explicit entropy measure               |
| **Reddi et al. (2024)**  | Yes (explicitly entropy-regularized) | Adversary                             | Gradually adjusts adversarial strength via adversary entropy | Adaptive adversary pressure             | Uses adversary entropy, not protagonist’s |


---

## Demo Video
<div class="video-container">
<video controls width="800">
  <source src="adv_rl.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</div>

---

# Thank You!

Questions?

