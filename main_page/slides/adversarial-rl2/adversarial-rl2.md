---
marp: true
theme: gaia
class: invert
paginate: true
style: |
  section {
    font-size: 26px;
  }
  h1 {
    font-size: 38px;
    color: #58a6ff;
  }
  h2 {
    font-size: 34px;
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
    font-size: 14px;
  }
  th, td {
    padding: 3px 6px;
  }
  .small-table table {
    font-size: 12px;
  }
math: mathjax
---

# Adversarial RL for Robust Policy Learning
## Literature Review & Proposed Method

Melik Bugra Ozcelik
Master's Thesis

---

## Overview

1. Introduction to Adversarial RL
2. Baseline Method: Value-Based ARL (Laux et al. 2020)
3. Proposed Method: Value + Uncertainty ARL
4. Literature Review Categories
5. Comparative Analysis
6. Novelty Assessment
7. Gap Analysis
8. Results (TBD)

---

## What is Adversarial Reinforcement Learning?

- **Two-agent game**: Protagonist vs Adversary
- **Goal**: Train robust policies that generalize beyond standard training conditions
- Adversary steers protagonist into challenging states
- Protagonist learns to handle worst-case scenarios

**Adversary Types:**
1. **Internal**: Applying disturbance/perturbation to actions (RARL)
2. **External**:
   - Separate agent takes protagonist to challenging states
   - Generates environment/initial conditions (PAIRED)

---

## Two-Phase Training Framework

$$
\begin{aligned}
& \text{for } i = 1 \text{ to } N: \\
& \quad \textbf{// Phase 1: Adversary Training} \\
& \quad \text{for } j = 1 \text{ to } K_a: \\
& \quad \quad \text{Adversary acts (stochastic) for } H_a \text{ steps} \\
& \quad \quad \text{Protagonist acts (greedy) for } H_p \text{ steps} \\
\\
& \quad \textbf{// Phase 2: Protagonist Training} \\
& \quad \text{for } j = 1 \text{ to } K_p: \\
& \quad \quad \text{Adversary acts (greedy) for } H_a \text{ steps} \\
& \quad \quad \text{Protagonist acts (stochastic) for } H_p \text{ steps and trains}
\end{aligned}
$$

- Episode truncates at $(H_a + H_p)$ steps
- Both agents use SAC (Soft Actor-Critic)

---

## Baseline Method: Value-Based Adversary Reward

### Laux et al. (2020) - IROS

Adversary reward is the **negative of protagonist's value function**:

$$r_A(s, a, s') = -\gamma_v \cdot \hat{V}_P(s')$$

**Soft Value Estimation (Monte Carlo):**

$$\hat{V}_P(s) \approx \frac{1}{K} \sum_{k=1}^{K} \left[ Q_P(s, a_k) - \alpha \log \pi_P(a_k|s) \right], \quad a_k \sim \pi_P(\cdot|s)$$

**Intuition:** Push protagonist to low-value (difficult) states

**Limitation:** Only exploits *known* weaknesses, ignores *unknown* regions

---

## Proposed Method: Value + Uncertainty ARL

### Our Main Contribution

$$r_A(s, a, s') = \underbrace{-\lambda_v(t) \cdot z_V(\hat{V}_P(s'))}_{\text{Value Suppression}} + \underbrace{\lambda_\sigma(t) \cdot z_\sigma^+(\text{CV}_Q(s'))}_{\text{Uncertainty Exploitation}}$$

**Two Components:**
1. **Value Suppression**: Push to low-value states (like baseline)
2. **Uncertainty Exploitation**: Push to high-uncertainty states (novel)

**Key Innovations:**
- Q-ensemble (6 heads) for uncertainty estimation
- Z-score normalization for stable training
- Dynamic lambda scheduling

---

## Component 1: Value Suppression

$$r_{\text{value}} = -\lambda_v(t) \cdot z_V(\hat{V}_P(s'))$$

**Running Z-Score Normalization:**
$$z_V(v) = \frac{v - \mu_V}{\sqrt{\sigma_V^2 + \epsilon}}$$

**Exponential Moving Average Updates:**
$$\mu_V^{(t)} = \beta \cdot \mu_V^{(t-1)} + (1-\beta) \cdot v_t$$
$$\sigma_V^{2(t)} = \beta \cdot \sigma_V^{2(t-1)} + (1-\beta) \cdot (v_t - \mu_V^{(t)})^2$$

**Why normalize?** Keeps reward signals in stable range as protagonist learns

---

## Component 2: Uncertainty Exploitation

$$r_{\text{uncertainty}} = \lambda_\sigma(t) \cdot z_\sigma^+(\text{CV}_Q(s'))$$

**Coefficient of Variation (Scale-Invariant Uncertainty):**
$$\text{CV}_Q(s') = \frac{\sigma_Q(s')}{\sqrt{|\bar{Q}(s')|^2 + c^2}}$$

- $\sigma_Q$ = Standard deviation across Q-heads
- $\bar{Q}$ = Mean Q-value
- $c = 5.0$ (stabilization constant)

**Positive Z-Score (Only reward above-average uncertainty):**
$$z_\sigma^+(x) = \min\left( \max\left(0, \frac{x - \mu_\sigma}{\sigma_\sigma} - m \right), \tau \right)$$

---

## Dynamic Lambda Scheduling

<div class="columns">
<div>

**Value Weight (Exponential Ramp-up):**
$$\lambda_v(t) = \lambda_v^{\text{base}} \cdot \left(1 - e^{-t/T_{\text{ramp}}}\right)$$

- Starts low, increases over time
- Avoids trusting unreliable early value estimates

</div>
<div>

**Sigma Weight (Warmup + Decay):**
$$\lambda_\sigma(t) = \begin{cases}
\lambda_\sigma^{\text{base}} & t < T_{\text{warmup}} \\
\lambda_\sigma^{\text{base}} \cdot \left(\eta + (1-\eta) \cdot e^{-(t-T_w)/T_d}\right) & \text{else}
\end{cases}$$

- High early (exploration phase)
- Decays later (exploitation phase)

</div>
</div>

**Curriculum Effect:** Exploration → Exploitation progression

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| $\lambda_v^{\text{base}}$ | 1.0 | Base value weight |
| $\lambda_\sigma^{\text{base}}$ | 0.35 | Base uncertainty weight |
| $T_{\text{ramp}}$ | 0.2 × Total Steps | Value ramp-up period |
| $T_{\text{warmup}}$ | 0.5 × Total Steps | Sigma warmup period |
| $T_{\text{decay}}$ | 0.4 × Total Steps | Sigma decay period |
| $\beta$ (EMA decay) | 0.99 / 0.995 | Running stats decay |
| $K$ (samples) | 8 | Action samples for V estimation |
| Q-heads | 6 (protagonist) / 2 (adversary) | Ensemble size |

---

## Literature Review: Categories

1. **Adversarial Reinforcement Learning**
   - RARL (Pinto 2017), Laux et al. (2020), Gleave et al. (2020)

2. **Robust RL and Policy Robustness**
   - Worst-case optimization, Domain randomization, EPOpt

3. **Uncertainty Quantification in RL**
   - Bootstrapped DQN, SUNRISE, REDQ

4. **Intrinsic Motivation and Curiosity**
   - ICM, RND, Disagreement-based exploration

5. **Curriculum Learning and Self-Play**
   - PAIRED, Asymmetric Self-Play, Goal GAN

---

## 1. Adversarial RL Literature

### RARL (Pinto et al., 2017) - ICML
- Adversary injects **force perturbations** during episodes
- Zero-sum: $r_A = -r_P$
- Outperformed domain randomization

### Laux et al. (2020) - IROS
- Adversary generates **initial states** via value function
- Two-phase SAC training
- **Our baseline method**

### PAIRED (Dennis et al., 2020) - NeurIPS
- Three agents: Teacher, Protagonist, Antagonist
- Maximizes **regret** between protagonist and antagonist
- Ensures task solvability

---

## 2. Robust RL Literature

### Domain Randomization vs Adversarial Training

| Aspect | Domain Randomization | Adversarial Training |
|--------|---------------------|---------------------|
| Approach | Random perturbations | Learned worst-cases |
| Focus | Breadth (coverage) | Depth (worst-case) |
| Adaptivity | Static distribution | Adapts to agent |
| Efficiency | May waste time on easy cases | Concentrated stress tests |

### EPOpt (Rajeswaran et al., 2017)
- Selects **worst-performing** environments for training
- Implicit adversarial filtering

---

## 3. Uncertainty Quantification

### Ensemble Methods for Epistemic Uncertainty

**Bootstrapped DQN (Osband et al., 2016)**
- Multiple Q-network heads
- Disagreement signals high uncertainty
- Thompson sampling for exploration

**REDQ (Chen et al., 2021)**
- 10 Q-networks for efficient SAC
- High update-to-data ratio
- State-of-the-art sample efficiency

**SUNRISE (Lee et al., 2021)**
- Q-variance for experience re-weighting
- UCB-style exploration bonus

---

## 4. Curiosity-Driven Exploration

### Disagreement-Based (Pathak et al., 2019)
- Ensemble of forward dynamics models
- **Intrinsic reward = model disagreement**
- No RL needed - direct policy optimization

### RND (Burda et al., 2018)
- Random Network Distillation
- Prediction error as curiosity signal
- Large-scale study across 54 games

**Connection to Our Work:**
- We use Q-ensemble disagreement as **adversary reward**
- "External curiosity" - adversary pushes protagonist to uncertain states

---

## 5. Curriculum Learning & Self-Play

### Asymmetric Self-Play (Sukhbaatar et al., 2018)
- Alice sets tasks, Bob solves them
- Alice rewarded when Bob fails (but tasks must be achievable)
- Emergent curriculum of increasing difficulty

### Goal GAN (Florensa et al., 2018)
- GAN generates goals of appropriate difficulty
- Discriminator ensures ~50% success rate
- Automatic curriculum at competence boundary

**Our Method as Curriculum:**
- Dynamic $\lambda$ scheduling = explicit curriculum
- Early: exploration (uncertainty bonus high)
- Late: exploitation (value term dominant)

---

## Comparative Analysis: 10 Key Methods

<div class="small-table">

| Feature | **Ours** | Laux 2020 | RARL | RARARL | PAIRED |
|---------|----------|-----------|------|--------|--------|
| Adversary Role | Start states | Start states | Force perturbation | Env parameters | Env generation |
| Adv. Objective | Value + Uncertainty | $-V_P(s')$ | $-r_P$ | Risk (variance) | Regret |
| Uncertainty | Q-ensemble (6) | None | None | Q-ensemble | Implicit |
| Normalization | Z-score | None | None | Partial | None |
| Scheduling | Dynamic $\lambda$ | Static | Static | Partial | Emergent |

</div>

---

## Comparative Analysis (Continued)

<div class="small-table">

| Feature | ASP | Pathak 2019 | Burda 2018 | Goal GAN | REDQ |
|---------|-----|-------------|------------|----------|------|
| Role | Task generation | Exploration | Exploration | Goal generation | Single-agent |
| Objective | Bob fails | Disagreement | Prediction error | 50% success | Efficient SAC |
| Uncertainty | None | Model ensemble | Prediction error | Implicit | Q-ensemble |
| Framework | Self-play | Single-agent | Single-agent | GAN + Agent | Single-agent |

</div>

**Key Insight:** No prior work combines:
- Two-agent adversarial training
- Q-ensemble uncertainty in adversary reward
- Z-score normalization
- Dynamic scheduling

---

## Novelty Assessment

### What is Novel

1. **Value + Uncertainty Adversary Reward**
   - First to combine both signals in adversary objective
   - Adversary with "built-in curiosity"

2. **Z-Score Normalization for Adversary Rewards**
   - Adaptive scaling for stable long training
   - Not found in prior ARL papers

3. **Dynamic Weight Scheduling**
   - Explicit curriculum in adversary objective
   - Only UACER (2025) has similar concept

4. **6-Head Q-Ensemble in ARL Context**
   - Larger ensemble than typical ARL (1-2 critics)

---

## Novelty Assessment (Continued)

### What is NOT Novel (Building on Prior Art)

- Two-phase adversarial training loop (Laux et al.)
- Negative value function as adversary reward (Laux et al.)
- Q-ensembles for exploration (Osband, Pathak)
- SAC algorithm and double-Q trick

### Positioning

Our method is a **natural extension of Laux et al. (2020)**:
- Addresses limitation of value-only approach
- Adds exploration-driven component
- Implements practical innovations for stability

---

## Gap Analysis

### Gap 1: Adversarial Curricula Lacking Exploration
- Prior methods target only *known* difficult states
- Miss states agent hasn't experienced
- **Our solution:** Uncertainty term finds novel states

### Gap 2: Unstable Adversarial Training
- Reward scales change as protagonist learns
- Non-stationarity issues (PAIRED noted this)
- **Our solution:** Z-score normalization + scheduling

### Gap 3: Robustness vs Exploration Trade-off
- DR: breadth but not targeted
- ARL: depth but not exploratory
- **Our solution:** Combine both via uncertainty reward

---

## Gap Analysis (Continued)

### Gap 4: Practical ARL Hyperparameter Guidance
- No established best practices for weighting
- When to explore vs exploit?
- **Our solution:** Concrete schedule + empirical guidance

### Gap 5: Evaluation Metrics for ARL
- Most work shows only final success rates
- Little analysis of adversary behavior evolution
- **Our contribution:** Heatmap analysis of adversary endpoints

### Remaining Gap
- No theoretical guarantee of task solvability
- PAIRED uses antagonist, ASP uses reversibility
- We rely on empirics + penalty for invalid states

---

## Method Comparison: Baseline vs Proposed

| Feature | Baseline (Value-Only) | Proposed (Value + Uncertainty) |
|---------|----------------------|-------------------------------|
| Reward components | Single (value) | Dual (value + uncertainty) |
| Normalization | None (raw scale) | Z-score normalization |
| Lambda scheduling | Static | Dynamic (ramp-up, warmup, decay) |
| Q-heads | 2 | 6 |
| Target states | Low value | Low value AND high uncertainty |
| Hypothesis | Push to weak states | Push to weak AND unknown states |

---

## Key Related Work: UACER (Zhang et al., 2025)

### Recent Concurrent Work

- **Uncertainty-Aware Critic Ensemble for Robust ARL**
- Similar philosophy: ensemble critics + decaying uncertainty
- Time-Decaying Uncertainty (TDU) aggregation

**Differences from Our Work:**
- UACER: uncertainty in protagonist's update
- Ours: uncertainty in adversary's reward
- Different applications but converging ideas

**Validation:** Independent discovery of similar concepts confirms relevance

---

## Essential References

1. **Pinto et al. (2017)** - RARL, ICML
2. **Laux et al. (2020)** - Value-based ARL, IROS
3. **Dennis et al. (2020)** - PAIRED, NeurIPS
4. **Pathak et al. (2019)** - Disagreement exploration, ICML
5. **Burda et al. (2018)** - Curiosity study, OpenAI
6. **Sukhbaatar et al. (2018)** - Asymmetric Self-Play, ICLR
7. **Pan et al. (2019)** - Risk Averse RARL
8. **Osband et al. (2016)** - Bootstrapped DQN, NIPS
9. **Chen et al. (2021)** - REDQ, ICLR
10. **Zhang et al. (2025)** - UACER

---

## Results

<!-- Results section - to be filled with experimental data -->

*Results will be added here*

---

## Summary

### Main Contributions

1. **Novel adversary reward**: Value suppression + Uncertainty exploitation
2. **Practical innovations**: Z-score normalization, dynamic scheduling
3. **Extension of Laux et al.**: Addresses value-only limitations
4. **Synthesis of ideas**: ARL + Curiosity + Curriculum learning

### Expected Outcomes

- Improved robustness to unseen start states
- Better generalization across maze layouts
- More stable training dynamics

---

# Thank You!

Questions?

---

## Appendix: Full Comparison Table

<div class="small-table">

| Feature | Ours | Laux | RARL | RARARL | PAIRED | ASP | Pathak | Burda | GoalGAN | REDQ |
|---------|------|------|------|--------|--------|-----|--------|-------|---------|------|
| Adv. Role | States | States | Forces | Params | Env | Tasks | - | - | Goals | - |
| Framework | 2-agent | 2-agent | 2-agent | 2-agent | 3-agent | Self-play | Single | Single | GAN | Single |
| Uncertainty | Q-ens. | No | No | Q-ens. | Implicit | No | Model ens. | Pred. err | No | Q-ens. |
| Normalization | Z-score | No | No | Partial | No | No | Partial | Partial | GAN | No |
| Scheduling | Yes | No | No | Partial | Emergent | Emergent | Natural | Partial | Yes | No |

</div>

