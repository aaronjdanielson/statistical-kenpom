Yes. The right way to do this is to stop thinking of “KenPom” as a branded ratings system and instead view it as a particular solution to a familiar statistical problem:

we observe noisy game-level outcomes on a graph of teams, and we want to infer latent offensive strength, defensive strength, and pace, after correcting for opponent quality, venue, and time.

From that point of view, his system is elegant but also quite classical. It is essentially a deterministic, iteratively reweighted, opponent-normalized empirical estimator. A statistician would immediately see the missing pieces: a likelihood, shrinkage, uncertainty, latent state evolution, and a clean generative interpretation.

I’ll do this in two parts:

1. write a mathematically explicit version of the classical KenPom-style system;
2. then write a more sophisticated model that preserves the spirit of the system while upgrading it into a modern statistical framework.

---

# 1. The data structure

Let teams be indexed by (i,j \in {1,\dots,T}), and games by (g \in {1,\dots,G}).

For each game (g), suppose team (i(g)) plays team (j(g)). Let:

* (y^{\text{pts}}_{ig}): points scored by team (i) in game (g),
* (y^{\text{pts}}_{jg}): points scored by team (j) in game (g),
* (p_g): estimated number of possessions in the game,
* (h_g \in {-1,0,1}): venue indicator from the perspective of team (i),
  where (1) = home, (0) = neutral, (-1) = away,
* (t_g): game date or time index.

Usually possessions are estimated by a box-score formula such as
[
p_g \approx \frac{1}{2}\Big[
(\mathrm{FGA}_i - \mathrm{ORB}_i + \mathrm{TO}_i + c,\mathrm{FTA}_i)
+
(\mathrm{FGA}_j - \mathrm{ORB}_j + \mathrm{TO}_j + c,\mathrm{FTA}_j)
\Big],
]
with (c \approx 0.475).

Then define raw game efficiencies:
[
e^{\text{off}}*{ig} = 100 \cdot \frac{y^{\text{pts}}*{ig}}{p_g},
\qquad
e^{\text{def}}*{ig} = 100 \cdot \frac{y^{\text{pts}}*{jg}}{p_g}.
]

So each game produces two efficiency observations for team (i): how well it scored, and how well it prevented scoring.

Let (\mu) denote the national average efficiency, in points per 100 possessions.

---

# 2. A clean mathematical reconstruction of the KenPom idea

## 2.1 Latent team strengths

Introduce latent team parameters:

* (O_i): offensive strength of team (i),
* (D_i): defensive strength of team (i),
* (P_i): pace tendency of team (i).

In the classical KenPom-style worldview, these are not estimated by a formal likelihood. Instead, they are defined implicitly through opponent-adjusted averages.

The central intuition is:

* if team (i) scores (e^{\text{off}}_{ig}) against team (j),
* and team (j) usually allows (D_j),
* then that offensive performance should be normalized relative to (D_j).

A multiplicative formulation is natural because efficiencies are positive and “relative to average” is scale-like.

Define normalized offensive and defensive multipliers:
[
\alpha_i = \frac{O_i}{\mu}, \qquad \delta_i = \frac{D_i}{\mu}.
]

Then (\alpha_i > 1) means above-average offense, and (\delta_i < 1) means above-average defense if lower defensive efficiency allowed is better.

A game-level multiplicative model is then
[
\mathbb{E}[e^{\text{off}}_{ig}] \approx \mu \cdot \alpha_i \cdot \delta_j \cdot \ell(h_g),
]
where (\ell(h_g)) is a location factor.

Similarly,
[
\mathbb{E}[e^{\text{def}}_{ig}] \approx \mu \cdot \alpha_j \cdot \delta_i \cdot \ell(-h_g).
]

This already captures the basic algebra of opponent adjustment.

---

## 2.2 Fixed-point equations

A KenPom-style estimator does not begin with a distributional model. It instead solves a system of self-consistency equations.

For each team (i), let (\mathcal{G}_i) be the set of games involving team (i). Let (w_g) be a game weight incorporating recency and perhaps margin compression.

Then define adjusted offense and defense by
[
O_i
===

\frac{\sum_{g \in \mathcal{G}*i} w_g ,
\frac{e^{\text{off}}*{ig}}{\delta_{j(g)},\ell(h_g)}}
{\sum_{g \in \mathcal{G}_i} w_g},
]
and
[
D_i
===

\frac{\sum_{g \in \mathcal{G}*i} w_g ,
\frac{e^{\text{def}}*{ig}}{\alpha_{j(g)},\ell(-h_g)}}
{\sum_{g \in \mathcal{G}_i} w_g}.
]

Equivalently, in multiplier form,
[
\alpha_i
========

\frac{1}{\mu}
\cdot
\frac{\sum_{g \in \mathcal{G}*i} w_g ,
\frac{e^{\text{off}}*{ig}}{\delta_{j(g)},\ell(h_g)}}
{\sum_{g \in \mathcal{G}_i} w_g},
]
[
\delta_i
========

\frac{1}{\mu}
\cdot
\frac{\sum_{g \in \mathcal{G}*i} w_g ,
\frac{e^{\text{def}}*{ig}}{\alpha_{j(g)},\ell(-h_g)}}
{\sum_{g \in \mathcal{G}_i} w_g}.
]

These are coupled fixed-point equations because (\alpha_i) depends on opponents’ (\delta_j), and (\delta_i) depends on opponents’ (\alpha_j).

The algorithm is then:

1. initialize (\alpha_i^{(0)} = 1), (\delta_i^{(0)} = 1) for all (i);
2. update offenses given current defenses;
3. update defenses given current offenses;
4. renormalize so league average remains anchored at (\mu);
5. iterate until convergence.

This is the statistical core of the classical system.

---

## 2.3 Location adjustment

Suppose home court is worth (H) points per 100 possessions.

Then a simple additive formulation is
[
e^{\text{off,neutral}}*{ig} = e^{\text{off}}*{ig} - H h_g.
]

A multiplicative variant is
[
\ell(h_g)=\exp(\eta h_g),
]
so that
[
\mathbb{E}[e^{\text{off}}_{ig}] \approx \mu \alpha_i \delta_j e^{\eta h_g}.
]

The additive version is easier to interpret in efficiency units; the multiplicative version is algebraically cleaner when combined with relative-strength adjustments. Either is plausible.

---

## 2.4 Recency weighting

Let (a_g) be the age of game (g), measured as time before the current date. Then define weights
[
w_g^{\text{time}} = \exp(-\lambda a_g),
]
or a piecewise variant.

This means recent games contribute more strongly:
[
w_g = w_g^{\text{time}} \cdot w_g^{\text{margin}} \cdot w_g^{\text{other}}.
]

---

## 2.5 Margin compression

A raw score differential contains useful information, but extreme blowouts are noisy because of substitutions, style changes, and strategic slowing.

One way to formalize “margin compression” is to replace raw differential (m_g) by a bounded transform
[
\tilde m_g = f(m_g),
]
where (f) is increasing and concave, such as
[
f(m)=\operatorname{sign}(m)\log(1+|m|),
]
or
[
f(m)=c\tanh(m/c_0).
]

In an efficiency framework, this may appear indirectly by dampening the influence of extreme games through
[
w_g^{\text{margin}} = \psi(|m_g|),
]
with (\psi) saturating, or by shrinking extreme game efficiencies back toward expectation.

The general principle is that game evidence should not scale linearly forever with margin.

---

## 2.6 Tempo adjustment

Pace is conceptually distinct from efficiency.

Let
[
\tau_g = p_g
]
be possessions in game (g). Then model expected possessions as a function of the two teams’ pace tendencies:
[
\mathbb{E}[\tau_g] \approx \bar \tau \cdot \pi_{i(g)} \pi_{j(g)} \cdot \ell_{\tau}(h_g),
]
where (\bar\tau) is national average tempo and (\pi_i) is a pace multiplier.

Then the tempo analogue of the efficiency fixed-point equations is
[
\pi_i
=====

\frac{1}{\bar\tau}
\cdot
\frac{\sum_{g \in \mathcal{G}*i} w_g ,
\frac{\tau_g}{\pi*{j(g)},\ell_\tau(h_g)}}
{\sum_{g \in \mathcal{G}_i} w_g}.
]

Again, this is an opponent-adjusted self-consistency estimator.

---

## 2.7 Final rating and Pythagorean expectation

Given adjusted offense and defense, define net rating
[
N_i = O_i - D_i.
]

Then define Pythagorean strength
[
\mathrm{Pyth}_i
===============

\frac{O_i^\gamma}{O_i^\gamma + D_i^\gamma},
]
where (\gamma > 0) is empirically fitted.

This is basically a nonlinear map from offensive and defensive efficiency to expected win percentage against an average opponent.

From a statistical point of view, this is not fundamental. It is a calibrated summary statistic layered on top of the adjusted efficiencies.

---

# 3. What the classical system is actually doing

Now let me restate the system in your language.

The classical system is solving a graph-normalization problem on pairwise contests. Each game provides noisy bilateral evidence about latent team offense, team defense, and pace. The estimator is:

* not fully generative,
* not likelihood-based,
* not explicitly regularized except through heuristic shrinkage and weighting,
* not uncertainty-aware,
* but highly stable because averaging over the schedule graph is powerful.

In modern terms, it is a hand-crafted alternating-projection estimator for latent node effects on a bipartite offense-defense interaction graph.

That is why it works.

It is also why its limitations are so clear.

---

# 4. A more sophisticated statistical version

Now let us write the model one would build if one wanted to preserve the spirit of KenPom but make it statistically coherent.

The first major move is to stop treating adjusted efficiencies as derived algebraic objects and instead model the game directly.

There are two natural approaches:

1. model points scored per possession or per 100 possessions;
2. model the possession-level scoring process itself.

I will begin with the game-level version, since it is the cleanest upgrade path.

---

# 5. A probabilistic game-level model

For a game (g) between teams (i) and (j), define latent states:

* (o_i(t_g)): offensive ability of team (i) at time (t_g),
* (d_i(t_g)): defensive ability of team (i) at time (t_g),
* (r_i(t_g)): pace tendency of team (i) at time (t_g).

These may be static or time-varying.

Let (y_{ig}) be points scored by team (i), and (p_g) observed/estimated possessions.

A log-link model for scoring rate is:
[
\log \lambda_{ig}
=================

\beta_0 + o_i(t_g) - d_j(t_g) + \beta_h h_g + \beta_x^\top x_g,
]
where (\lambda_{ig}) is expected points per possession for team (i), and (x_g) may include rest, travel, injuries, etc.

Then
[
y_{ig} \mid p_g, \lambda_{ig}
\sim
\text{Poisson}(p_g \lambda_{ig}),
]
or more realistically,
[
y_{ig} \mid p_g, \lambda_{ig}
\sim
\text{NegBin}(p_g \lambda_{ig}, \phi),
]
to allow overdispersion.

Likewise,
[
\log \lambda_{jg}
=================

\beta_0 + o_j(t_g) - d_i(t_g) - \beta_h h_g + \beta_x^\top x'_g.
]

This is already a much more coherent version of the offense-defense decomposition.

---

## 5.1 Modeling pace jointly

Instead of taking possessions as fixed, model them too:
[
\log \nu_g
==========

\gamma_0 + r_i(t_g) + r_j(t_g) + \gamma_h h_g + \gamma_x^\top z_g,
]
and then
[
p_g \sim \text{Poisson}(\nu_g)
]
or a rounded continuous model if one wants.

Now the full game model is joint:
[
p_g \sim p(p_g \mid r_i, r_j, h_g),
]
[
y_{ig} \sim p(y_{ig} \mid p_g, o_i, d_j, h_g),
\qquad
y_{jg} \sim p(y_{jg} \mid p_g, o_j, d_i, h_g).
]

This is already “KenPom made probabilistic.”

---

## 5.2 Hierarchical shrinkage

The second major upgrade is shrinkage.

In the classical system, small-sample teams and early-season teams are stabilized through informal priors. A proper model writes those priors explicitly:
[
o_i(0) \sim \mathcal{N}(\mu_o + \xi_i^{\text{pre}}, \sigma_o^2),
]
[
d_i(0) \sim \mathcal{N}(\mu_d + \zeta_i^{\text{pre}}, \sigma_d^2),
]
[
r_i(0) \sim \mathcal{N}(\mu_r + \rho_i^{\text{pre}}, \sigma_r^2).
]

Here (\xi_i^{\text{pre}}, \zeta_i^{\text{pre}}, \rho_i^{\text{pre}}) are preseason covariates:
returning minutes, prior year efficiency, coaching continuity, recruiting class, transfers, injuries, and so on.

This turns “preseason prior” from a heuristic blend into a legitimate prior mean.

---

## 5.3 Dynamic latent states

A modern system should not assume team strength is fixed through the season.

Let strengths evolve as state processes:
[
o_i(t) = o_i(t-1) + \epsilon_{it}^{(o)}, \qquad \epsilon_{it}^{(o)} \sim \mathcal{N}(0, q_o^2),
]
[
d_i(t) = d_i(t-1) + \epsilon_{it}^{(d)}, \qquad \epsilon_{it}^{(d)} \sim \mathcal{N}(0, q_d^2),
]
[
r_i(t) = r_i(t-1) + \epsilon_{it}^{(r)}, \qquad \epsilon_{it}^{(r)} \sim \mathcal{N}(0, q_r^2).
]

Or, with mean reversion,
[
o_i(t) = \mu_o + \phi_o(o_i(t-1)-\mu_o) + \epsilon_{it}^{(o)}.
]

This makes recency weighting endogenous. You no longer need to choose an exponential decay by hand. The state-space model learns how fast team strength changes.

That is a major conceptual improvement.

---

# 6. A ridge / RAPM-style linearized version

There is another way to formalize KenPom, closer to adjusted plus-minus.

Suppose we write game-level offensive efficiency as
[
e^{\text{off}}_{ig}
===================

\mu + \theta_i^{\text{off}} - \theta_j^{\text{def}} + \eta h_g + \varepsilon_{ig}.
]

Similarly,
[
e^{\text{off}}_{jg}
===================

\mu + \theta_j^{\text{off}} - \theta_i^{\text{def}} - \eta h_g + \varepsilon_{jg}.
]

This is just a two-way additive model with offense and defense effects. Stack all observations into a linear regression:
[
y = X\beta + \varepsilon,
]
where (\beta) contains all team offense effects and defense effects.

Because schedules are unbalanced and collinear, estimate via ridge:
[
\hat\beta
=========

\arg\min_\beta
\left{
|W^{1/2}(y-X\beta)|_2^2 + \lambda |\beta|_2^2
\right}.
]

This is, in some sense, the cleanest statistical reinterpretation of KenPom.

The differences are:

* KenPom is multiplicative and iterative,
* this is additive and penalized,
* KenPom uses heuristic recency/location/margin adjustments,
* this absorbs them directly through covariates and weights.

A statistician would probably trust this formulation more immediately because it has an objective function and a clear estimator.

---

# 7. A more expressive nonlinear model

Now let us go one step further, into a model that respects your background.

The weakness of classical systems is that “team offense” and “team defense” are too coarse. Teams are not scalar objects. They are structured collections of players, lineups, styles, and contexts.

So define team-state embeddings.

For team (i) in game (g), let (z_{ig}) be a learned representation based on:

* projected lineup,
* player embeddings,
* injury status,
* rest,
* scheme/style features,
* coach/system latent traits,
* recent possession-level tendencies.

Then let expected scoring rate be
[
\log \lambda_{ig}
=================

\beta_0 + f_{\theta}(z_{ig}, z_{jg}, h_g, x_g),
]
where (f_\theta) is a neural network or structured interaction model.

But do not stop there. The problem with a generic neural network is that it discards the interpretability that makes KenPom valuable.

So the right move is a structured decomposition:
[
f_{\theta}(z_{ig}, z_{jg}, h_g, x_g)
====================================

u_\theta(z_{ig}) - v_\theta(z_{jg}) + \kappa_\theta(z_{ig}, z_{jg}) + \beta_h h_g + \beta_x^\top x_g.
]

Interpretation:

* (u_\theta(z_{ig})): team (i)’s offensive quality,
* (v_\theta(z_{jg})): team (j)’s defensive quality,
* (\kappa_\theta(z_{ig}, z_{jg})): matchup interaction.

This is the modern generalization of KenPom:
main effects plus interaction.

The classical system effectively assumes (\kappa_\theta \equiv 0).

That assumption is often approximately fine at the team level. But the moment you care about lineup-specific matchups, it becomes the wrong model.

---

# 8. A Bayesian matchup model

If we want a truly sophisticated system, I would write something like this.

For each game (g), for both teams (i) and (j):
[
y_{ig} \mid p_g, \lambda_{ig} \sim \text{NegBin}(p_g \lambda_{ig}, \phi_y),
]
[
\log \lambda_{ig}
=================

\beta_0

* o_i(t_g)

- d_j(t_g)

* \kappa(i,j,t_g)
* \beta_h h_g
* \beta_x^\top x_g.
  ]

Pace:
[
p_g \sim \text{NegBin}(\nu_g, \phi_p),
\qquad
\log \nu_g
==========

\gamma_0 + r_i(t_g) + r_j(t_g) + \gamma_h h_g + \gamma_z^\top z_g.
]

Dynamic priors:
[
o_i(t) \sim \mathcal{N}(o_i(t-1), q_o^2), \quad
d_i(t) \sim \mathcal{N}(d_i(t-1), q_d^2), \quad
r_i(t) \sim \mathcal{N}(r_i(t-1), q_r^2).
]

Matchup interaction prior:
[
\kappa(i,j,t_g)
===============

a_i^\top b_j + \tilde\kappa(x_{ig},x_{jg}),
]
where (a_i,b_j) are low-rank latent style vectors, and (\tilde\kappa) may be a neural interaction term.

This says:
team matchups differ not only because of scalar offense and defense, but because style components interact.

That is the real upgrade path from classical ratings to modern prediction.

---

# 9. Why this is more sophisticated

Relative to the classical system, this model adds five things.

First, a **likelihood**.
Now we know what “fit” means.

Second, **regularization through priors**.
Now early season and weak schedules are handled correctly.

Third, **uncertainty quantification**.
We can produce posterior intervals for offense, defense, tempo, and win probability.

Fourth, **time variation**.
Recency is no longer heuristic; it is part of the state dynamics.

Fifth, **interaction structure**.
Now we can express that some offenses are especially effective or ineffective against particular defensive styles.

That last part is exactly where a deep learning background becomes useful.

---

# 10. A particularly clean “statistician’s” version

If I had to write the model in a paper in a way that would feel natural to a PhD statistician, I would probably present this version first:

[
e_{ig}
======

\mu + o_i - d_j + \eta h_g + \varepsilon_{ig},
\qquad
\varepsilon_{ig} \sim \mathcal{N}(0,\sigma^2 / p_g).
]

with priors
[
o_i \sim \mathcal{N}(0,\tau_o^2), \qquad d_i \sim \mathcal{N}(0,\tau_d^2),
]
and identifiability constraints
[
\sum_i o_i = 0, \qquad \sum_i d_i = 0.
]

Then perhaps a separate pace model
[
\log p_g = \gamma_0 + r_i + r_j + \gamma_h h_g + \xi_g,
\qquad
r_i \sim \mathcal{N}(0,\tau_r^2).
]

This is the clean bridge between KenPom and the modern mixed-effects / penalized-regression view.

Then I would say:

the traditional system is a deterministic approximation to this latent-effects model, with hand-crafted weighting and iterative opponent normalization in place of full estimation.

That sentence gets to the heart of it.

---

# 11. How I would phrase the difference conceptually

KenPom asks:

given the schedule graph, what is each team’s adjusted efficiency after repeatedly normalizing by opponent strength?

A more sophisticated model asks:

what latent offensive, defensive, and tempo processes most likely generated the observed scores and possessions, after accounting for venue, schedule, time variation, and matchup-specific interactions?

The first is an adjustment procedure.
The second is a generative inference problem.

That is the real shift.

---

# 12. Where your own point of view naturally enters

From your perspective, the most interesting extension is probably not merely “Bayesian KenPom.” It is a structured interaction model between team composition and opponent composition.

A natural architecture would be:

* encode each team by current active players and their roles;
* aggregate via a permutation-invariant model or Set Transformer;
* decompose into offense main effect, defense main effect, and offense-defense interaction;
* model pace jointly;
* train under a count or ordinal likelihood for game points or possession outcomes.

In symbols:
[
z_{ig} = \mathrm{Enc}*\theta(\text{lineup}*{ig}, \text{roster}_{ig}, \text{context}*g),
]
[
\log \lambda*{ig}
=================

\beta_0 + u_\theta(z_{ig}) - v_\theta(z_{jg}) + \kappa_\theta(z_{ig}, z_{jg}) + \beta_h h_g.
]

That is, in my view, the true modern descendant of KenPom.

It keeps what is essential:
opponent adjustment, pace adjustment, interpretable offense/defense decomposition.

But it drops the fiction that teams are static scalar entities.

---

# 13. A compact summary

If I compress everything:

**Classical KenPom-style system**
[
\text{Adjusted strength}
========================

\text{weighted average of game efficiency normalized by opponent strength},
]
solved by iteration.

**Statistical reformulation**
[
\text{Observed score/efficiency}
================================

## \text{offense effect}

\text{defense effect}
+
\text{venue}
+
\text{time dynamics}
+
\text{noise},
]
estimated by likelihood or penalized regression.

**Modern extension**
[
\text{Observed score/efficiency}
================================

\text{team representation}
+
\text{opponent representation}
+
\text{matchup interaction}
+
\text{tempo process}
+
\text{uncertainty}.
]

---

# 14. My honest view

KenPom’s system is excellent because it solves the right first-order problem with a disciplined bias toward stability and interpretability.

But mathematically, it is not the end of the story. It is what you get when a very smart analyst builds a strong empirical estimator before writing down the full probabilistic model.

A statistician sees immediately how to finish the job.

The real opportunity is not to “beat KenPom” with a black box. It is to preserve the decomposition that makes his ratings useful, while replacing fixed-point heuristics with a coherent latent-variable model that allows shrinkage, dynamics, matchup effects, and uncertainty.

That would be the proper modernization.

If you’d like, I can turn this into a polished LaTeX note with sections, equations, and a final proposed model written in paper style.
