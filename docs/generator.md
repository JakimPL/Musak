# Synthetic Sight-Reading Exercise Generator

## 1. Goals and approach

The Musak project trains an LLM-based sight-reading model on a corpus of piano exercises. The synthetic generator
described in this document is a *classical, stochastic* counterpart to that model: a deliberately non-neural process
whose purpose is to produce arbitrarily large synthetic corpora that complement the LLM's training and evaluation data.
Three properties drive the design.

It must be **controllable** through interpretable style inputs — the register a hand occupies, how dense and how syncopated
its activity is, how strongly the two hands coordinate, what harmonic palette it draws on. It must be **faithful** to the
empirical figure statistics of the reference corpus, measured by the project's existing total-variation distance metric
on figure n-gram distributions per `(scale_type, hand, n)` group. And it must be **cheap** enough to produce large
corpora on demand without the GPU footprint of the neural model.

The architecture decomposes the generator into two layers that meet in one place. A small set of **low-order global
stochastic processes** controls macro structure — the slow register trajectory, the rhythmic activity field, the
harmonic chord progression. A second, **empirical micro layer** — the figure n-gram vocabulary already extracted from
the training corpus — owns local contour and rhythm. At each step of generation, the register curve proposes an
absolute diatonic anchor, the accent field proposes a metrical weight and an onset decision, the chord track proposes
a harmonic context, and a single figure-substitution step combines these signals with the empirical vocabulary to
choose, anchor and emit the next figure. The emitted token prefix is gated through the project's existing playability
constraint engine, which is the only source of hard rejection.

This factorisation is principled rather than aesthetic. The figure n-grams the project already extracts are by
construction translation-invariant in pitch and scale-invariant in time; they cleanly carry the corpus's high-order
*local* structure. The global processes therefore need only be *low-order* — matching second-order statistics of
register and density, and a first-order Markov chain of harmonic transitions. Conversely, this is why a flat Markov
chain over absolute notes would be strictly worse here: it would re-learn what the figure vocabulary already encodes
while losing the interpretable style knobs that low-order global processes give for free. Empirically, the same
TV-distance metric that defines "faithful to reference" then becomes the generator's calibration objective, closing the
loop the design opens.

## 2. Representations

The generator does almost all of its work in scale-degree space rather than in semitones. The token vocabulary and the
figure n-grams already commit to this representation; the generator builds on top.

### 2.1 The diatonic lattice

Pitch in this system lives on a one-dimensional integer lattice indexed by *diatonic position*

$$p \;=\; o \cdot s \;+\; (d - 1),$$

where $d \in \{1, \dots, s\}$ is the scale degree of the note, $o \in \mathbb{Z}$ its octave offset relative to the
hand's home register, and $s = |I|$ the scale's degree count (the cardinality of `SCALE_INTERVALS[scale_type]`). The
three scales currently supported — major, harmonic minor, melodic minor — all happen to have $s = 7$, but the algorithms
below derive $s$ from the scale at every use site so that a future addition of a non-heptatonic scale would not silently
break their semantics.

Where the diatonic lattice carries position, the *pitch class*

$$\pi(d, a) \;=\; \bigl(\sigma(d) + a\bigr) \bmod 12,$$

with $\sigma(d)$ the semitone offset of degree $d$ from the tonic and $a \in \{-1, 0, +1\}$ the note's accidental,
carries the residue used for chord membership tests. The two coordinates are cheap to translate between, and the
generator moves between them only at the substitution step where it must score figure pitches against chord tone sets.

### 2.2 Figures as anchor-relative templates

A figure n-gram is the central unit of the empirical vocabulary. Each `FigureNGram` is a sequence of onsets, each of
which carries a tuple of `(relative_position, accidental)` degrees and a normalised duration. The "relative" qualifier
is precise: the figure builder anchors every figure to the minimum diatonic position of its first onset, subtracts that
anchor from all positions in the window, and rescales every duration by the shortest onset's duration. The result is a
*contour and rhythmic template* — translation-invariant in pitch, scale-invariant in time — that records *what shape*
the figure has but not where on the keyboard or where in the bar it happens to occur.

This invariance is what makes the figure vocabulary reusable across keys and tempi; it is also what makes the global
processes necessary, because any actual sight-reading exercise must commit somewhere to an absolute register, an
absolute time, and an implied harmony. The generator's task is to supply that absolute context to the relative
templates without distorting the templates' own distribution beyond what the conditioning strictly requires.

### 2.3 Chords as key-relative symbols

Chords share the figures' coordinate system. A chord symbol is

$$\bigl(d_r,\; a_r,\; q,\; e\bigr),$$

with $d_r \in \{1, \dots, s\}$ the root degree (key-relative, so the chord track is transposition-invariant in the same
way that figures are anchor-relative), $a_r \in \{-1, 0, +1\}$ the root accidental, $q$ a quality drawn from a
YAML-configurable vocabulary (the four standard triad qualities — major, minor, diminished, augmented — enabled by
default), and $e$ an optional extension. Extensions for sevenths, ninths, elevenths and the altered variants ♭9 and ♯11
are *defined* in the configuration but disabled in v1 so that the initial generator runs over triads only.

The tone set of a chord is obtained by generic-third stacking. The $m$-th chord member sits at generic degree
$d_m = ((d_r - 1 + 2m) \bmod s) + 1$, and its accidental is

$$\alpha_m \;=\; \bigl[(\sigma(d_r) + a_r + q_m) - \sigma(d_m)\bigr] \bmod 12,$$

interpreted as a signed residue in $\{-6, \dots, +6\}$, where $q_m$ is the $m$-th semitone interval of the quality
(e.g. $(0, 3, 7)$ for minor; $(0, 4, 7)$ for major; $(0, 3, 6)$ for diminished). For members beyond the triad core the
$q_m$ is absent and the natural diatonic accidental is used (zero by default, with optional alterations declared by the
extension). The construction is general enough to handle the chords that motivated working in scale-degree space in the
first place. Borrowed-quality chords — minor iv in C major, for instance — produce $(4, 0), (6, -1), (1, 0)$: the ♭6
appears automatically as accidental $-1$ on degree 6, exactly how `NoteToken` and `FigureDegree` already encode chromatic
notes. Secondary dominants such as V/V in C major produce $(4, +1)$ for F♯; chromatic-root chords such as ♭VI live as
root $(6, -1)$ with major quality. A chord whose construction would require an accidental outside $\{-1, 0, +1\}$ is
unspellable in `NoteToken` and is rejected at expansion time. In practice this constraint is rarely binding and acts as
a natural ceiling on the vocabulary rather than a practical obstacle.

The chord track in the generator is then a sequence $C(t)$ taking values in this symbolic space, drawn from a Markov
chain whose transition matrix can come either from a uniform-with-stickiness prior or from empirical chord decoding on
the training corpus.

### 2.4 The reference database

A relative `FigureNGram` keyed by $(\text{scale\_type}, \text{hand}, n)$ is enough to make the TV-distance
metric work, but the generator needs more than its marginal. The figure distribution conditional on a chord
root drives harmonic conditioning; the figure distribution conditional on metrical position drives
placement; an empirical base-duration distribution is what reconstructs real time from figures' normalised
rhythms; and the dependence on time signature is what keeps placement statistics from being averaged over
incompatible meters. The reference store therefore records, alongside each relative figure, the absolute
anchor it was extracted at (degree, accidental, octave), the base duration that normalisation divided out,
the bar-relative onset, and the time signature, so that any of those conditional marginals is one query
away. The relative figure itself remains the canonical key for the TV-distance metric — the marginal that
comparison machinery has always operated on is unchanged.

A small but important refinement: figures' relative contour is largely meter-invariant, so the contour vocabulary is
pooled across time signatures (more data per figure, lower TV-distance variance), while *where* a figure tends to start
within a bar is strongly meter-dependent, so the placement statistics are conditioned on time signature and
`bar_relative_onset`. Conditioning the contour itself on time signature would only fragment the counts without gain.

## 3. The register curve

The register curve $P_i(t)$, defined per hand $i \in \{\text{L}, \text{R}\}$, is the stochastic process on the diatonic
lattice whose role is to anchor figures in absolute pitch space. At each step it produces an integer diatonic position;
its value tells the substitution step *where* to drop the next figure, and its local slope tells it *which* figure to
prefer (an ascending figure when the curve trends up, a descending one when it trends down).

The unifying view is a Gaussian process on the lattice,

$$P_i(t) \;\sim\; \mathcal{GP}\bigl(\mu_i, \, k(t, t')\bigr), \qquad k = k_{\text{arch}} + k_{\text{wobble}}, \qquad
\mu_i = o_i \cdot s,$$

where $\mu_i$ is the hand's home register ($o_i = 5$ for the right hand, $o_i = 3$ for the left), $k_{\text{arch}}$ a
long-lengthscale kernel encoding the slow phrase arch, and $k_{\text{wobble}}$ a short-lengthscale Matérn-½ kernel
encoding local register drift. Rather than realise this kernel as a generic GP, the implementation samples its two
components as cheap $O(N)$ trajectories whose hyperparameters are interpretable moments of the corpus.

The **arch** is a band-limited random function,

$$P_i^{\text{arch}}(k) \;=\; \sum_{j=1}^{J} a_j \, \cos\!\bigl(\pi j (k + \tfrac{1}{2})/N\bigr), \qquad
a_j \sim \mathcal{N}\!\bigl(0,\; A^2 / j^{\,2d}\bigr),$$

evaluated on the index grid $k = 0, \dots, N-1$. The mid-cell DCT basis avoids the periodic wrap-around that a pure-sine
basis would impose at segment boundaries; the geometric coefficient decay $1/j^d$ plays the role of an empirical power
spectral density (Welch-averaged from length-normalised training trajectories in the planned fitting procedure) and
concentrates energy in the low frequencies, so the trajectory is smooth by construction. A small basis count $J$ —
three by default — gives a fixed, controllable generator with very few interpretable parameters.

The **wobble** is an Ornstein–Uhlenbeck residual mean-reverting toward zero,

$$r_{k+1} \;=\; (1 - \theta)\,r_k \;+\; \varepsilon_k, \qquad \varepsilon_k \sim \mathcal{N}(0, \sigma^2), \qquad r_0 = 0,$$

with $\theta \in (0, 1]$ the mean-reversion rate and $\sigma$ the per-step innovation. The stationary variance
$\sigma^2 / (2\theta - \theta^2)$ and lag-1 autocorrelation $1 - \theta$ are direct functions of $(\theta, \sigma)$ and
are exactly the two moments fitted from the corpus: empirical register spread per hand and lag-1 onset-pitch
autocorrelation. Mean reversion earns its place in the model in a specific way. A plain Gaussian random walk
wanders without bound and drifts out of the playable register over a long exercise; the OU residual instead
pulls back toward zero at every step, at a rate set by $\theta$. The pull is a *constant attractor* — the
same at every $k$, with no end-of-segment effect — so the marginal distribution of $r_k$ stays centred on
zero throughout the trajectory rather than only meeting it at the endpoints. Combined with the arch's
mean-zero basis coefficients, the full curve $\mu_i + P_i^{\text{arch}}(k) + r_k$ then has marginal mean
$\mu_i$ at every step, and the trajectory is *anchored* on the home register rather than incidentally
returning to it.

The trajectory delivered to the substitution step is the lattice-quantised sum

$$P_i(k) \;=\; \mathrm{round}\bigl(\mu_i \,+\, P_i^{\text{arch}}(k) \,+\, r_k\bigr),$$

an integer diatonic position per onset step.

## 4. The accent field

The accent field is a marked point process on the bar-aligned grid: every grid cell is either an onset or
it is not, and each onset carries an accent weight. The per-cell probability $p_i(t)$ does two jobs at once,
and that double role is the design point. As the parameter of a Bernoulli draw it decides whether the cell
becomes an onset at all — $p_i \to 1$ leaves no rests, $p_i \to 0$ leaves only rests — so the field is what
controls overall filling. The same number is also the accent value the surviving onsets carry forward: a
high-weight onset is by construction near a strong metrical position or inside a busy region of the
envelope, so a single scalar per cell tells the substitution step both that the cell is more likely to
sound and that, when it does, it carries phrasing weight worth conditioning on.

Following the log-Gaussian Cox process picture, the log-intensity at grid cell $t$ decomposes into a meter-locked term
and a smooth random envelope,

$$\log \lambda_i(t) \;=\; \text{metric weight at } t \;+\; g_i(t),$$

where the metric weight rewards strong metrical positions and $g_i(t)$ is sampled from the same band-limited-random
machinery that produces the arch (its amplitude and basis count are independent style knobs from the pitch process). To
guarantee that the resulting per-cell onset probability lies in $[0, 1]$ the implementation passes the logit through a
numerically stable sigmoid (`scipy.special.expit`):

$$p_i(t) \;=\; \mathrm{expit}\!\bigl(\beta_0 \,+\, \beta_1 \cdot \mathrm{ind}(t)^{\gamma} \,+\, g_i(t)\bigr),$$

with $\beta_0$ a global density baseline (low values give sparse onsets), $\beta_1$ the strength with which metrically
strong positions are preferred, and $\gamma$ a shape exponent on the indispensability. The Bernoulli draw at parameter
$p_i(t)$ decides the onset; per the double-role above, the cell's accent weight is that same $p_i(t)$ —
strongest on downbeats modulated by the envelope, weakest off-beat in sparse regions.

Metrical indispensability is computed by a single rule that works uniformly across simple and compound meters: for a
bar discretised into $M$ grid cells, $\mathrm{ind}(k) = \gcd(k, M) / M$ for $k \in \{0, \dots, M-1\}$. In 4/4 with a
sixteenth-note grid ($M = 16$) this recovers the familiar hierarchy $\mathrm{ind}(0) = 1$ on the downbeat,
$\mathrm{ind}(8) = 1/2$ on the half-bar, $\mathrm{ind}(\{4, 12\}) = 1/4$ on the remaining quarter divisions, and so on
down to $\mathrm{ind}(\text{odd}) = 1/16$ on the sixteenth-note off-beats; in 6/8 with $M = 6$ it correctly identifies
the secondary strong beat at position 3 with $\mathrm{ind}(3) = 1/2$, second only to the downbeat. The smooth envelope
$g_i$ is what gives the generator phrasing: cells in a busy region are correlated through it, producing auto-correlated
dense and sparse passages rather than the salt-and-pepper onsets that independent Bernoulli draws alone would yield.

The rhythmic figure n-gram statistics of the corpus — IOI distributions, duration ratios, durational variety — are not
modelled directly by the LGCP. They are reproduced because the figures chosen at substitution carry their own
normalised IOIs and are sampled from the empirical rhythm-aware vocabulary. The LGCP's job is the *skeleton* and the
accent weights; the figures fill the rhythmic detail.

The point-process layer also carries a coarse activity gate $A_i(t) \in \{0, 1\}$ that controls whether onsets are drawn
at all in a region; that gate is the layer at which the co-activity coupling between hands acts (§7).

## 5. The chord track

The chord track $C(t)$ is the harmonic spine of the exercise: a slowly varying, key-relative chord per
harmonic-rhythm window. Its purpose is to *condition* the concrete pitches that figure substitution chooses, biasing
them toward harmonic coherence without *forcing* chord tones. Passing and neighbor notes — characteristic of
sight-reading material — must remain available on weak beats, so the harmonic shape acts as tonal gravity, not as a
hard constraint.

The track is modelled as a first-order Markov chain over chord symbols in the representation of §2.3,

$$C(t+1) \mid C(t) \;\sim\; P\bigl(\cdot \mid C(t)\bigr),$$

with an initial distribution $\pi$ and a row-stochastic transition matrix $P$ over the chord vocabulary. In v1 the
transition matrix is supplied externally: the bundled helper produces a uniform matrix with a configurable
self-transition bias, which is the same stickiness device the chord decoder uses, and once chord-decoded statistics
have been accumulated against the corpus the empirical transition matrix replaces the uniform prior. The
harmonic-rhythm resolution — one chord per whole note, half note, or quarter note — is itself a configurable
power-of-two note value consistent with the rest of the duration system; the windowing is bar-aligned so chords never
span barlines even in odd meters such as 3/4 or 6/8.

Concretely, the chord grid is laid down from the downbeat of each bar at the configured resolution and
truncated at the next barline. Half-note resolution in 3/4 therefore yields a half-bar chord followed by a
quarter-bar tail chord per bar; in 5/4 it yields two half-bar chords and a quarter-bar tail; in 6/8 a
dotted-quarter resolution gives two equal chord windows, while a half-note resolution gives a half-bar
chord plus a quarter-bar tail. Musically more natural subdivisions for odd meters — 2:2:1 for 5/4, the
3+3 / 3+2+2 / 2+2+3 groupings for compound or asymmetric meters — are deferred beyond v1; the
bar-aligned-with-truncation rule is the single deterministic policy until the data supports the choice.

### 5.1 Decoding the chord track from the corpus

Beyond sampling at generation time, the same chord representation is also *recovered* from the training
corpus, so that the transition matrix and the chord-conditioned figure distribution $p(\text{figure} \mid C)$
that §6's harmonic-fit score reads from are fitted to real harmonic motion rather than to a uniform prior.
The training corpus contains notes and rests but no harmonic labels, and the symbols decoded onto each
segment must be exactly the symbols the sampler will later emit — decoder and sampler share the same state
space, so the parameters of one read straight into the other without translation.

The decoding procedure has three components. First, *sounding windows*: each segment is partitioned into
the bar-aligned chord grid defined just above, and within each window every sounding pitch class
accumulates a mass equal to the total duration it overlaps the window. The window thereby becomes a small
categorical distribution over pitch classes, weighted by how long each class actually sounded. Second,
*candidates*: the chord vocabulary of §2.3 is expanded into every spellable chord symbol — every
combination of root degree, root accidental, quality and extension — each precomputed as its pitch-class
set; a chord whose construction would require an accidental outside $\{-1, 0, +1\}$ is dropped at this
stage rather than scored. Third, an *emission score* that rewards a candidate by the mass of its pitch
classes that the window contains and penalises it by the mass that lands outside: a soft scoring that lets
passing and neighbor tones knock the score down by exactly their own mass times a non-chord-tone penalty
rather than disqualifying a candidate outright — the same softness the sampler exhibits at substitution
time, and for the same reason. Standard Viterbi over those windows, with the same self-transition
stickiness the sampler uses as its prior, returns the most-likely chord per window.

Once each training segment is decoded, the empirical transition matrix is the count of adjacent decoded
chord symbols pooled across the corpus, and the figure-by-chord conditional is the co-occurrence count of
each figure with the chord active at its first onset, normalised per chord. The harmonic-fit score in §6
reads from this same conditional — because decoder and sampler share their symbols, the two ends of the
loop are aligned by construction rather than by separate bookkeeping.

## 6. Figure substitution as an I-projection

The three global processes meet the empirical figure vocabulary in one substitution step. At each anchor step the
substitution chooses a figure $f$ given the local global state — register-curve value $P_i(k)$, accent weight
$\lambda_i(k)$, current chord $C(t)$ — from the conditional

$$p\bigl(f \mid \text{group},\, P_i,\, \lambda,\, C\bigr) \;\propto\; \underbrace{p_{\text{emp}}\bigl(f \mid \text{group}\bigr)}_{\text{empirical figure prior}} \,\cdot\, \exp\!\Bigl(\lambda_{\text{curve}} \, S(f, P_i) \,+\, \lambda_{\text{harm}} \, H(f, C, m) \,+\, \lambda_{\text{accent}} \, A\bigl(f, \lambda_i(k)\bigr)\Bigr),$$

where $p_{\text{emp}}(f \mid \text{group})$ is the empirical figure distribution at the group
$(\text{scale}, \text{hand}, n)$, $S(f, P_i)$ is a slope-fit score comparing the figure's net contour
($\sum$ relative steps across the figure's onsets) to the local slope of the register curve, $H(f, C, m)$
is a harmonic-fit score comparing the figure's concrete pitches (after anchoring to $P_i$) to the chord
tones of $C$ at a weight that depends on the figure's metrical position $m$ in the bar — chord tones
rewarded on strong beats, non-chord tones permitted and indeed favoured on weak beats *between* chord
tones — and $A(f, \lambda_i(k))$ is an accent-fit score comparing the figure's internal accent shape
(longer notes and chord-tone onsets aligned with the figure's strong points) to the LGCP envelope value
$\lambda_i(k)$ at the current cell, with the envelope reading from §4.

The construction is an exponential-family tilt of the empirical distribution, and that is the point. Geometrically the
tilted distribution is the **I-projection** of $p_{\text{emp}}$ onto the constraint manifold defined by the curve,
harmonic and accent targets — the distribution closest in Kullback–Leibler divergence to the reference that still
respects the local conditioning. Operationally, each coefficient $\lambda_{\text{curve}}$, $\lambda_{\text{harm}}$
and $\lambda_{\text{accent}}$ is an independent stability ↔ fidelity dial: setting any of them to zero recovers the
reference marginal exactly along that direction; increasing it sharpens the control at the cost of moving away
from the reference. This is the precise sense in which the design is *data-based yet stable, close to reference*:
both the marginal preservation in the limit and the controllability away from it are properties of the same
exponential family.

A further tilt already exists in the repository and is kept strictly orthogonal to the conditioning tilts. The
`commonness_bias` parameter $\beta$ in `synthetic/figures.py` flattens or sharpens the empirical figure frequencies
themselves via

$$p_{\text{emp}}(f) \;\propto\; c(f)^{\beta},$$

where $c(f)$ is the figure's training count: small $\beta$ produces a flatter distribution favouring rare figures,
large $\beta$ concentrates mass on the most common ones. It is a "commonality" style knob, conceptually distinct from
the conditioning tilts. Mixing it with the $\lambda$ tilts is straightforward — the exponents simply add along their
respective directions — and the knobs are exposed as separate inputs so that the style API stays interpretable.

## 7. Hand interaction

Two hands play one piece. The generator separates the question of how they coordinate into three orthogonal couplings,
each acting at a different layer of the model so that they can be controlled independently and so that conflations
between them do not propagate through the design.

**Co-activity**, parameterised by $h_o \in [0, 1]$, acts on the activity gate $A_i(t) \in \{0, 1\}$ of the accent point
process. The two gates are coupled by a Gaussian copula: drawing

$$(z_L, z_R) \;\sim\; \mathcal{N}\!\left(\mathbf{0},\, \begin{pmatrix} 1 & \rho \\ \rho & 1 \end{pmatrix}\right), \qquad A_i \;=\; \mathbb{1}\!\bigl[z_i > \Phi^{-1}(1 - h_a^i)\bigr],$$

with off-diagonal correlation $\rho = 2 h_o - 1 \in [-1, +1]$, gives each gate a marginal active probability equal to
the prescribed per-hand activity $h_a^i$ and a controllable degree of joint activity. The extremes are musical: at
$h_o = 0$ the gates are anti-correlated and the hands' active spans are disjoint, producing strict alternation
(hocket); at $h_o = 1$ the gates coincide and an onset never lands on the other hand's rest; intermediate values
interpolate continuously between the two. The Fréchet bound $h_a^L + h_a^R \le 1$ is the only constraint on the
disjoint extreme; the copula degrades gracefully outside it.

**Sync**, parameterised by $h_s \in [0, 1]$, acts one layer down on onset placement: given both hands active in a
window, $h_s$ is the probability that their attacks coincide on the grid. It is realised either as a shared onset mask
drawn with weight $h_s$ plus independent masks with weight $1 - h_s$, or equivalently as a boost to one hand's intensity
at cells where the other is attacking.

**Harmony** is the chord track itself, shared between the hands. At sync-coincident attacks both hands' degree choices
are tilted toward chord tones of the same $C(t)$, so the resulting vertical interval is consonant by construction —
without any chord-forcing. Where attacks do not coincide, each hand is still governed by the same harmonic context.

Keeping these three on separate layers matters because they answer separate musical questions: *do the hands share
active regions?* (co-activity), *do their attacks line up?* (sync), and *do their concurrent pitches make harmonic
sense?* (harmony). Conflating any two of them yields uncontrollable interactions; separating them gives a style API
whose dimensions a user can move along one at a time.

## 8. Hard playability

The stochastic model is kept deliberately soft: no hard cutoffs on melodic gap, hand span, notes per onset, or bar
filling. These are enforced instead by the existing generation constraint engine, `GenerationConstraintState` in
`musak_model/generation/constraints.py`, which already validates token prefixes during neural decoding. The synthetic
generator submits each emitted figure to that engine and resamples on rejection. This isolates the probabilistic model
from playability mechanics, keeps it interpretable, and lets the constraint engine — which is already tested against
the LLM — serve as the single source of truth for what counts as a playable token sequence.

The constraint engine is itself scale-aware where it must be: maximum static hand span is measured in scale degrees and
therefore requires a scale, exactly because hand-span semantics depend on the scale being played. That requirement is
enforced at engine construction and respected at substitution time.

## 9. Fitting and validation

Each global process has interpretable hyperparameters that are fit by moment matching against the corpus, and one
nonparametric coefficient that is calibrated against the project's own reference metric.

For the **register curve**, the OU parameters $(\theta_i, \sigma_i)$ per hand are determined by the empirical register
spread and lag-1 autocorrelation of the training onset-pitch sequences, and the arch amplitudes $\{A_j\}$ by the
empirical low-frequency power spectral density of length-normalised trajectories — roughly four numbers per
`(scale_type, hand)`. For the **accent field**, the baseline logit, metric gain, and exponent are fit to the measured
strong/weak onset ratio and overall density, both already exposed by the rhythm extractor. For the **chord track**, the
transition matrix is the empirical transition count between adjacent Viterbi-decoded chord windows on the training
corpus, learned once during the chord-segmentation pass; the chord-conditioned figure distribution that drives the
harmonic-fit score is accumulated in the same pass.

The substitution tilts $\lambda_{\text{curve}}$, $\lambda_{\text{harm}}$ and $\lambda_{\text{accent}}$ are
not moment-matched in the same way; they are calibrated against the project's existing reference metric. The
generator is run at several values of each, the resulting figure distribution is scored by `figure_distribution_metrics`
— the mean total-variation distance per group against the reference — and the largest tilt is chosen that keeps the
TV distance below a target threshold (a sensible initial choice is $0.1$). The same metric thus serves as both the
design's stated objective and the empirical calibration knob: the closed loop the design opens at §1 is closed here.

A companion validation metric specific to harmony — the harmonic-consonance rate at coincident onsets — is planned so
that the harmonic tilt is checked against ground truth, not just against the unconditioned figure marginal.

## 10. Scope and what this is not

A few design positions are worth stating explicitly so that the boundaries of the generator are clear.

**This is not a neural model**, and that is the point. A neural autoregressive model is what `musak_model` already is;
the value of the classical generator is precisely that it is cheap, controllable, interpretable, and trivially
auditable. The two are complementary: the LLM captures whatever the classical generator does not, and the classical
generator gives the LLM a source of training and evaluation data whose properties can be characterised analytically.

**Flat Markov chains over absolute notes are rejected.** They conflate register drift with local contour, hide the
style knobs that should be interpretable, wander out of register without a cap, and re-learn what the figure vocabulary
already encodes. The figure-substitution architecture sidesteps all four problems by construction.

**Forcing chord tones for harmony is rejected.** Passing and neighbor tones are characteristic of sight-reading
material and must be permitted; the soft harmonic tilt of §6 provides tonal gravity without eliminating them. The
functional-bass / figured-bass alternative — where the left hand defines harmony and the right hand conditions on it —
is asymmetric and weak for two-melodic-line textures; it is retained only as an optional texture mode for Alberti and
block-chord settings.

**The chord vocabulary in v1 is small by choice.** The chord-representation machinery covers any tertian chord that can
be spelled with single accidentals; the YAML vocabulary turns extensions on and off so that the v1 generator runs over
diatonic triads only, and the more colourful chords can be enabled once the empirical transition matrix has been
estimated against them.

The natural future directions follow from the seams the design exposes. The chord-conditioned figure distribution that
gives the harmonic-fit score its empirical grounding is the next data product: it requires running the chord decoder
over the training corpus and accumulating figure counts per chord context. Extending the chord template set to common
borrowed and applied chords — iv, ♭VI, ♭VII, V/V, the characteristic V and vii° of harmonic minor — becomes
data-supported once the empirical transition matrix is in place. And the `commonness_bias`, the smooth gate of the
co-activity coupling, and the harmonic-rhythm resolution are all knobs whose default values are sensible but whose
calibrated-from-data versions would tighten the generator further.

The decisive structural point on which the entire design rests is the one already stated in §1: `FigureNGram` is
anchor-relative and rhythm-normalised, so it already owns the high-order local structure. Exactly because of that, the
global processes can stay low-order, harmony enters as soft conditioning rather than as a vocabulary change, and the
exponential-family substitution provably stays close to the reference along whatever directions are not actively being
controlled. The generator is, in the end, a small set of well-chosen low-order models stitched together by one
information-projection.
