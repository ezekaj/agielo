"""
Centralized Constants Configuration
====================================

All magic numbers and configuration constants used across the system are defined here
to ensure consistency, maintainability, and self-documentation.

Usage:
    from config.constants import DEFAULT_DECAY_RATE, EXPOSURE_DECAY_RATE

Or import via config:
    from config import DEFAULT_DECAY_RATE, EXPOSURE_DECAY_RATE

Categories:
- Memory Forgetting: Parameters for Ebbinghaus forgetting curve simulation
- Active Learning: Parameters for curiosity-driven learning
- Bayesian Surprise: Parameters for novelty detection
- Episodic Memory: Parameters for episodic memory store
- Numerical Stability: Bounds for safe mathematical operations
"""

# =============================================================================
# Memory Forgetting Constants (neuro_memory/memory/forgetting.py)
# =============================================================================

# Default decay rate for memory activation
# Controls how fast memories fade without rehearsal
# Based on Ebbinghaus forgetting curve research
DEFAULT_DECAY_RATE = 0.5

# Rehearsal boost multiplier - strength increase when memory is accessed
# Higher values mean rehearsal is more effective at preserving memories
REHEARSAL_BOOST_MULTIPLIER = 1.5

# Minimum activation threshold
# Below this threshold, memories are considered "forgotten"
MIN_ACTIVATION_THRESHOLD = 0.1

# Ebbinghaus stability parameters (in hours)
BASE_STABILITY_HOURS = 1.0  # Initial stability when memory is first encoded
STABILITY_MULTIPLIER = 1.5  # How much stability increases on successful retrieval
MIN_STABILITY_HOURS = 0.5  # Minimum allowed stability
MAX_STABILITY_HOURS = 720.0  # Maximum stability (30 days)

# Forgetting threshold - retention level below which memory may be forgotten
FORGET_RETENTION_THRESHOLD = 0.3

# Spaced repetition timing
IMMEDIATE_REVIEW_DELAY_HOURS = 0.25  # 15 minutes for immediate review

# =============================================================================
# Active Learning Constants (integrations/active_learning.py)
# =============================================================================

# Exposure decay rate - how much novelty bonus decreases per exposure
# Formula: novelty_bonus = 1.0 / (1.0 + exposure_count * EXPOSURE_DECAY_RATE)
EXPOSURE_DECAY_RATE = 0.1

# Learning priority weights for uncertainty, curiosity, and novelty
# Must sum to <= 1.0 for proper priority calculation
UNCERTAINTY_PRIORITY_WEIGHT = 0.4  # Weight for learning unknowns
CURIOSITY_PRIORITY_WEIGHT = 0.4    # Weight for learning interesting things
NOVELTY_PRIORITY_WEIGHT = 0.2      # Weight for new topics

# Default confidence and curiosity values for new topics
DEFAULT_TOPIC_CONFIDENCE = 0.5
DEFAULT_TOPIC_CURIOSITY = 0.5

# RND (Random Network Distillation) curiosity parameters
RND_CURIOSITY_WEIGHT = 0.5  # Weight for RND curiosity in combined score
RND_HIGH_CURIOSITY_THRESHOLD = 0.6  # Threshold for boosting curiosity
LOW_CURIOSITY_THRESHOLD = 0.5  # Below this, curiosity can be boosted

# Known topic priority reduction factor
# Applied when selecting topics to penalize already-known topics
KNOWN_TOPIC_PRIORITY_FACTOR = 0.3

# Default curiosity boost amount
DEFAULT_CURIOSITY_BOOST = 0.2

# High confidence threshold - above this, topic is considered "known"
HIGH_CONFIDENCE_THRESHOLD = 0.8

# Curiosity update - exponential moving average weights
CURIOSITY_DECAY_WEIGHT = 0.9  # Old curiosity weight
CURIOSITY_UPDATE_WEIGHT = 0.1  # New complexity-interest weight

# =============================================================================
# Bayesian Surprise Constants (neuro_memory/surprise/bayesian_surprise.py)
# =============================================================================

# Window size for prior estimation
# Number of recent observations used to estimate the prior distribution
SURPRISE_WINDOW_SIZE = 50

# Surprise threshold for triggering memory encoding
# Observations with surprise above this trigger special handling
SURPRISE_ENCODING_THRESHOLD = 0.7

# Minimum observations needed before calculating meaningful surprise
MIN_SURPRISE_OBSERVATIONS = 10

# Exponential moving average smoothing factor for surprise tracking
SURPRISE_SMOOTHING_ALPHA = 0.1

# Adaptive threshold percentile
# When adaptive threshold is enabled, set threshold at this percentile
# of recent surprise values
BAYESIAN_SURPRISE_PERCENTILE = 75

# Observation variance for Bayesian update
# Default variance assumption when processing new observations
DEFAULT_OBSERVATION_VARIANCE = 0.1

# =============================================================================
# Episodic Memory Constants (neuro_memory/memory/episodic_store.py)
# =============================================================================

# Default importance value for new episodes
DEFAULT_EPISODE_IMPORTANCE = 0.5

# Importance decay factor - applied over time to reduce old memory importance
IMPORTANCE_DECAY_FACTOR = 0.99

# Forgetting background task interval (seconds)
# How often the background forgetting thread runs
FORGETTING_BACKGROUND_INTERVAL_SECONDS = 3600.0  # 1 hour

# Review threshold - retention level triggering review scheduling
REVIEW_RETENTION_THRESHOLD = 0.3

# Forgetting thread timeout (seconds)
FORGETTING_THREAD_TIMEOUT_SECONDS = 5.0

# High importance threshold - episodes above this get special handling
HIGH_IMPORTANCE_THRESHOLD = 0.6

# Initial retention calculation parameters
# Formula: initial_retention = 0.5 + (0.5 * importance)
INITIAL_RETENTION_BASE = 0.5
INITIAL_RETENTION_IMPORTANCE_FACTOR = 0.5

# Importance sigmoid scale - controls steepness of importance calculation
# Used in: importance = 1.0 / (1.0 + exp(-surprise + IMPORTANCE_SIGMOID_OFFSET))
IMPORTANCE_SIGMOID_OFFSET = 2.0

# =============================================================================
# Numerical Stability Constants
# =============================================================================

# Clip bounds for exponential operations to prevent overflow/underflow
EXP_CLIP_MIN = -500
EXP_CLIP_MAX = 500

# Small epsilon for avoiding division by zero in variance calculations
EPSILON = 1e-8

# Default value for uninitialized success rates
DEFAULT_SUCCESS_RATE = 0.0

# =============================================================================
# Predictive Model Constants (bayesian_surprise.py)
# =============================================================================

# Default hidden dimension for LSTM predictor
DEFAULT_LSTM_HIDDEN_DIM = 256

# Default number of LSTM layers
DEFAULT_LSTM_NUM_LAYERS = 2

# Dropout rate for multi-layer LSTM
LSTM_DROPOUT_RATE = 0.1

# =============================================================================
# LLM Context Window Constants
# =============================================================================

# Maximum characters for system prompt (including cognitive context)
MAX_SYSTEM_PROMPT_CHARS = 8000

# Maximum characters for user input
MAX_USER_INPUT_CHARS = 16000

# Maximum total context characters (~50k tokens)
MAX_CONTEXT_CHARS = 100000

# Maximum response characters to keep from LLM output
MAX_RESPONSE_CHARS = 16000

# =============================================================================
# Demo/Test Constants
# =============================================================================

# Random surprise multiplier for demo episode generation
DEMO_SURPRISE_MULTIPLIER = 3.0
