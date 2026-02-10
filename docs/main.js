/* =============================================
   Human Cognition AI - Shared JavaScript
   ============================================= */

// =====================================================
// MODULE DATA
// =====================================================
const modules = [
  { num: 1,  abbr: 'PC', name: 'Predictive Coding',   cat: 'foundation', desc: 'Hierarchical prediction error minimization following Karl Friston\'s free-energy principle. Generates top-down predictions and computes bottom-up prediction errors across cortical layers.', basis: 'Based on: Friston (2010), Rao & Ballard (1999)' },
  { num: 2,  abbr: 'MS', name: 'Memory Systems',      cat: 'foundation', desc: 'Four-store memory architecture: episodic memory (hippocampus-inspired), semantic memory (cortical networks), procedural memory (basal ganglia), and working memory (prefrontal buffer). Supports encoding, consolidation, and retrieval.', basis: 'Based on: Tulving (1985), Baddeley (2000)' },
  { num: 3,  abbr: 'WM', name: 'Working Memory',      cat: 'foundation', desc: 'Capacity-limited active maintenance buffer implementing the phonological loop, visuospatial sketchpad, and central executive. Supports chunking and attentional refresh.', basis: 'Based on: Baddeley & Hitch (1974)' },
  { num: 4,  abbr: 'PM', name: 'Procedural Memory',   cat: 'foundation', desc: 'Skill and habit learning through reinforcement. Stores action sequences as compiled productions that execute automatically with practice.', basis: 'Based on: Anderson (1982), Squire (2004)' },
  { num: 5,  abbr: 'PR', name: 'Perception',          cat: 'foundation', desc: 'Multi-modal sensory processing with hierarchical feature extraction. Integrates top-down predictions with bottom-up sensory signals to construct perceptual representations.', basis: 'Based on: Marr (1982), Helmholtz (1867)' },
  { num: 6,  abbr: 'DP', name: 'Dual Process',        cat: 'processing', desc: 'System 1 (fast, intuitive, automatic) and System 2 (slow, deliberate, effortful) thinking with dynamic switching based on task demands, novelty detection, and cognitive load.', basis: 'Based on: Kahneman (2011), Evans (2003)' },
  { num: 7,  abbr: 'EF', name: 'Executive Function',  cat: 'processing', desc: 'Cognitive control including task switching, inhibitory control, and working memory updating. Coordinates other modules and resolves conflicts between competing processes.', basis: 'Based on: Miyake et al. (2000), Miller & Cohen (2001)' },
  { num: 8,  abbr: 'DM', name: 'Decision Making',     cat: 'processing', desc: 'Multi-criteria decision evaluation using expected utility, prospect theory weighting, and satisficing. Integrates emotional valence through somatic markers.', basis: 'Based on: Damasio (1994), Tversky & Kahneman (1992)' },
  { num: 9,  abbr: 'AT', name: 'Attention',           cat: 'processing', desc: 'Selective, sustained, and divided attention mechanisms. Implements both bottom-up salience-driven and top-down goal-directed attention with competition resolution.', basis: 'Based on: Posner & Petersen (1990), Desimone & Duncan (1995)' },
  { num: 10, abbr: 'RS', name: 'Reasoning',           cat: 'processing', desc: 'Deductive, inductive, and abductive reasoning engines. Supports logical inference, probabilistic reasoning, and analogical mapping across domains.', basis: 'Based on: Johnson-Laird (1983), Gentner (1983)' },
  { num: 11, abbr: 'CU', name: 'Curiosity',           cat: 'motivation', desc: 'Information-gap detection and intrinsic motivation to explore. Balances novelty-seeking with information gain optimization using prediction error as a reward signal.', basis: 'Based on: Berlyne (1960), Loewenstein (1994)' },
  { num: 12, abbr: 'EM', name: 'Emotion',             cat: 'motivation', desc: 'Dimensional emotion modeling (valence, arousal, dominance) with somatic markers that bias cognition. Implements basic emotions, mood states, and affective forecasting.', basis: 'Based on: Damasio (1994), Russell (2003)' },
  { num: 13, abbr: 'SA', name: 'Self-Awareness',      cat: 'motivation', desc: 'Recursive self-modeling with metacognitive monitoring. Tracks confidence, detects errors, and adjusts cognitive strategies. Maintains a dynamic self-model.', basis: 'Based on: Flavell (1979), Fleming et al. (2012)' },
  { num: 14, abbr: 'GL', name: 'Goal Management',     cat: 'motivation', desc: 'Hierarchical goal decomposition, prioritization, and progress monitoring. Manages goal conflicts, subgoal generation, and dynamic reprioritization based on context.', basis: 'Based on: Carver & Scheier (1998), Austin & Vancouver (1996)' },
  { num: 15, abbr: 'MO', name: 'Motivation',          cat: 'motivation', desc: 'Drive system integrating homeostatic needs, incentive salience, and intrinsic motivation. Computes motivational intensity and direction for behavior selection.', basis: 'Based on: Hull (1943), Berridge (2004)' },
  { num: 16, abbr: 'LG', name: 'Language',            cat: 'interface', desc: 'Natural language processing including parsing, semantic interpretation, pragmatic reasoning, and generation. Supports context-dependent meaning construction and discourse tracking.', basis: 'Based on: Chomsky (1957), Tomasello (2003)' },
  { num: 17, abbr: 'SC', name: 'Social Cognition',    cat: 'interface', desc: 'Social norm understanding, reputation tracking, and cooperative behavior modeling. Enables navigation of social environments and group dynamics.', basis: 'Based on: Tomasello (2009), Dunbar (1998)' },
  { num: 18, abbr: 'EC', name: 'Embodied Cognition',  cat: 'interface', desc: 'Sensorimotor grounding of abstract concepts. Maintains body schema representations and couples cognition with simulated environmental interaction.', basis: 'Based on: Barsalou (1999), Clark (1997)' },
  { num: 19, abbr: 'TM', name: 'Theory of Mind',      cat: 'interface', desc: 'Mental state attribution for modeling beliefs, desires, and intentions of other agents. Supports perspective-taking and strategic social reasoning.', basis: 'Based on: Premack & Woodruff (1978), Baron-Cohen (1995)' },
  { num: 20, abbr: 'CR', name: 'Creativity',          cat: 'advanced', desc: 'Divergent thought generation through conceptual blending, analogical mapping, and constraint relaxation. Evaluates novelty and usefulness of generated ideas.', basis: 'Based on: Fauconnier & Turner (2002), Boden (2004)' },
  { num: 21, abbr: 'SL', name: 'Sleep Consolidation', cat: 'advanced', desc: 'Offline memory replay and synaptic homeostasis simulation. Consolidates episodic memories into semantic knowledge and prunes weak connections.', basis: 'Based on: Tononi & Cirelli (2006), Diekelmann & Born (2010)' },
  { num: 22, abbr: 'TP', name: 'Time Perception',     cat: 'advanced', desc: 'Internal clock mechanisms for duration estimation, temporal ordering, and prospective time management. Supports planning over multiple time horizons.', basis: 'Based on: Treisman (1963), Block & Zakay (1997)' },
  { num: 23, abbr: 'CB', name: 'Conceptual Blending', cat: 'advanced', desc: 'Cross-domain mapping and integration of mental spaces to generate novel conceptual structures. Enables metaphorical thinking and creative problem solving.', basis: 'Based on: Fauconnier & Turner (2002), Koestler (1964)' },
];

const catStyles = {
  foundation: { tile: 'tile-foundation', badge: 'bg-blue-100 text-blue-800' },
  processing: { tile: 'tile-processing', badge: 'bg-green-100 text-green-800' },
  motivation: { tile: 'tile-motivation', badge: 'bg-yellow-100 text-yellow-800' },
  interface:  { tile: 'tile-interface',  badge: 'bg-amber-100 text-amber-800' },
  advanced:   { tile: 'tile-advanced',   badge: 'bg-orange-100 text-orange-800' },
};

const catLabels = {
  foundation: 'Foundation',
  processing: 'Processing',
  motivation: 'Motivation',
  interface: 'Interface',
  advanced: 'Advanced',
};

// =====================================================
// MODULE GRID BUILDER
// =====================================================
function buildModuleGrid(containerId, panelId) {
  const grid = document.getElementById(containerId);
  if (!grid) return;
  let activeTile = null;

  modules.forEach((mod, idx) => {
    const tile = document.createElement('div');
    tile.className = `module-tile ${catStyles[mod.cat].tile}`;
    tile.innerHTML = `<span class="tile-num">${mod.num}</span><span class="tile-abbr">${mod.abbr}</span>`;
    tile.title = mod.name;
    tile.setAttribute('data-idx', idx);
    tile.addEventListener('click', () => {
      if (activeTile) activeTile.classList.remove('active');
      tile.classList.add('active');
      activeTile = tile;

      const panel = document.getElementById(panelId);
      if (!panel) return;
      const badge = panel.querySelector('.panel-badge');
      const title = panel.querySelector('.panel-title');
      const desc = panel.querySelector('.panel-desc');
      const basis = panel.querySelector('.panel-basis');
      if (badge) {
        badge.className = `panel-badge inline-block px-2 py-0.5 rounded text-xs font-semibold ${catStyles[mod.cat].badge}`;
        badge.textContent = catLabels[mod.cat];
      }
      if (title) title.textContent = mod.name;
      if (desc) desc.textContent = mod.desc;
      if (basis) basis.textContent = mod.basis;
      panel.classList.remove('hidden');
      panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    });
    grid.appendChild(tile);
  });
}

// =====================================================
// ARCHITECTURE LAYER ACCORDION
// =====================================================
function initArchAccordion() {
  document.querySelectorAll('.arch-layer').forEach(layer => {
    const header = layer.querySelector(':scope > .flex');
    if (header) {
      header.addEventListener('click', (e) => {
        e.stopPropagation();
        layer.classList.toggle('expanded');
        const chevron = layer.querySelector(':scope > .flex .arch-chevron');
        if (chevron) {
          chevron.style.transform = layer.classList.contains('expanded') ? 'rotate(180deg)' : 'rotate(0deg)';
        }
      });
    }
  });
}

// =====================================================
// MOBILE MENU
// =====================================================
function initMobileMenu() {
  const hamburger = document.getElementById('hamburger');
  const mobileMenu = document.getElementById('mobileMenu');
  if (!hamburger || !mobileMenu) return;
  hamburger.addEventListener('click', () => {
    mobileMenu.classList.toggle('open');
  });
  mobileMenu.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', () => {
      mobileMenu.classList.remove('open');
    });
  });
}

// =====================================================
// BIBTEX POPUP
// =====================================================
function initBibtexPopup() {
  const popup = document.getElementById('bibtexPopup');
  if (!popup) return;

  document.querySelectorAll('[data-action="cite"]').forEach(btn => {
    btn.addEventListener('click', () => popup.classList.add('show'));
  });

  const closeBtn = document.getElementById('closeBibtex');
  if (closeBtn) {
    closeBtn.addEventListener('click', () => popup.classList.remove('show'));
  }
  popup.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) popup.classList.remove('show');
  });

  const copyBtn = document.getElementById('copyBibtex');
  if (copyBtn) {
    copyBtn.addEventListener('click', () => {
      const bibtex = popup.querySelector('pre').textContent;
      navigator.clipboard.writeText(bibtex).then(() => {
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = 'Copy to Clipboard'; }, 2000);
      });
    });
  }
}

// =====================================================
// FAQ ACCORDION
// =====================================================
function initFAQ() {
  document.querySelectorAll('.faq-item').forEach(item => {
    const question = item.querySelector('.faq-question');
    if (question) {
      question.addEventListener('click', () => {
        item.classList.toggle('open');
      });
    }
  });
}

// =====================================================
// SMOOTH SCROLL
// =====================================================
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const targetId = this.getAttribute('href');
      if (targetId === '#') return;
      e.preventDefault();
      const target = document.querySelector(targetId);
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

// =====================================================
// SCROLL PROGRESS BAR
// =====================================================
function initScrollProgress() {
  const bar = document.getElementById('scrollProgress');
  if (!bar) return;
  window.addEventListener('scroll', () => {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    bar.style.width = progress + '%';
  }, { passive: true });
}

// =====================================================
// NAV SCROLL EFFECT
// =====================================================
function initNavScrollEffect() {
  const nav = document.getElementById('mainNav');
  if (!nav) return;
  let lastScroll = 0;
  window.addEventListener('scroll', () => {
    const currentScroll = window.scrollY;
    if (currentScroll > 20) {
      nav.classList.add('scrolled');
    } else {
      nav.classList.remove('scrolled');
    }
    lastScroll = currentScroll;
  }, { passive: true });
}

// =====================================================
// SCROLL REVEAL WITH STAGGER
// =====================================================
function initScrollReveal() {
  const reveals = document.querySelectorAll('.reveal');
  if (!reveals.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -40px 0px'
  });

  reveals.forEach(el => observer.observe(el));
}

// =====================================================
// ANIMATED COUNTERS
// =====================================================
function initCounters() {
  const counters = document.querySelectorAll('[data-counter]');
  if (!counters.length) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.5 });

  counters.forEach(el => observer.observe(el));
}

function animateCounter(el) {
  const target = parseInt(el.getAttribute('data-counter'));
  const suffix = el.getAttribute('data-suffix') || '';
  const prefix = el.getAttribute('data-prefix') || '';
  const duration = 1800;
  const startTime = performance.now();

  function update(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    const current = Math.round(eased * target);
    el.textContent = prefix + current + suffix;
    if (progress < 1) {
      requestAnimationFrame(update);
    }
  }
  requestAnimationFrame(update);
}

// =====================================================
// PARALLAX
// =====================================================
function initParallax() {
  const parallaxEls = document.querySelectorAll('[data-parallax]');
  if (!parallaxEls.length) return;

  window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    parallaxEls.forEach(el => {
      const speed = parseFloat(el.getAttribute('data-parallax')) || 0.1;
      const rect = el.getBoundingClientRect();
      const visible = rect.top < window.innerHeight && rect.bottom > 0;
      if (visible) {
        const offset = (scrollY - el.offsetTop + window.innerHeight) * speed;
        el.style.transform = `translateY(${offset}px)`;
      }
    });
  }, { passive: true });
}

// =====================================================
// DYNAMIC YEAR
// =====================================================
function initDynamicYear() {
  const yearEls = document.querySelectorAll('[data-year]');
  const currentYear = new Date().getFullYear();
  yearEls.forEach(el => {
    el.textContent = currentYear;
  });
}

// =====================================================
// API CONNECTION
// =====================================================
const API_BASE = 'https://zedigital-human-cognition.fly.dev';
let apiConnected = false;

const scenarioKeyMap = [
  'sentence_processing',
  'memory_recall',
  'financial_decision',
  'word_learning',
  'danger_response',
];

const apiModuleToSvg = {
  prediction: 'sensory',
  memory: 'memory',
  learning: 'longterm_memory',
  dual_process: 'dualprocess',
  executive: 'executive',
  reasoning: 'cortical',
  motivation: 'decision',
  emotion: 'emotion',
  self_awareness: 'executive',
  language: 'language',
  social: 'cortical',
  embodied: 'motor',
  creativity: 'cortical',
  cognitive_maps: 'longterm_memory',
  time: 'cortical',
  sleep: 'longterm_memory',
};

async function checkApiConnection() {
  const dot = document.getElementById('apiDot');
  const label = document.getElementById('apiLabel');
  if (!dot || !label) return false;
  try {
    const res = await fetch(API_BASE + '/api/health', { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const data = await res.json();
      apiConnected = true;
      dot.className = 'w-2 h-2 rounded-full bg-green-500';
      label.textContent = data.agent_available ? 'Live (Real Engine)' : 'Live (Fallback)';
      const status = document.getElementById('apiStatus');
      if (status) status.className = 'flex items-center gap-1.5 text-xs text-green-600';
      return true;
    }
  } catch (e) {}
  apiConnected = false;
  dot.className = 'w-2 h-2 rounded-full bg-gray-300';
  label.textContent = 'Offline';
  const status = document.getElementById('apiStatus');
  if (status) status.className = 'flex items-center gap-1.5 text-xs text-gray-400';
  return false;
}

async function apiSimulate(scenarioKey) {
  const res = await fetch(API_BASE + '/api/simulate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario: scenarioKey }),
    signal: AbortSignal.timeout(15000),
  });
  if (!res.ok) throw new Error('API error: ' + res.status);
  return await res.json();
}

// =====================================================
// BRAIN SIMULATOR MODULE INFO
// =====================================================
const modulePositions = {
  sensory:         { x: 250, y: 26,  label: 'Sensory Input' },
  motor:           { x: 250, y: 474, label: 'Motor Output' },
  memory:          { x: 107, y: 152, label: 'Memory' },
  emotion:         { x: 415, y: 313, label: 'Emotion Engine' },
  language:        { x: 415, y: 143, label: 'Language' },
  dualprocess:     { x: 100, y: 333, label: 'Dual Process' },
  executive:       { x: 250, y: 250, label: 'Global Workspace' },
  working_memory:  { x: 155, y: 200, label: 'Working Memory' },
  decision:        { x: 170, y: 320, label: 'Decision Module' },
  amygdala:        { x: 340, y: 350, label: 'Amygdala' },
  longterm_memory: { x: 120, y: 100, label: 'Long-term Memory' },
  cortical:        { x: 330, y: 150, label: 'Cortical Processing' },
};

const moduleInfo = {
  sensory: {
    name: 'Sensory Input',
    desc: 'Multi-modal sensory processing interface. Receives raw environmental signals (visual, auditory, tactile) and converts them into internal representations.',
    role: 'Gateway for all external information entering the cognitive system.',
    params: [
      { name: 'Processing Speed', value: 0.15, min: 0.05, max: 0.5, unit: 's', key: 'speed' },
      { name: 'Sensitivity', value: 0.8, min: 0.1, max: 1.0, unit: '', key: 'sensitivity' },
    ],
    connections: ['language', 'memory', 'executive', 'amygdala', 'cortical'],
    color: '#22C55E',
  },
  motor: {
    name: 'Motor Output',
    desc: 'Action execution and response generation. Translates cognitive decisions into behavioral output.',
    role: 'Final output pathway for all behavioral responses.',
    params: [
      { name: 'Response Latency', value: 0.2, min: 0.05, max: 0.5, unit: 's', key: 'speed' },
      { name: 'Precision', value: 0.9, min: 0.3, max: 1.0, unit: '', key: 'precision' },
    ],
    connections: ['executive', 'dualprocess', 'amygdala'],
    color: '#22C55E',
  },
  memory: {
    name: 'Memory Systems',
    desc: 'Integrated memory architecture combining episodic, semantic, and procedural memory stores.',
    role: 'Stores and retrieves all learned information across multiple memory types.',
    params: [
      { name: 'Retrieval Speed', value: 0.3, min: 0.1, max: 0.8, unit: 's', key: 'speed' },
      { name: 'Capacity', value: 7, min: 3, max: 12, unit: 'items', key: 'capacity' },
      { name: 'Encoding Strength', value: 0.7, min: 0.2, max: 1.0, unit: '', key: 'encoding' },
    ],
    connections: ['executive', 'working_memory', 'emotion', 'language', 'longterm_memory'],
    color: '#3B82F6',
  },
  emotion: {
    name: 'Emotion Engine',
    desc: 'Dimensional emotion processing with valence, arousal, and dominance axes. Implements somatic markers.',
    role: 'Provides emotional valence to all cognitive processes.',
    params: [
      { name: 'Processing Speed', value: 0.25, min: 0.1, max: 0.6, unit: 's', key: 'speed' },
      { name: 'Reactivity', value: 0.6, min: 0.1, max: 1.0, unit: '', key: 'reactivity' },
      { name: 'Regulation', value: 0.5, min: 0.1, max: 1.0, unit: '', key: 'regulation' },
    ],
    connections: ['executive', 'memory', 'amygdala', 'decision', 'language'],
    color: '#EF4444',
  },
  language: {
    name: 'Language Processing',
    desc: 'Full natural language pipeline: phonological analysis, syntactic parsing, semantic interpretation, and generation.',
    role: 'Encodes and decodes linguistic information for communication and inner speech.',
    params: [
      { name: 'Parse Speed', value: 0.3, min: 0.1, max: 0.6, unit: 's', key: 'speed' },
      { name: 'Vocabulary Size', value: 50000, min: 10000, max: 100000, unit: 'words', key: 'vocab' },
    ],
    connections: ['sensory', 'memory', 'executive', 'cortical'],
    color: '#A855F7',
  },
  dualprocess: {
    name: 'Dual Process Thinking',
    desc: 'System 1 (fast, automatic, heuristic) and System 2 (slow, deliberate, analytical) with dynamic switching.',
    role: 'Selects between fast intuitive and slow analytical processing modes.',
    params: [
      { name: 'System 1 Speed', value: 0.1, min: 0.05, max: 0.3, unit: 's', key: 'sys1_speed' },
      { name: 'System 2 Speed', value: 0.8, min: 0.3, max: 2.0, unit: 's', key: 'sys2_speed' },
      { name: 'Switch Threshold', value: 0.5, min: 0.2, max: 0.9, unit: '', key: 'threshold' },
    ],
    connections: ['executive', 'working_memory', 'decision', 'motor'],
    color: '#EAB308',
  },
  executive: {
    name: 'Global Workspace',
    desc: 'Central hub implementing Baars\' Global Workspace Theory. Broadcasts information across all modules.',
    role: 'Central coordinator that integrates and broadcasts information across all modules.',
    params: [
      { name: 'Broadcast Speed', value: 0.2, min: 0.05, max: 0.5, unit: 's', key: 'speed' },
      { name: 'Broadcast Capacity', value: 4, min: 1, max: 8, unit: 'streams', key: 'capacity' },
    ],
    connections: ['sensory', 'motor', 'memory', 'emotion', 'language', 'dualprocess', 'working_memory', 'decision'],
    color: '#F97316',
  },
  working_memory: {
    name: 'Working Memory',
    desc: 'Capacity-limited active maintenance buffer. Phonological loop, visuospatial sketchpad, and episodic buffer.',
    role: 'Short-term active storage for items currently being processed.',
    params: [
      { name: 'Capacity', value: 7, min: 3, max: 12, unit: 'items', key: 'capacity' },
      { name: 'Decay Rate', value: 0.3, min: 0.1, max: 0.8, unit: '/s', key: 'decay' },
    ],
    connections: ['executive', 'memory', 'dualprocess', 'longterm_memory'],
    color: '#3B82F6',
  },
  decision: {
    name: 'Decision Module',
    desc: 'Multi-criteria decision evaluation with expected utility, prospect theory, and somatic marker integration.',
    role: 'Evaluates options and selects actions based on integrated cognitive/emotional inputs.',
    params: [
      { name: 'Deliberation Time', value: 0.4, min: 0.1, max: 1.0, unit: 's', key: 'speed' },
      { name: 'Risk Tolerance', value: 0.5, min: 0.0, max: 1.0, unit: '', key: 'risk' },
    ],
    connections: ['executive', 'emotion', 'dualprocess', 'motor'],
    color: '#EAB308',
  },
  amygdala: {
    name: 'Amygdala (Fast Path)',
    desc: 'Rapid threat-detection circuit that bypasses cortical processing for survival-critical stimuli.',
    role: 'Ultra-fast emotional detection for threats, bypassing higher-order processing.',
    params: [
      { name: 'Response Speed', value: 0.08, min: 0.03, max: 0.2, unit: 's', key: 'speed' },
      { name: 'Threat Sensitivity', value: 0.9, min: 0.3, max: 1.0, unit: '', key: 'sensitivity' },
    ],
    connections: ['sensory', 'motor', 'emotion', 'cortical'],
    color: '#EF4444',
  },
  longterm_memory: {
    name: 'Long-term Memory',
    desc: 'Persistent storage for consolidated memories. Episodic and semantic memories with associative retrieval.',
    role: 'Permanent storage and retrieval of consolidated knowledge and experiences.',
    params: [
      { name: 'Encoding Speed', value: 0.5, min: 0.2, max: 1.0, unit: 's', key: 'speed' },
      { name: 'Consolidation Strength', value: 0.7, min: 0.2, max: 1.0, unit: '', key: 'strength' },
    ],
    connections: ['memory', 'working_memory', 'executive'],
    color: '#3B82F6',
  },
  cortical: {
    name: 'Cortical Processing',
    desc: 'Higher-order cortical analysis including pattern recognition, abstract reasoning, and stimulus evaluation.',
    role: 'Detailed, high-fidelity processing of complex stimuli and abstract thought.',
    params: [
      { name: 'Processing Depth', value: 0.6, min: 0.2, max: 1.0, unit: 's', key: 'speed' },
      { name: 'Abstraction Level', value: 0.7, min: 0.1, max: 1.0, unit: '', key: 'abstraction' },
    ],
    connections: ['sensory', 'language', 'executive', 'amygdala'],
    color: '#A855F7',
  },
};

const paramOverrides = {};

// =====================================================
// SCENARIOS
// =====================================================
const scenarios = [
  {
    name: 'Process a Sentence',
    title: '"The cat sat on the mat"',
    duration: 3000,
    steps: [
      { module: 'sensory',   delay: 0,    duration: 400, input: 'Auditory waveform received', output: 'Phoneme sequence extracted' },
      { module: 'language',  delay: 400,  duration: 500, input: 'Phoneme stream incoming', output: 'Parse tree: [S [NP the cat] [VP sat [PP on [NP the mat]]]]' },
      { module: 'memory',    delay: 900,  duration: 400, input: 'Semantic query: "cat", "mat"', output: 'Retrieved: cat=feline, mat=surface, sat=past(sit)' },
      { module: 'executive', delay: 1300, duration: 500, input: 'Integrated meaning frame', output: 'Broadcast: "Agent(cat) Action(sit) Location(mat)"' },
      { module: 'motor',     delay: 1800, duration: 400, input: 'Comprehension confirmed', output: 'Response: understanding acknowledged' },
    ],
    thought: 'The auditory signal was parsed into phonemes, then syntactically structured. Semantic memory retrieved meanings which were integrated in the Global Workspace into a coherent situational model.',
    humanTime: 300,
  },
  {
    name: 'Recall a Childhood Memory',
    title: 'Episodic retrieval with emotional coloring',
    duration: 4000,
    steps: [
      { module: 'executive',       delay: 0,    duration: 500, input: 'Goal: retrieve childhood memory', output: 'Search cue: temporal=childhood, valence=positive' },
      { module: 'memory',          delay: 500,  duration: 600, input: 'Episodic search: childhood + positive', output: 'Match: "Birthday party, age 7"' },
      { module: 'longterm_memory', delay: 600,  duration: 500, input: 'Deep retrieval: context expansion', output: 'Details: cake, candles, garden, laughter' },
      { module: 'emotion',         delay: 1100, duration: 700, input: 'Memory content + somatic markers', output: 'Emotion: nostalgia (valence=0.8, arousal=0.4)' },
      { module: 'executive',       delay: 1800, duration: 500, input: 'Emotional memory integration', output: 'Broadcast: vivid episodic memory with emotional coloring' },
      { module: 'language',        delay: 2300, duration: 600, input: 'Memory + emotion bundle', output: 'Verbal: "I remember my 7th birthday..."' },
    ],
    thought: 'Executive control initiated a directed memory search. Episodic memory returned a scene, expanded with contextual details. The emotion engine colored the memory with nostalgia.',
    humanTime: 500,
  },
  {
    name: 'Financial Decision',
    title: 'System 1 vs System 2 conflict',
    duration: 5000,
    steps: [
      { module: 'sensory',        delay: 0,    duration: 400, input: 'Visual: investment data', output: 'Parsed: 18% return, high volatility' },
      { module: 'dualprocess',    delay: 400,  duration: 300, input: 'High-return investment', output: 'System 1: "Invest now!" (excitement bias)' },
      { module: 'dualprocess',    delay: 700,  duration: 800, input: 'Override: System 2 engaged', output: 'System 2: Analyzing risk/reward ratio...' },
      { module: 'working_memory', delay: 1500, duration: 500, input: 'Hold: return=18%, risk=high', output: 'Comparing: portfolio risk, liquidity needs' },
      { module: 'emotion',        delay: 2000, duration: 500, input: 'Somatic marker check', output: 'Gut: anxiety + excitement conflict' },
      { module: 'decision',       delay: 2500, duration: 600, input: 'System 2 + somatic markers', output: 'Decision: invest 30% (compromise)' },
      { module: 'executive',      delay: 3100, duration: 400, input: 'Decision result', output: 'Broadcast: partial investment, confidence=0.72' },
      { module: 'motor',          delay: 3500, duration: 400, input: 'Execute decision', output: 'Action: submit order for 30%' },
    ],
    thought: 'System 1 impulse met System 2 analysis. The emotion engine contributed conflicting somatic markers. The decision module resolved with a compromise.',
    humanTime: 2000,
  },
  {
    name: 'Learn a New Word',
    title: 'Language acquisition through memory encoding',
    duration: 3500,
    steps: [
      { module: 'sensory',         delay: 0,    duration: 350, input: 'Auditory: "Petrichor means..."', output: 'Phonemes: /petrikhor/' },
      { module: 'language',        delay: 350,  duration: 500, input: 'Unknown word: "petrichor"', output: 'Parsed: noun, "smell of rain on dry earth"' },
      { module: 'working_memory',  delay: 850,  duration: 500, input: 'New entry: petrichor', output: 'Rehearsing: phonological loop active' },
      { module: 'longterm_memory', delay: 1350, duration: 700, input: 'Encode: petrichor -> semantic net', output: 'Linked to: rain, earth, smell, nature' },
      { module: 'executive',       delay: 2050, duration: 500, input: 'Encoding confirmed', output: 'New word acquired, needs consolidation' },
    ],
    thought: 'Novel phoneme sequence identified. Working memory rehearsed the form-meaning pair. Long-term memory encoded by linking to existing semantic nodes.',
    humanTime: 400,
  },
  {
    name: 'Respond to Danger',
    title: 'Amygdala fast-path, fight-or-flight',
    duration: 2000,
    steps: [
      { module: 'sensory',  delay: 0,   duration: 150, input: 'Visual: large shape approaching fast', output: 'Threat signal: size=large, velocity=high' },
      { module: 'amygdala', delay: 150, duration: 200, input: 'THREAT (fast path, bypassing cortex)', output: 'ALARM: fight-or-flight, adrenaline surge' },
      { module: 'motor',    delay: 350, duration: 250, input: 'Amygdala emergency signal', output: 'IMMEDIATE: jump back, arms up' },
      { module: 'cortical', delay: 200, duration: 600, input: 'Parallel slow path analyzing', output: 'Object identified: a ball, not threat' },
      { module: 'executive', delay: 800, duration: 400, input: 'Cortical override + amygdala', output: 'False alarm, downregulate response' },
      { module: 'emotion',  delay: 1200, duration: 300, input: 'Resolution signal', output: 'Relief (0.6), residual arousal declining' },
    ],
    thought: 'Amygdala fast-path triggered fight-or-flight in under 350ms. Slower cortical pathway identified false alarm. Global Workspace broadcast resolution.',
    humanTime: 150,
  },
];

// =====================================================
// SIMULATOR ENGINE
// =====================================================
let selectedScenario = 0;
let isSimulating = false;
let simTimeouts = [];

function initScenarioCards() {
  document.querySelectorAll('.scenario-card').forEach(card => {
    card.addEventListener('click', () => {
      if (isSimulating) return;
      document.querySelectorAll('.scenario-card').forEach(c => c.classList.remove('selected'));
      card.classList.add('selected');
      selectedScenario = parseInt(card.dataset.scenario);
    });
  });
}

function inspectModule(moduleKey) {
  if (isSimulating) return;
  const info = moduleInfo[moduleKey];
  if (!info) return;

  const panel = document.getElementById('inspectorPanel');
  const placeholder = document.getElementById('inspectorPlaceholder');
  if (!panel) return;
  if (placeholder) placeholder.classList.add('hidden');

  const connectionsHtml = info.connections.map(c => {
    const ci = moduleInfo[c];
    return `<span class="inline-block px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-700 mr-1 mb-1">${ci ? ci.name : c}</span>`;
  }).join('');

  const paramsHtml = info.params.map(p => {
    const override = paramOverrides[moduleKey + '_' + p.key];
    const currentVal = override !== undefined ? override : p.value;
    const displayVal = (p.unit === 'words' || p.unit === 'items' || p.unit === 'streams') ? Math.round(currentVal) : currentVal.toFixed(2);
    return `
      <div class="mb-3">
        <div class="flex justify-between text-xs mb-1">
          <span class="font-medium text-gray-700">${p.name}</span>
          <span class="text-accent font-mono" id="paramVal_${moduleKey}_${p.key}">${displayVal}${p.unit ? ' ' + p.unit : ''}</span>
        </div>
        <input type="range" class="w-full" min="${p.min}" max="${p.max}" step="${(p.max - p.min) / 100}" value="${currentVal}"
          onInput="updateParam('${moduleKey}', '${p.key}', this.value, '${p.unit}', ${p.min}, ${p.max})">
      </div>
    `;
  }).join('');

  panel.innerHTML = `
    <div class="p-4">
      <div class="flex items-center gap-2 mb-2">
        <div class="w-3 h-3 rounded-full" style="background:${info.color}"></div>
        <h4 class="font-serif text-base font-bold text-navy">${info.name}</h4>
      </div>
      <p class="text-xs text-gray-600 leading-relaxed mb-3">${info.desc}</p>
      <p class="text-xs text-gray-400 italic mb-4">${info.role}</p>
      <div class="border-t border-gray-100 pt-3 mb-3">
        <div class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Parameters</div>
        ${paramsHtml}
      </div>
      <div class="border-t border-gray-100 pt-3">
        <div class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Connections</div>
        <div class="flex flex-wrap">${connectionsHtml}</div>
      </div>
    </div>
  `;

  panel.classList.add('open');
}

function updateParam(moduleKey, paramKey, value, unit) {
  const numVal = parseFloat(value);
  paramOverrides[moduleKey + '_' + paramKey] = numVal;
  const display = (unit === 'words' || unit === 'items' || unit === 'streams') ? Math.round(numVal) : numVal.toFixed(2);
  const el = document.getElementById('paramVal_' + moduleKey + '_' + paramKey);
  if (el) el.textContent = display + (unit ? ' ' + unit : '');
}

function lightModule(moduleKey) {
  const el = document.querySelector(`.sim-module[data-module="${moduleKey}"]`);
  if (el) {
    el.classList.add('lit', 'active-module');
    const info = moduleInfo[moduleKey];
    if (info) {
      const colorMap = {
        '#22C55E': 'url(#glowGreen)',
        '#3B82F6': 'url(#glowBlue)',
        '#EF4444': 'url(#glowRed)',
        '#A855F7': 'url(#glowPurple)',
        '#F97316': 'url(#glowOrange)',
        '#EAB308': 'url(#glowYellow)',
      };
      el.style.filter = colorMap[info.color] || 'url(#glowOrange)';
    }
  }
}

function unlightModule(moduleKey) {
  const el = document.querySelector(`.sim-module[data-module="${moduleKey}"]`);
  if (el) {
    el.classList.remove('lit', 'active-module');
    el.style.filter = '';
  }
}

function unlightAllModules() {
  document.querySelectorAll('.sim-module').forEach(el => {
    el.classList.remove('lit', 'active-module');
    el.style.filter = '';
  });
}

function drawFlowArrow(fromKey, toKey) {
  const from = modulePositions[fromKey];
  const to = modulePositions[toKey];
  if (!from || !to) return null;

  const arrowGroup = document.getElementById('flowArrows');
  if (!arrowGroup) return null;

  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const offsetFrom = 35;
  const offsetTo = 35;

  const x1 = from.x + (dx / dist) * offsetFrom;
  const y1 = from.y + (dy / dist) * offsetFrom;
  const x2 = to.x - (dx / dist) * offsetTo;
  const y2 = to.y - (dy / dist) * offsetTo;

  const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  line.setAttribute('x1', x1);
  line.setAttribute('y1', y1);
  line.setAttribute('x2', x2);
  line.setAttribute('y2', y2);
  line.setAttribute('stroke', '#F97316');
  line.setAttribute('stroke-width', '2.5');
  line.setAttribute('marker-end', 'url(#arrowOrange)');
  line.classList.add('flow-arrow');
  arrowGroup.appendChild(line);

  return line;
}

function clearFlowArrows() {
  const arrowGroup = document.getElementById('flowArrows');
  if (arrowGroup) arrowGroup.innerHTML = '';
}

function addLogEntry(time, module, text, isError) {
  const log = document.getElementById('simLog');
  if (!log) return;
  const entry = document.createElement('div');
  entry.className = 'sim-log-entry mb-1';
  const color = isError ? 'text-red-500' : 'text-gray-700';
  const moduleName = moduleInfo[module] ? moduleInfo[module].name : module;
  entry.innerHTML = `<span class="text-accent font-semibold">[${time.toFixed(1)}s]</span> <span class="font-semibold ${color}">${moduleName}:</span> <span class="text-gray-600">${text}</span>`;
  log.appendChild(entry);
  requestAnimationFrame(() => {
    entry.classList.add('visible');
    log.scrollTop = log.scrollHeight;
  });
}

function initSimulator() {
  const runBtn = document.getElementById('runSimBtn');
  if (!runBtn) return;

  initScenarioCards();

  runBtn.addEventListener('click', () => {
    if (isSimulating) return;
    runSimulation();
  });
}

async function runSimulation() {
  isSimulating = true;
  const btn = document.getElementById('runSimBtn');
  if (btn) {
    btn.textContent = 'Simulating...';
    btn.classList.add('opacity-60', 'cursor-not-allowed');
  }

  unlightAllModules();
  clearFlowArrows();
  const simLog = document.getElementById('simLog');
  if (simLog) simLog.innerHTML = '';
  const resultsPanel = document.getElementById('resultsPanel');
  if (resultsPanel) resultsPanel.classList.add('hidden');

  const scenario = scenarios[selectedScenario];
  simTimeouts = [];

  if (apiConnected) {
    try {
      addLogEntry(0, 'executive', 'Connecting to real cognitive engine...');
      const scenarioKey = scenarioKeyMap[selectedScenario];
      const apiResult = await apiSimulate(scenarioKey);
      runApiVisualization(apiResult, scenario);
      return;
    } catch (e) {
      if (simLog) simLog.innerHTML = '';
      addLogEntry(0, 'executive', 'API unavailable, running client-side simulation.');
    }
  }

  runClientSimulation(scenario);
}

function runApiVisualization(apiResult, fallbackScenario) {
  const steps = apiResult.steps || [];
  const engineLabel = apiResult.engine === 'real' ? 'Real Cognitive Engine' : 'Fallback Engine';
  addLogEntry(0, 'executive', 'Engine: ' + engineLabel + ' | Scenario: ' + apiResult.scenario_name);

  let cumDelay = 0;
  const stepDelay = 600;
  let prevSvgModule = null;

  steps.forEach((step) => {
    const svgModule = apiModuleToSvg[step.module] || 'executive';
    const delay = cumDelay;

    const t1 = setTimeout(() => {
      lightModule(svgModule);
      addLogEntry(delay / 1000, svgModule, `[Phase ${step.phase}: ${step.phase_name}] ${step.module_name}: ${step.action}`);
      if (prevSvgModule && prevSvgModule !== svgModule) {
        const arrow = drawFlowArrow(prevSvgModule, svgModule);
        if (arrow) setTimeout(() => arrow.classList.add('active-flow'), 50);
      }
      prevSvgModule = svgModule;
    }, delay);
    simTimeouts.push(t1);

    const t2 = setTimeout(() => {
      addLogEntry((delay + stepDelay * 0.7) / 1000, svgModule, step.output);
      setTimeout(() => unlightModule(svgModule), 200);
    }, delay + stepDelay * 0.7);
    simTimeouts.push(t2);

    cumDelay += stepDelay;
  });

  const totalVizTime = cumDelay + 500;
  const t3 = setTimeout(() => {
    addLogEntry(totalVizTime / 1000, 'executive', 'Simulation complete (' + engineLabel + ').');
    unlightAllModules();
    showApiResults(apiResult, fallbackScenario);
    isSimulating = false;
    const btn = document.getElementById('runSimBtn');
    if (btn) {
      btn.textContent = 'Run Simulation';
      btn.classList.remove('opacity-60', 'cursor-not-allowed');
    }
  }, totalVizTime);
  simTimeouts.push(t3);
}

function runClientSimulation(scenario) {
  const steps = scenario.steps;

  steps.forEach((step, idx) => {
    const t1 = setTimeout(() => {
      lightModule(step.module);
      addLogEntry(step.delay / 1000, step.module, step.input);
      if (idx > 0) {
        const prevKey = steps[idx - 1].module;
        if (prevKey !== step.module) {
          const arrow = drawFlowArrow(prevKey, step.module);
          if (arrow) setTimeout(() => arrow.classList.add('active-flow'), 50);
        }
      }
    }, step.delay);
    simTimeouts.push(t1);

    const t2 = setTimeout(() => {
      addLogEntry((step.delay + step.duration) / 1000, step.module, step.output);
      setTimeout(() => unlightModule(step.module), 200);
    }, step.delay + step.duration);
    simTimeouts.push(t2);
  });

  const t3 = setTimeout(() => {
    addLogEntry(scenario.duration / 1000, 'executive', 'Simulation complete (client-side).');
    unlightAllModules();
    showResults(scenario);
    isSimulating = false;
    const btn = document.getElementById('runSimBtn');
    if (btn) {
      btn.textContent = 'Run Simulation';
      btn.classList.remove('opacity-60', 'cursor-not-allowed');
    }
  }, scenario.duration + 500);
  simTimeouts.push(t3);
}

function showApiResults(apiResult, fallbackScenario) {
  const panel = document.getElementById('resultsPanel');
  const timings = document.getElementById('resultsTimings');
  const thought = document.getElementById('resultsThought');
  const comparison = document.getElementById('resultsComparison');
  if (!panel) return;

  const apiSteps = apiResult.steps || [];
  const maxDuration = Math.max(...apiSteps.map(s => s.duration_ms || 1), 1);
  let timingsHtml = '';
  apiSteps.forEach(step => {
    const dur = step.duration_ms || 0;
    const pct = (dur / maxDuration) * 100;
    const svgMod = apiModuleToSvg[step.module] || 'executive';
    const info = moduleInfo[svgMod];
    const color = info ? info.color : '#F97316';
    timingsHtml += `
      <div class="flex items-center gap-3 text-xs">
        <div class="w-28 font-medium text-gray-700 truncate">${step.module_name}</div>
        <div class="flex-1 results-bar"><div class="results-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <div class="w-14 text-right font-mono text-gray-500">${dur.toFixed(1)}ms</div>
      </div>`;
  });
  if (timings) timings.innerHTML = timingsHtml;

  const engineTag = apiResult.engine === 'real'
    ? '<span class="inline-block px-2 py-0.5 rounded text-xs font-semibold bg-green-100 text-green-800 mr-2">Real Engine</span>'
    : '<span class="inline-block px-2 py-0.5 rounded text-xs font-semibold bg-yellow-100 text-yellow-800 mr-2">Fallback Engine</span>';

  const phasesStr = (apiResult.phases_traversed || []).map(p => 'Phase ' + p).join(' -> ');
  if (thought) {
    thought.innerHTML = `
      <div class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Processing Details</div>
      <div class="mb-2">${engineTag} Phases: ${phasesStr}</div>
      <p class="text-sm text-gray-700 leading-relaxed">${fallbackScenario.thought}</p>`;
  }

  const totalMs = apiResult.total_duration_ms || 0;
  if (comparison) {
    comparison.innerHTML = `
      <div class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Performance</div>
      <div>Cognitive engine: ${totalMs.toFixed(1)}ms</div>
      <div>Human brain: ~${fallbackScenario.humanTime}ms</div>`;
  }

  panel.classList.remove('hidden');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showResults(scenario) {
  const panel = document.getElementById('resultsPanel');
  const timings = document.getElementById('resultsTimings');
  const thought = document.getElementById('resultsThought');
  const comparison = document.getElementById('resultsComparison');
  if (!panel) return;

  const maxDuration = Math.max(...scenario.steps.map(s => s.duration));
  let timingsHtml = '';
  scenario.steps.forEach(step => {
    const info = moduleInfo[step.module];
    const pct = (step.duration / maxDuration) * 100;
    const color = info ? info.color : '#F97316';
    timingsHtml += `
      <div class="flex items-center gap-3 text-xs">
        <div class="w-28 font-medium text-gray-700 truncate">${info ? info.name : step.module}</div>
        <div class="flex-1 results-bar"><div class="results-bar-fill" style="width:${pct}%;background:${color}"></div></div>
        <div class="w-14 text-right font-mono text-gray-500">${step.duration}ms</div>
      </div>`;
  });
  if (timings) timings.innerHTML = timingsHtml;

  if (thought) {
    thought.innerHTML = `
      <div class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Thought Process</div>
      <span class="inline-block px-2 py-0.5 rounded text-xs font-semibold bg-gray-100 text-gray-600 mb-2">Client-Side Simulation</span>
      <p class="text-sm text-gray-700 leading-relaxed">${scenario.thought}</p>`;
  }

  if (comparison) {
    comparison.innerHTML = `
      <div class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Performance Comparison</div>
      <div>Human brain: ~${scenario.humanTime}ms</div>
      <div>Simulation: ${scenario.duration}ms (${(scenario.duration/1000).toFixed(1)}s visual)</div>
      <div class="mt-1 text-gray-400">Actual neural processing is massively parallel. This serializes steps for educational clarity.</div>`;
  }

  panel.classList.remove('hidden');
  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// =====================================================
// INIT ON DOM READY
// =====================================================
document.addEventListener('DOMContentLoaded', () => {
  initMobileMenu();
  initArchAccordion();
  initBibtexPopup();
  initFAQ();
  initSmoothScroll();
  initSimulator();
  initScrollProgress();
  initNavScrollEffect();
  initScrollReveal();
  initCounters();
  initParallax();
  initDynamicYear();

  if (document.getElementById('moduleGrid')) {
    buildModuleGrid('moduleGrid', 'modulePanel');
  }

  if (document.getElementById('apiDot')) {
    checkApiConnection();
    setInterval(checkApiConnection, 30000);
  }
});
