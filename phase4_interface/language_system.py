"""
Language Processing System - Grounded Language Understanding

Implements:
1. Compositional Semantics (meaning from parts)
2. Syntax-Semantics Interface
3. Pragmatics (context-dependent meaning)
4. Grounded Language (language tied to perception/action)
5. Discourse Processing (coherent conversation)
6. Language Production (generation)

Based on research:
- Construction Grammar (Goldberg)
- Embodied semantics (Barsalou, Glenberg)
- Discourse Representation Theory
- Pragmatics (Grice, Sperber & Wilson)

Performance: Efficient parsing, O(n) for simple structures
Comparison vs existing:
- LLMs: Statistical patterns but no grounding
- Symbolic NLP: Rules but no flexibility
- This: Grounded, compositional, flexible
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import re


class SyntacticCategory(Enum):
    """Basic syntactic categories."""
    NOUN = auto()
    VERB = auto()
    ADJECTIVE = auto()
    ADVERB = auto()
    PREPOSITION = auto()
    DETERMINER = auto()
    PRONOUN = auto()
    CONJUNCTION = auto()
    INTERJECTION = auto()


class SemanticRole(Enum):
    """Thematic/semantic roles."""
    AGENT = auto()      # Doer of action
    PATIENT = auto()    # Affected by action
    THEME = auto()      # Thing moved/changed
    EXPERIENCER = auto()  # One who experiences
    INSTRUMENT = auto()  # Tool used
    LOCATION = auto()   # Where
    SOURCE = auto()     # From where
    GOAL = auto()       # To where
    TIME = auto()       # When
    MANNER = auto()     # How
    CAUSE = auto()      # Why


class SpeechAct(Enum):
    """Types of speech acts (Austin/Searle)."""
    ASSERT = auto()      # Stating a fact
    QUESTION = auto()    # Asking
    COMMAND = auto()     # Ordering
    REQUEST = auto()     # Asking politely
    PROMISE = auto()     # Committing to action
    WARN = auto()        # Alerting to danger
    THANK = auto()       # Expressing gratitude
    APOLOGIZE = auto()   # Expressing regret
    GREET = auto()       # Social greeting


@dataclass
class Word:
    """A word with its properties."""
    form: str
    category: SyntacticCategory
    embedding: np.ndarray
    grounded_meaning: Optional[np.ndarray] = None  # Perceptual grounding
    frequency: float = 1.0


@dataclass
class Construction:
    """
    A form-meaning pair (Construction Grammar).

    Constructions are patterns like:
    - "X causes Y" -> causation schema
    - "X gives Y Z" -> transfer schema
    """
    name: str
    pattern: List[Union[SyntacticCategory, str]]  # Pattern to match
    meaning_schema: np.ndarray  # Semantic representation
    roles: List[SemanticRole]   # Roles in the construction
    examples: List[str] = field(default_factory=list)


@dataclass
class Proposition:
    """A structured meaning representation."""
    predicate: str
    arguments: Dict[SemanticRole, Any]
    embedding: np.ndarray
    truth_value: Optional[bool] = None
    modal: Optional[str] = None  # possible, necessary, etc.


@dataclass
class DiscourseEntity:
    """An entity tracked in discourse."""
    id: str
    description: str
    embedding: np.ndarray
    mentions: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    salient: bool = True


class Lexicon:
    """
    Mental lexicon with grounded word meanings.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.words: Dict[str, Word] = {}

        # Category embeddings (learned)
        self.category_embeddings = {
            cat: np.random.randn(dim) * 0.1 for cat in SyntacticCategory
        }

        # Initialize with basic words
        self._init_basic_lexicon()

    def _init_basic_lexicon(self):
        """Initialize basic vocabulary."""
        basics = [
            # Determiners
            ('the', SyntacticCategory.DETERMINER),
            ('a', SyntacticCategory.DETERMINER),
            ('this', SyntacticCategory.DETERMINER),
            ('that', SyntacticCategory.DETERMINER),
            # Pronouns
            ('I', SyntacticCategory.PRONOUN),
            ('you', SyntacticCategory.PRONOUN),
            ('he', SyntacticCategory.PRONOUN),
            ('she', SyntacticCategory.PRONOUN),
            ('it', SyntacticCategory.PRONOUN),
            ('they', SyntacticCategory.PRONOUN),
            ('we', SyntacticCategory.PRONOUN),
            # Common verbs
            ('is', SyntacticCategory.VERB),
            ('are', SyntacticCategory.VERB),
            ('have', SyntacticCategory.VERB),
            ('do', SyntacticCategory.VERB),
            ('go', SyntacticCategory.VERB),
            ('see', SyntacticCategory.VERB),
            ('give', SyntacticCategory.VERB),
            ('take', SyntacticCategory.VERB),
            ('make', SyntacticCategory.VERB),
            # Prepositions
            ('in', SyntacticCategory.PREPOSITION),
            ('on', SyntacticCategory.PREPOSITION),
            ('at', SyntacticCategory.PREPOSITION),
            ('to', SyntacticCategory.PREPOSITION),
            ('from', SyntacticCategory.PREPOSITION),
            ('with', SyntacticCategory.PREPOSITION),
            # Conjunctions
            ('and', SyntacticCategory.CONJUNCTION),
            ('or', SyntacticCategory.CONJUNCTION),
            ('but', SyntacticCategory.CONJUNCTION),
        ]

        for form, category in basics:
            embedding = self.category_embeddings[category] + np.random.randn(self.dim) * 0.05
            self.words[form.lower()] = Word(form, category, embedding)

    def lookup(self, word: str) -> Optional[Word]:
        """Look up word in lexicon."""
        return self.words.get(word.lower())

    def add_word(self,
                 form: str,
                 category: SyntacticCategory,
                 embedding: Optional[np.ndarray] = None,
                 grounded: Optional[np.ndarray] = None):
        """Add word to lexicon."""
        if embedding is None:
            embedding = self.category_embeddings[category] + np.random.randn(self.dim) * 0.05

        self.words[form.lower()] = Word(
            form=form,
            category=category,
            embedding=embedding,
            grounded_meaning=grounded
        )

    def ground_word(self, word: str, perceptual_embedding: np.ndarray):
        """Ground word in perceptual experience."""
        if word.lower() in self.words:
            self.words[word.lower()].grounded_meaning = perceptual_embedding.copy()


class ConstructionGrammar:
    """
    Construction Grammar implementation.

    Constructions are form-meaning pairs that can be:
    - Atomic (single words)
    - Complex (multi-word patterns)
    - Schematic (abstract patterns)
    """

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.constructions: List[Construction] = []

        # Initialize basic constructions
        self._init_basic_constructions()

    def _init_basic_constructions(self):
        """Initialize core grammatical constructions."""
        constructions = [
            # Transitive construction: SUBJ VERB OBJ
            Construction(
                name='transitive',
                pattern=[SyntacticCategory.NOUN, SyntacticCategory.VERB, SyntacticCategory.NOUN],
                meaning_schema=np.random.randn(self.dim) * 0.1,
                roles=[SemanticRole.AGENT, SemanticRole.PATIENT],
                examples=['cat chases mouse', 'dog bites man']
            ),
            # Ditransitive: SUBJ VERB OBJ1 OBJ2
            Construction(
                name='ditransitive',
                pattern=[SyntacticCategory.NOUN, SyntacticCategory.VERB,
                        SyntacticCategory.NOUN, SyntacticCategory.NOUN],
                meaning_schema=np.random.randn(self.dim) * 0.1,
                roles=[SemanticRole.AGENT, SemanticRole.THEME, SemanticRole.GOAL],
                examples=['he gave her flowers', 'I sent you message']
            ),
            # Caused-motion: SUBJ VERB OBJ PREP LOC
            Construction(
                name='caused_motion',
                pattern=[SyntacticCategory.NOUN, SyntacticCategory.VERB,
                        SyntacticCategory.NOUN, SyntacticCategory.PREPOSITION,
                        SyntacticCategory.NOUN],
                meaning_schema=np.random.randn(self.dim) * 0.1,
                roles=[SemanticRole.AGENT, SemanticRole.THEME, SemanticRole.GOAL],
                examples=['put book on table', 'throw ball to me']
            ),
            # Resultative: SUBJ VERB OBJ ADJ
            Construction(
                name='resultative',
                pattern=[SyntacticCategory.NOUN, SyntacticCategory.VERB,
                        SyntacticCategory.NOUN, SyntacticCategory.ADJECTIVE],
                meaning_schema=np.random.randn(self.dim) * 0.1,
                roles=[SemanticRole.AGENT, SemanticRole.PATIENT],
                examples=['paint wall red', 'hammer metal flat']
            ),
        ]

        self.constructions.extend(constructions)

    def match_construction(self, categories: List[SyntacticCategory]) -> List[Construction]:
        """Find constructions matching category sequence."""
        matches = []

        for construction in self.constructions:
            pattern_cats = [p for p in construction.pattern if isinstance(p, SyntacticCategory)]
            if len(pattern_cats) == len(categories):
                if all(p == c for p, c in zip(pattern_cats, categories)):
                    matches.append(construction)

        return matches

    def apply_construction(self,
                           construction: Construction,
                           words: List[Word]) -> Proposition:
        """Apply construction to extract meaning."""
        # Map words to roles
        arguments = {}
        for i, role in enumerate(construction.roles):
            if i < len(words):
                arguments[role] = words[i].form

        # Compute composed embedding
        word_embeddings = np.array([w.embedding for w in words])
        composed = construction.meaning_schema + np.mean(word_embeddings, axis=0)

        # Extract predicate (usually the verb)
        verb_words = [w for w in words if w.category == SyntacticCategory.VERB]
        predicate = verb_words[0].form if verb_words else 'unknown'

        return Proposition(
            predicate=predicate,
            arguments=arguments,
            embedding=composed
        )


class PragmaticsEngine:
    """
    Pragmatic interpretation (context-dependent meaning).

    Implements:
    - Speech act recognition
    - Implicature computation
    - Reference resolution
    - Common ground tracking
    """

    def __init__(self):
        # Common ground (shared knowledge)
        self.common_ground: Set[str] = set()

        # Speech act patterns
        self.speech_act_patterns = {
            SpeechAct.QUESTION: [r'\?$', r'^(what|who|where|when|why|how|is|are|do|does|can|will)'],
            SpeechAct.COMMAND: [r'^(do|make|go|come|stop|start|please)', r'!$'],
            SpeechAct.REQUEST: [r'^(could you|would you|can you|please)', r'\?$'],
            SpeechAct.ASSERT: [r'\.$'],
            SpeechAct.GREET: [r'^(hi|hello|hey|good morning|good evening)'],
            SpeechAct.THANK: [r'^(thank|thanks)'],
            SpeechAct.APOLOGIZE: [r'^(sorry|apologize|forgive)'],
        }

    def identify_speech_act(self, utterance: str) -> SpeechAct:
        """Identify the speech act of an utterance."""
        utterance_lower = utterance.lower().strip()

        # Check patterns
        for act, patterns in self.speech_act_patterns.items():
            for pattern in patterns:
                if re.search(pattern, utterance_lower):
                    return act

        # Default to assertion
        return SpeechAct.ASSERT

    def compute_implicature(self,
                            utterance: str,
                            literal_meaning: Proposition,
                            context: Dict[str, Any]) -> List[str]:
        """
        Compute conversational implicatures (Grice).

        What the speaker means beyond what they literally say.
        """
        implicatures = []

        # Quantity: Don't say more than needed
        # If speaker says "some", they implicate "not all"
        if 'some' in utterance.lower():
            implicatures.append('not_all')

        # Relevance: Utterances should be relevant
        # If response seems off-topic, there might be hidden connection
        if context.get('previous_topic') and context['previous_topic'] not in utterance.lower():
            implicatures.append('topic_shift_intended')

        # Manner: Be clear and orderly
        # Unusual word order might signal emphasis
        if literal_meaning.predicate:
            verb_position = utterance.lower().find(literal_meaning.predicate)
            if verb_position == 0:
                implicatures.append('emphasis_on_action')

        return implicatures

    def update_common_ground(self, proposition: str, accepted: bool = True):
        """Update shared common ground."""
        if accepted:
            self.common_ground.add(proposition)
        else:
            self.common_ground.discard(proposition)

    def is_in_common_ground(self, proposition: str) -> bool:
        """Check if proposition is in common ground."""
        return proposition in self.common_ground


class DiscourseModel:
    """
    Discourse representation and tracking.

    Tracks:
    - Entities mentioned
    - Discourse relations
    - Topic structure
    - Coherence
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Active entities in discourse
        self.entities: Dict[str, DiscourseEntity] = {}

        # Discourse history
        self.utterance_history: List[Tuple[str, Proposition]] = []

        # Current topic
        self.current_topic: Optional[np.ndarray] = None

        # Salience decay
        self.salience_decay = 0.9

    def introduce_entity(self,
                         id: str,
                         description: str,
                         embedding: np.ndarray,
                         properties: Optional[Dict[str, Any]] = None):
        """Introduce new entity to discourse."""
        self.entities[id] = DiscourseEntity(
            id=id,
            description=description,
            embedding=embedding.copy(),
            mentions=[description],
            properties=properties or {},
            salient=True
        )

    def mention_entity(self, id: str, mention: str):
        """Record mention of entity."""
        if id in self.entities:
            self.entities[id].mentions.append(mention)
            self.entities[id].salient = True

    def resolve_reference(self, mention: str, mention_embedding: np.ndarray) -> Optional[DiscourseEntity]:
        """
        Resolve a referring expression to a discourse entity.

        Uses:
        - Recency (more recent = more likely)
        - Salience
        - Semantic fit
        """
        if not self.entities:
            return None

        best_entity = None
        best_score = -float('inf')

        for entity in self.entities.values():
            # Recency (via salience)
            recency_score = 1.0 if entity.salient else 0.5

            # Semantic similarity
            semantic_score = np.dot(entity.embedding, mention_embedding) / (
                np.linalg.norm(entity.embedding) * np.linalg.norm(mention_embedding) + 1e-8
            )

            # String match
            string_score = 0.0
            if mention.lower() in entity.description.lower():
                string_score = 1.0
            elif any(mention.lower() in m.lower() for m in entity.mentions):
                string_score = 0.8

            # Combined score
            score = 0.3 * recency_score + 0.4 * semantic_score + 0.3 * string_score

            if score > best_score:
                best_score = score
                best_entity = entity

        if best_score > 0.3:
            return best_entity
        return None

    def add_utterance(self, utterance: str, meaning: Proposition):
        """Add utterance to discourse history."""
        self.utterance_history.append((utterance, meaning))

        # Update topic
        if self.current_topic is None:
            self.current_topic = meaning.embedding.copy()
        else:
            # Gradual topic shift
            self.current_topic = 0.8 * self.current_topic + 0.2 * meaning.embedding

        # Decay salience of old entities
        for entity in self.entities.values():
            entity.salient = False

        # Mark mentioned entities as salient
        for role, arg in meaning.arguments.items():
            for entity in self.entities.values():
                if arg in entity.mentions or arg == entity.description:
                    entity.salient = True

    def get_context(self) -> Dict[str, Any]:
        """Get current discourse context."""
        return {
            'entities': {e.id: e.description for e in self.entities.values()},
            'salient_entities': [e.id for e in self.entities.values() if e.salient],
            'topic': self.current_topic.tolist() if self.current_topic is not None else None,
            'history_length': len(self.utterance_history)
        }


class LanguageGenerator:
    """
    Language production system.

    Generates utterances from meanings.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Templates for generation
        self.templates = {
            'transitive': '{agent} {verb} {patient}',
            'ditransitive': '{agent} {verb} {recipient} {theme}',
            'location': '{theme} is {prep} {location}',
            'property': '{entity} is {property}',
            'question_what': 'What {verb} {patient}?',
            'question_where': 'Where is {entity}?',
        }

    def generate(self,
                 proposition: Proposition,
                 speech_act: SpeechAct = SpeechAct.ASSERT) -> str:
        """Generate utterance from proposition."""
        # Select template based on roles
        roles = set(proposition.arguments.keys())

        if SemanticRole.AGENT in roles and SemanticRole.PATIENT in roles:
            if SemanticRole.GOAL in roles:
                template = self.templates['ditransitive']
            else:
                template = self.templates['transitive']
        elif SemanticRole.LOCATION in roles:
            template = self.templates['location']
        else:
            template = self.templates['property']

        # Fill template
        fill_dict = {
            'verb': proposition.predicate,
            **{role.name.lower(): arg for role, arg in proposition.arguments.items()}
        }

        try:
            utterance = template.format(**fill_dict)
        except KeyError:
            # Fallback to simple generation
            args = ' '.join(str(v) for v in proposition.arguments.values())
            utterance = f"{proposition.predicate} {args}"

        # Adjust for speech act
        if speech_act == SpeechAct.QUESTION:
            if not utterance.endswith('?'):
                utterance = 'Is ' + utterance + '?'
        elif speech_act == SpeechAct.COMMAND:
            utterance = utterance.capitalize() + '!'

        return utterance.capitalize()


class LanguageSystem:
    """
    Complete language processing system.

    Integrates lexicon, grammar, pragmatics, discourse, and generation.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

        # Components
        self.lexicon = Lexicon(dim)
        self.grammar = ConstructionGrammar(dim)
        self.pragmatics = PragmaticsEngine()
        self.discourse = DiscourseModel(dim)
        self.generator = LanguageGenerator(dim)

    def understand(self, utterance: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Full language understanding pipeline.
        """
        context = context or {}

        # 1. Tokenize (simple)
        tokens = utterance.lower().split()

        # 2. Lexical lookup
        words = []
        unknown = []
        for token in tokens:
            word = self.lexicon.lookup(token)
            if word:
                words.append(word)
            else:
                # Unknown word - guess category
                unknown.append(token)
                # Default to noun for unknowns
                self.lexicon.add_word(token, SyntacticCategory.NOUN)
                words.append(self.lexicon.lookup(token))

        # 3. Syntactic analysis (get categories)
        categories = [w.category for w in words if w]

        # 4. Construction matching
        constructions = self.grammar.match_construction(categories)

        # 5. Semantic interpretation
        if constructions:
            proposition = self.grammar.apply_construction(constructions[0], words)
        else:
            # Fallback: simple embedding composition
            embeddings = [w.embedding for w in words if w]
            composed = np.mean(embeddings, axis=0) if embeddings else np.zeros(self.dim)
            proposition = Proposition(
                predicate='unknown',
                arguments={},
                embedding=composed
            )

        # 6. Speech act identification
        speech_act = self.pragmatics.identify_speech_act(utterance)

        # 7. Reference resolution
        for role, arg in list(proposition.arguments.items()):
            if isinstance(arg, str):
                arg_embedding = self.lexicon.lookup(arg)
                if arg_embedding:
                    resolved = self.discourse.resolve_reference(arg, arg_embedding.embedding)
                    if resolved:
                        proposition.arguments[role] = resolved.id

        # 8. Pragmatic inference
        implicatures = self.pragmatics.compute_implicature(utterance, proposition, context)

        # 9. Update discourse
        self.discourse.add_utterance(utterance, proposition)

        return {
            'utterance': utterance,
            'tokens': tokens,
            'unknown_words': unknown,
            'construction': constructions[0].name if constructions else None,
            'proposition': {
                'predicate': proposition.predicate,
                'arguments': {r.name: a for r, a in proposition.arguments.items()},
                'embedding': proposition.embedding.tolist()
            },
            'speech_act': speech_act.name,
            'implicatures': implicatures,
            'context': self.discourse.get_context()
        }

    def generate(self,
                 predicate: str,
                 arguments: Dict[str, str],
                 speech_act: SpeechAct = SpeechAct.ASSERT) -> str:
        """Generate utterance from meaning."""
        # Convert string keys to SemanticRole
        role_args = {}
        for key, value in arguments.items():
            try:
                role = SemanticRole[key.upper()]
                role_args[role] = value
            except KeyError:
                pass

        # Create proposition
        embedding = np.zeros(self.dim)
        for role, arg in role_args.items():
            word = self.lexicon.lookup(arg)
            if word:
                embedding += word.embedding

        proposition = Proposition(
            predicate=predicate,
            arguments=role_args,
            embedding=embedding
        )

        return self.generator.generate(proposition, speech_act)

    def ground_language(self, word: str, perceptual_embedding: np.ndarray):
        """Ground word meaning in perception."""
        self.lexicon.ground_word(word, perceptual_embedding)

    def learn_word(self,
                   word: str,
                   category: SyntacticCategory,
                   context_embedding: Optional[np.ndarray] = None):
        """Learn new word from context."""
        self.lexicon.add_word(word, category, context_embedding)

    def get_state(self) -> Dict[str, Any]:
        """Get language system state."""
        return {
            'vocabulary_size': len(self.lexicon.words),
            'constructions': len(self.grammar.constructions),
            'discourse_entities': len(self.discourse.entities),
            'common_ground_size': len(self.pragmatics.common_ground)
        }
