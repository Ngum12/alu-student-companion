
"""
Nyptho: A Meta-Learning AI System
---------------------------------
Nyptho learns from other AI models to improve performance over time.
It consists of three core components:

1. NypthoCore: The central engine for generating responses
2. MetaLearningEngine: Observes and learns from other AI models
3. KnowledgeDistiller: Extracts and organizes knowledge from observations
"""

from .nyptho_core import NypthoCore
from .meta_learning_engine import MetaLearningEngine
from .knowledge_distillation import KnowledgeDistiller

# Configuration constants
NYPTHO_VERSION = "1.0.0"
DEFAULT_TRAITS = {
    "helpfulness": 0.85,
    "creativity": 0.7,
    "precision": 0.9,
    "friendliness": 0.8
}

# Public API
__all__ = [
    "NypthoCore", 
    "MetaLearningEngine", 
    "KnowledgeDistiller",
    "NYPTHO_VERSION",
    "DEFAULT_TRAITS",
    "create_nyptho_system"
]

def create_nyptho_system(custom_traits=None):
    """
    Factory function to create a fully initialized Nyptho system with all components
    connected and ready to use.
    
    Args:
        custom_traits (dict, optional): Custom personality traits to initialize Nyptho with
        
    Returns:
        tuple: (NypthoCore, MetaLearningEngine, KnowledgeDistiller)
    """
    # Initialize with default or custom traits
    traits = custom_traits or DEFAULT_TRAITS
    
    # Create and connect components
    core = NypthoCore()
    core.set_personality(traits)
    
    meta_learning = MetaLearningEngine(core)
    knowledge = KnowledgeDistiller()
    
    return core, meta_learning, knowledge

# Import important utility functions from submodules to make them easily accessible
try:
    from .nyptho_core import format_response, evaluate_response_quality
    __all__.extend(["format_response", "evaluate_response_quality"])
except ImportError:
    # These might not exist in the core module, which is fine
    pass

