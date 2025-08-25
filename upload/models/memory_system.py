"""
Human-like Memory System for InternImage-XL + UperNet
Enhanced memory system with episodic, semantic, and working memory components
"""
import torch
import torch.nn.functional as F
import numpy as np
import logging
from collections import deque

logger = logging.getLogger(__name__)

class HumanLikeMemoryBank:
    """Enhanced memory system with human-like characteristics"""
    
    def __init__(self, max_size=1000, feature_dim=128):
        self.max_size = max_size
        self.feature_dim = feature_dim
        
        # Multiple memory types (like human memory systems)
        self.episodic_memory = []      # Specific experiences
        self.semantic_memory = {}      # Generalized knowledge
        self.working_memory = deque(maxlen=10)  # Recent context
        
        # Memory attributes
        self.patterns = []             # Feature patterns
        self.contexts = []             # Contextual information
        self.emotions = []             # Success/failure "emotions"
        self.timestamps = []           # When stored
        self.access_counts = []        # How often accessed
        self.importance_scores = []    # Importance weights
        
        # Memory dynamics
        self.global_time = 0
        self.forgetting_rate = 0.995   # Gradual forgetting
        
    def add_experience(self, pattern, context, success_score, metadata=None):
        """Add new experience with rich context"""
        self.global_time += 1
        
        # Store episodic memory
        episode = {
            'pattern': pattern.detach().cpu(),
            'context': context,
            'emotion': self._compute_emotion(success_score),
            'timestamp': self.global_time,
            'metadata': metadata or {}
        }
        
        # Add to working memory first
        self.working_memory.append(episode)
        
        # Consolidate to long-term memory
        if self._should_consolidate(episode):
            self._consolidate_memory(episode)
            
        # Update semantic memory (generalized patterns)
        self._update_semantic_memory(pattern, success_score)
        
        # Apply forgetting
        self._apply_forgetting()
    
    def recall(self, query_pattern, query_context=None, top_k=5, recall_type='episodic'):
        """Human-like memory recall with context and associations"""
        if recall_type == 'episodic':
            return self._episodic_recall(query_pattern, query_context, top_k)
        elif recall_type == 'semantic':
            return self._semantic_recall(query_pattern, top_k)
        else:  # associative
            return self._associative_recall(query_pattern, query_context, top_k)
    
    def _compute_emotion(self, success_score):
        """Compute emotional valence (positive/negative feeling)"""
        if success_score > 0.8:
            return 'very_positive'
        elif success_score > 0.6:
            return 'positive'
        elif success_score > 0.4:
            return 'neutral'
        elif success_score > 0.2:
            return 'negative'
        else:
            return 'very_negative'
    
    def _should_consolidate(self, episode):
        """Decide if episode should be moved to long-term memory"""
        # Consolidate based on emotional significance and novelty
        emotion_weights = {
            'very_positive': 0.9, 'positive': 0.7, 'neutral': 0.3,
            'negative': 0.6, 'very_negative': 0.8  # We learn from failures too!
        }
        
        emotion_score = emotion_weights.get(episode['emotion'], 0.5)
        novelty_score = self._compute_novelty(episode['pattern'])
        
        return (emotion_score + novelty_score) / 2 > 0.6
    
    def _consolidate_memory(self, episode):
        """Move episode to long-term memory with importance weighting"""
        if len(self.patterns) >= self.max_size:
            # Human-like forgetting: remove least important, not oldest
            self._forget_least_important()
        
        self.patterns.append(episode['pattern'])
        self.contexts.append(episode['context'])
        self.emotions.append(episode['emotion'])
        self.timestamps.append(episode['timestamp'])
        self.access_counts.append(0)
        
        # Compute importance based on emotion and recency
        importance = self._compute_importance(episode)
        self.importance_scores.append(importance)
    
    def _episodic_recall(self, query_pattern, query_context, top_k):
        """Recall specific episodes similar to current situation"""
        if not self.patterns:
            return []
        
        # Ensure query_pattern is on CPU for comparison with stored patterns
        query_pattern_cpu = query_pattern.detach().cpu() if query_pattern.is_cuda else query_pattern
        
        similarities = []
        for i, pattern in enumerate(self.patterns):
            # Pattern similarity - both tensors now on CPU
            pattern_sim = F.cosine_similarity(
                query_pattern_cpu.flatten(), 
                pattern.flatten(), 
                dim=0
            ).item()
            
            # Context similarity (if available)
            context_sim = 0.0
            if query_context and self.contexts[i]:
                context_sim = self._context_similarity(query_context, self.contexts[i])
            
            # Recency and importance boost
            recency_boost = self._recency_boost(self.timestamps[i])
            importance_boost = self.importance_scores[i]
            
            # Combined similarity with human-like weighting
            total_sim = (0.4 * pattern_sim + 
                        0.2 * context_sim + 
                        0.2 * recency_boost + 
                        0.2 * importance_boost)
            
            similarities.append((total_sim, i))
        
        # Sort by similarity and return top-k
        similarities.sort(reverse=True)
        
        recalled_memories = []
        for sim_score, idx in similarities[:top_k]:
            # Update access count (strengthens memory)
            self.access_counts[idx] += 1
            self.importance_scores[idx] *= 1.05  # Boost importance
            
            recalled_memories.append({
                'pattern': self.patterns[idx],
                'context': self.contexts[idx],
                'emotion': self.emotions[idx],
                'similarity': sim_score,
                'confidence': self._compute_recall_confidence(sim_score)
            })
        
        return recalled_memories
    
    def _semantic_recall(self, query_pattern, top_k):
        """Recall generalized semantic knowledge"""
        if not self.patterns:
            return []
        
        # Find patterns with positive emotions and high access counts
        candidates = []
        for i, (pattern, emotion, access_count) in enumerate(
            zip(self.patterns, self.emotions, self.access_counts)
        ):
            if emotion in ['positive', 'very_positive']:
                score = access_count * self.importance_scores[i]
                candidates.append((score, i))
        
        candidates.sort(reverse=True)
        return [{'pattern': self.patterns[i], 'emotion': self.emotions[i]} for _, i in candidates[:top_k]]
    
    def _associative_recall(self, query_pattern, query_context, top_k):
        """Recall through associations and chains of memory"""
        # Start with episodic recall
        initial_memories = self._episodic_recall(query_pattern, query_context, top_k//2)
        
        # Find associated memories through context links
        associated_memories = []
        for memory in initial_memories:
            # Find memories with similar contexts
            for i, context in enumerate(self.contexts):
                if (context and memory['context'] and 
                    self._context_similarity(memory['context'], context) > 0.7):
                    associated_memories.append({
                        'pattern': self.patterns[i],
                        'context': self.contexts[i],
                        'emotion': self.emotions[i],
                        'association_type': 'contextual'
                    })
        
        return initial_memories + associated_memories[:top_k//2]
    
    def _compute_novelty(self, new_pattern):
        """Compute how novel this pattern is"""
        if not self.patterns:
            return 1.0
        
        # Ensure new_pattern is on CPU for comparison
        new_pattern_cpu = new_pattern.detach().cpu() if new_pattern.is_cuda else new_pattern
        
        max_similarity = 0.0
        for pattern in self.patterns[-10:]:  # Check against recent patterns
            similarity = F.cosine_similarity(
                new_pattern_cpu.flatten(), 
                pattern.flatten(), 
                dim=0
            ).item()
            max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity
    
    def _compute_importance(self, episode):
        """Compute importance score for memory consolidation"""
        emotion_weights = {
            'very_positive': 1.0, 'positive': 0.8, 'neutral': 0.3,
            'negative': 0.7, 'very_negative': 0.9
        }
        
        emotion_score = emotion_weights.get(episode['emotion'], 0.5)
        recency_score = 1.0  # New memories start with high importance
        novelty_score = self._compute_novelty(episode['pattern'])
        
        return (emotion_score + recency_score + novelty_score) / 3
    
    def _recency_boost(self, timestamp):
        """Boost score based on recency (recent memories more accessible)"""
        age = self.global_time - timestamp
        return np.exp(-age / 100.0)  # Exponential decay
    
    def _context_similarity(self, ctx1, ctx2):
        """Compute similarity between contexts"""
        if not isinstance(ctx1, dict) or not isinstance(ctx2, dict):
            return 0.0
        
        # Compare numeric context features
        similarity = 0.0
        count = 0
        
        for key in ctx1.keys():
            if key in ctx2 and isinstance(ctx1[key], (int, float)) and isinstance(ctx2[key], (int, float)):
                # Normalize difference to [0, 1] similarity
                diff = abs(ctx1[key] - ctx2[key])
                max_val = max(abs(ctx1[key]), abs(ctx2[key]), 1e-6)
                similarity += 1.0 - min(diff / max_val, 1.0)
                count += 1
        
        return similarity / count if count > 0 else 0.0
    
    def _compute_recall_confidence(self, similarity_score):
        """Compute confidence in recalled memory"""
        # Human memory confidence is not always accurate!
        base_confidence = similarity_score
        
        # Add some realistic uncertainty
        noise = np.random.normal(0, 0.1)
        confidence = np.clip(base_confidence + noise, 0.0, 1.0)
        
        return confidence
    
    def _apply_forgetting(self):
        """Apply gradual forgetting to all memories"""
        for i in range(len(self.importance_scores)):
            # Memories with higher access counts decay slower
            access_factor = 1.0 + 0.1 * self.access_counts[i]
            decay_rate = self.forgetting_rate * access_factor
            self.importance_scores[i] *= decay_rate
    
    def _forget_least_important(self):
        """Remove the least important memory"""
        if not self.importance_scores:
            return
        
        min_idx = np.argmin(self.importance_scores)
        
        # Remove from all lists
        del self.patterns[min_idx]
        del self.contexts[min_idx]
        del self.emotions[min_idx]
        del self.timestamps[min_idx]
        del self.access_counts[min_idx]
        del self.importance_scores[min_idx]
    
    def _update_semantic_memory(self, pattern, success_score):
        """Update semantic memory with generalized patterns"""
        emotion = self._compute_emotion(success_score)
        if emotion not in self.semantic_memory:
            self.semantic_memory[emotion] = []
        
        # Store compressed representation for semantic memory
        if len(self.semantic_memory[emotion]) < 50:  # Limit semantic memories
            self.semantic_memory[emotion].append(pattern.detach().cpu())
    
    def consolidate_during_sleep(self):
        """Simulate memory consolidation during 'sleep' (between epochs)"""
        logger.info(f"🧠 Memory consolidation: {len(self.patterns)} long-term memories")
        
        # Strengthen important memories
        for i in range(len(self.importance_scores)):
            if self.emotions[i] in ['very_positive', 'very_negative']:
                self.importance_scores[i] *= 1.1
        
        # Log memory statistics
        emotion_counts = {}
        for emotion in self.emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        logger.info(f"🧠 Memory emotions: {emotion_counts}")
        logger.info(f"🧠 Working memory: {len(self.working_memory)} recent experiences")
