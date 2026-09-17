from typing import List, Dict, Any
from pydantic import BaseModel, Field


class Concern(BaseModel):
    name: str
    intensity: float = 0.5
    distress: float = 0.5
    sensitivity: float = 0.5
    recurrence: float = Field(default=0.5, description="1.0 for recurring, 0.5 for new")
    relevance: float = Field(default=1.0, description="Relevance to conversation")
    risk: float = 0.0


class RankedConcern(BaseModel):
    concern: Concern
    priority_score: float


class RankingAlgorithm:
    """
    Ranks multiple concerns found in user interaction.
    Formula: Priority = Intensity * Distress * Sensitivity * Recurrence * Relevance * RiskMultiplier
    """
    
    @staticmethod
    def calculate_priority(concern: Concern) -> float:
        risk_multiplier = max(1.0, concern.risk * 5)
        score = (
            concern.intensity * 
            concern.distress * 
            concern.sensitivity * 
            concern.recurrence * 
            concern.relevance * 
            risk_multiplier
        )
        return round(score, 3)

    @staticmethod
    def rank_concerns(concerns: List[Concern]) -> Dict[str, Any]:
        if not concerns:
            return {}
            
        ranked_list = []
        for c in concerns:
            score = RankingAlgorithm.calculate_priority(c)
            ranked_list.append(RankedConcern(concern=c, priority_score=score))
            
        ranked_list.sort(key=lambda x: x.priority_score, reverse=True)
        
        primary = ranked_list[0].concern.name
        secondary = [r.concern.name for r in ranked_list[1:3]] if len(ranked_list) > 1 else []
        associated = [r.concern.name for r in ranked_list[3:]] if len(ranked_list) > 3 else []
        
        return {
            "primary_concern": primary,
            "secondary_contributors": secondary,
            "associated_effects": associated,
            "top_score": ranked_list[0].priority_score
        }
