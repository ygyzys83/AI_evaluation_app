from pydantic import BaseModel, Field
from typing import Literal

class JudgeVerdict(BaseModel):
    """
    Structured output schema for the LLM-as-judge grading step.
    Replaces fragile string parsing with validated, typed output.
    """
    verdict: Literal["PASS", "FAIL"] = Field(
        description="PASS if the AI answer is factually correct, FAIL otherwise."
    )
    similarity_score: int = Field(
        ge=1, le=5,
        description="Semantic similarity of the AI answer to ground truth, 1-5."
    )
    reasoning: str = Field(
        description="One sentence explaining the verdict."
    )