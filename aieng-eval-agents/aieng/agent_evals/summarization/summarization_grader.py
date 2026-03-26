"""Summarization grader for evaluating summarization agent responses.

This module provides LLM-as-a-judge evaluation methodology for summarization
tasks, assessing summary quality across multiple dimensions including accuracy,
completeness, conciseness, and readability.
"""

import logging
from enum import Enum
from typing import Any

from aieng.agent_evals.async_client_manager import AsyncClientManager
from aieng.agent_evals.evaluation.graders._utils import run_structured_parse_call
from aieng.agent_evals.evaluation.graders.config import LLMRequestConfig
from aieng.agent_evals.evaluation.types import Evaluation
from pydantic import BaseModel, Field


logger = logging.getLogger(__name__)


class SummaryQuality(str, Enum):
    """Possible overall quality ratings for summary evaluation."""
    
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class SummarizationResult(BaseModel):
    """Result from summarization evaluation with multiple quality dimensions.
    
    This provides a comprehensive assessment of summary quality across
    key dimensions important for financial news summarization.
    """
    
    accuracy: float = Field(
        default=0.0, 
        description="How factually correct the summary is (0-1). Checks for hallucinations, misrepresentations, and verbatim copying.",
        ge=0.0, 
        le=1.0
    )
    completeness: float = Field(
        default=0.0, 
        description="How well the summary captures ALL main points (0-1). Assesses comprehensive coverage of critical information.",
        ge=0.0, 
        le=1.0
    )
    conciseness: float = Field(
        default=0.0, 
        description="How well the summary avoids unnecessary detail (0-1). Evaluates brevity and focus.",
        ge=0.0, 
        le=1.0
    )
    clarity: float = Field(
        default=0.0, 
        description="How clear and readable the summary is (0-1). Assesses language quality and coherence.",
        ge=0.0, 
        le=1.0
    )
    overall_quality: SummaryQuality = Field(
        default=SummaryQuality.POOR,
        description="Overall quality assessment of the summary"
    )
    explanation: str = Field(
        default="", 
        description="Detailed explanation of the evaluation reasoning"
    )
    
    @property
    def average_score(self) -> float:
        """Calculate the average of all numeric scores."""
        return (self.accuracy + self.completeness + self.conciseness + self.clarity) / 4.0
    
    def to_evaluations(self) -> list[Evaluation]:
        """Convert this result to Langfuse Evaluation objects.
        
        Returns
        -------
        list[Evaluation]
            Six evaluations: Accuracy, Completeness, Conciseness, Clarity, 
            Average Score, and Overall Quality.
        """
        comment_parts = [
            f"Overall Quality: {self.overall_quality.value.title()}",
            f"Accuracy: {self.accuracy:.2f}",
            f"Completeness: {self.completeness:.2f}",
            f"Conciseness: {self.conciseness:.2f}",
            f"Clarity: {self.clarity:.2f}",
            f"Average Score: {self.average_score:.2f}",
        ]
        
        if self.explanation:
            comment_parts.append(f"\nExplanation: {self.explanation}")
        
        comment = "\n".join(comment_parts)
        
        return [
            Evaluation(
                name="Accuracy",
                value=self.accuracy,
                comment=f"Factual correctness & original phrasing: {self.accuracy:.2f}"
            ),
            Evaluation(
                name="Completeness", 
                value=self.completeness,
                comment=f"Coverage of ALL main points: {self.completeness:.2f}"
            ),
            Evaluation(
                name="Conciseness",
                value=self.conciseness, 
                comment=f"Brevity and focus: {self.conciseness:.2f}"
            ),
            Evaluation(
                name="Clarity",
                value=self.clarity,
                comment=f"Readability and coherence: {self.clarity:.2f}"
            ),
            Evaluation(
                name="Average Score",
                value=self.average_score,
                comment=comment
            ),
            Evaluation(
                name="Overall Quality",
                value=self.overall_quality.value.title(),
                comment=self.explanation
            ),
        ]
    
    @staticmethod
    def error_evaluations(error_msg: str) -> list[Evaluation]:
        """Create error evaluations when evaluation fails.
        
        Parameters
        ----------
        error_msg : str
            Description of the error that occurred.
            
        Returns
        -------
        list[Evaluation]
            Six evaluations matching the success path with error values.
        """
        comment = f"Evaluation error: {error_msg}"
        return [
            Evaluation(name="Accuracy", value=0.0, comment=comment),
            Evaluation(name="Completeness", value=0.0, comment=comment),
            Evaluation(name="Conciseness", value=0.0, comment=comment),
            Evaluation(name="Clarity", value=0.0, comment=comment),
            Evaluation(name="Average Score", value=0.0, comment=comment),
            Evaluation(name="Overall Quality", value="Poor", comment=comment),
        ]


class SummarizationGraderResponse(BaseModel):
    """Structured response from the summarization grader."""
    
    summary_evaluation: dict[str, Any] = Field(
        alias="Summary Evaluation",
        description=(
            "Dictionary containing: "
            "Accuracy (float 0-1) - Factual correctness & original phrasing (no verbatim copying); "
            "Completeness (float 0-1) - Coverage of ALL main points systematically verified; "
            "Conciseness (float 0-1) - Brevity and focus; "
            "Clarity (float 0-1) - Readability and coherence; "
            "Overall Quality (str) - excellent/good/fair/poor; "
            "Explanation (str) - Detailed reasoning"
        ),
    )


# Summarization grader prompt designed for financial news
SUMMARIZATION_GRADER_PROMPT = """\
Evaluate this AI-generated financial news summary across four dimensions (0-1 scale):

**Summary Evaluation Task**
* **Purpose:** Assess whether the AI summary meets the specific requirements for financial news summarization: 2-4 sentences, captures ALL main points, includes key entities, mentions financial figures, uses original phrasing (no verbatim copying), and stays grounded in the source material.
* **Process:**
  * Read the original article (title + body) carefully to identify ALL main points, events, announcements, key companies/people involved, and significant financial figures.
  * Analyze the AI-generated summary against the original article and the specific summarization requirements.
  * Check specifically for verbatim copying of sentences or phrases from the original text.
  * Evaluate the summary across four key dimensions:
    * **Accuracy (0-1)**: Are all facts in the summary correct and based solely on the article content? Are there any hallucinations, misrepresentations, or added outside information? Does the summary avoid verbatim copying of sentences from the original text?
    * **Completeness (0-1)**: Does the summary capture ALL main points from the article including the primary event/announcement, ALL key companies or people involved, and ALL significant financial figures when present? Systematically verify each major point is addressed.
    * **Conciseness (0-1)**: Is the summary 2-4 sentences long? Does it avoid unnecessary details while including essential information? Is it appropriately brief for financial news?
    * **Clarity (0-1)**: Is the summary well-written, coherent, and easy to understand? Is the language clear and professional without headings, labels, or preamble?
  * Assign an overall quality rating: "excellent", "good", "fair", or "poor"
* **Explanation:** Provide a detailed explanation of your assessment, referencing how well the summary meets the specific requirements and how it relates to the original article.

**Scoring Guidelines:**
* **1.0**: Excellent - fully meets requirements
* **0.8-0.9**: Good - meets requirements with minor issues
* **0.6-0.7**: Fair - partially meets requirements, some problems
* **0.4-0.5**: Poor - significant issues, major improvements needed
* **0.0-0.3**: Very poor - fails to meet basic requirements

**Overall Quality Ratings:**
* "excellent": All dimensions 0.8+
* "good": Most dimensions 0.6+, minor improvements needed
* "fair": Mixed performance, significant improvements needed
* "poor": Multiple dimensions below 0.5, major revision required

**Output Format:**
Return a JSON dictionary with "Summary Evaluation" containing: Accuracy (float), Completeness (float), Conciseness (float), Clarity (float), Overall Quality (string), and Explanation (string with detailed reasoning).

Original Article:
Title: {title}

Body: {body}

--------------------
AI-Generated Summary to Evaluate:
{summary}

--------------------
Rating:"""


def _calculate_result_from_grader(
    grader_result: dict[str, Any],
) -> SummarizationResult:
    """Extract and validate scores from grader output.
    
    Parameters
    ----------
    grader_result : dict
        Output from the LLM grader with evaluation scores.
        
    Returns
    -------
    SummarizationResult
        Validated evaluation result with all scores and quality assessment.
    """
    accuracy = float(grader_result.get("Accuracy", 0.0))
    completeness = float(grader_result.get("Completeness", 0.0))
    conciseness = float(grader_result.get("Conciseness", 0.0))
    clarity = float(grader_result.get("Clarity", 0.0))
    explanation = str(grader_result.get("Explanation", ""))
    
    # Validate and convert overall quality
    quality_str = grader_result.get("Overall Quality", "poor").lower()
    try:
        overall_quality = SummaryQuality(quality_str)
    except ValueError:
        logger.warning(f"Invalid quality rating '{quality_str}', defaulting to 'poor'")
        overall_quality = SummaryQuality.POOR
    
    # Clamp scores to valid range
    accuracy = max(0.0, min(1.0, accuracy))
    completeness = max(0.0, min(1.0, completeness))
    conciseness = max(0.0, min(1.0, conciseness))
    clarity = max(0.0, min(1.0, clarity))
    
    return SummarizationResult(
        accuracy=accuracy,
        completeness=completeness,
        conciseness=conciseness,
        clarity=clarity,
        overall_quality=overall_quality,
        explanation=explanation,
    )


async def evaluate_summarization_async(
    *,
    title: str,
    body: str,
    summary: str,
    model_config: LLMRequestConfig | None = None,
) -> SummarizationResult:
    """Evaluate a summary using LLM-as-a-judge methodology.
    
    Parameters
    ----------
    title : str
        The original article title.
    body : str
        The original article body text.
    summary : str
        The AI-generated summary to evaluate.
    model_config : LLMRequestConfig | None, optional
        Optional model configuration. If None, defaults are used.
        
    Returns
    -------
    SummarizationResult
        The evaluation result with scores across multiple dimensions.
    """
    config = model_config or LLMRequestConfig()
    client_manager = AsyncClientManager.get_instance()
    
    # Build the grader prompt
    user_prompt = SUMMARIZATION_GRADER_PROMPT.format(
        title=title,
        body=body,
        summary=summary,
    )
    
    try:
        completion = await run_structured_parse_call(
            openai_client=client_manager.openai_client,
            default_model=client_manager.configs.default_evaluator_model,
            system_prompt="",  # All instructions in user prompt
            user_prompt=user_prompt,
            response_format=SummarizationGraderResponse,
            model_config=config,
        )
        
        grader_response: SummarizationGraderResponse | None = completion.choices[0].message.parsed
        
        if grader_response is None:
            raise ValueError("Grader returned null response")
            
        return _calculate_result_from_grader(grader_response.summary_evaluation)
        
    except Exception as e:
        logger.warning(f"Failed to evaluate with summarization grader: {e}")
        return SummarizationResult(
            accuracy=0.0,
            completeness=0.0,
            conciseness=0.0,
            clarity=0.0,
            overall_quality=SummaryQuality.POOR,
            explanation=f"Grader error: {e}",
        )