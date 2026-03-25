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
        description="How factually correct the summary is (0-1). Checks for hallucinations and misrepresentations.",
        ge=0.0, 
        le=1.0
    )
    completeness: float = Field(
        default=0.0, 
        description="How well the summary captures key information (0-1). Assesses coverage of important points.",
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
                comment=f"Factual correctness: {self.accuracy:.2f}"
            ),
            Evaluation(
                name="Completeness", 
                value=self.completeness,
                comment=f"Coverage of key information: {self.completeness:.2f}"
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
            "Accuracy (float 0-1) - Factual correctness; "
            "Completeness (float 0-1) - Coverage of key information; "
            "Conciseness (float 0-1) - Brevity and focus; "
            "Clarity (float 0-1) - Readability and coherence; "
            "Overall Quality (str) - excellent/good/fair/poor; "
            "Explanation (str) - Detailed reasoning"
        ),
    )


# Summarization grader prompt designed for financial news
SUMMARIZATION_GRADER_PROMPT = """\
Your task is to evaluate the quality of an AI-generated summary of a financial news article.

**Summary Evaluation Task**
* **Purpose:** Assess whether the AI summary meets the specific requirements for financial news summarization: 2-4 sentences, captures main events, includes key entities, mentions financial figures, and stays grounded in the source material.
* **Process:**
  * Read the original article (title + body) carefully to identify the main event/announcement, key companies/people involved, and significant financial figures.
  * Analyze the AI-generated summary against the original article and the specific summarization requirements.
  * Evaluate the summary across four key dimensions:
    * **Accuracy (0-1)**: Are all facts in the summary correct and based solely on the article content? Are there any hallucinations, misrepresentations, or added outside information?
    * **Completeness (0-1)**: Does the summary capture the main event/announcement, key companies or people involved, and significant financial figures when present? Are critical elements missing?
    * **Conciseness (0-1)**: Is the summary 2-4 sentences long? Does it avoid unnecessary details while including essential information? Is it appropriately brief for financial news?
    * **Clarity (0-1)**: Is the summary well-written, coherent, and easy to understand? Is the language clear and professional without headings, labels, or preamble?
  * Assign an overall quality rating: "excellent", "good", "fair", or "poor"
* **Explanation:** Provide a detailed explanation of your assessment, referencing how well the summary meets the specific requirements and how it relates to the original article.

**Scoring Guidelines:**

**Accuracy (Factual Correctness & Groundedness):**
* **1.0**: All facts are correct and based solely on article content, no hallucinations or outside information
* **0.8-0.9**: Facts are correct with minor interpretation issues, well-grounded in source
* **0.6-0.7**: Mostly correct facts but some minor inaccuracies or slight overreach beyond article
* **0.4-0.5**: Several factual errors or some information not found in the original article
* **0.2-0.3**: Major factual errors or significant hallucinated content
* **0.0-0.1**: Severely inaccurate or completely fabricated information

**Completeness (Coverage of Required Elements):**
* **1.0**: Captures main event/announcement, key companies/people, and all relevant financial figures
* **0.8-0.9**: Captures main elements with minor omissions of secondary details
* **0.6-0.7**: Captures main event but misses some key companies, people, or financial figures
* **0.4-0.5**: Captures basic information but omits several important elements
* **0.2-0.3**: Misses main event or most key entities/figures
* **0.0-0.1**: Fails to capture the primary purpose or content of the article

**Conciseness (Length & Focus):**
* **1.0**: Exactly 2-4 sentences, perfect balance of brevity and essential information
* **0.8-0.9**: 2-4 sentences with excellent focus, minor wordiness or slight under-coverage
* **0.6-0.7**: Appropriate length but some unnecessary details or missing key points
* **0.4-0.5**: Too long (5+ sentences) or too short (1 sentence), affects information balance
* **0.2-0.3**: Significantly too long or too short, poor information prioritization
* **0.0-0.1**: Extremely poor length control, verbose or overly terse

**Clarity (Professional Communication):**
* **1.0**: Clear, professional language with no headings/labels/preamble, excellent readability
* **0.8-0.9**: Very clear and professional with minor style issues
* **0.6-0.7**: Generally clear but some awkward phrasing or minor formatting issues
* **0.4-0.5**: Readable but includes unwanted headings/labels or unclear language
* **0.2-0.3**: Poor clarity, includes significant formatting issues or confusing language
* **0.0-0.1**: Very unclear, includes headings/preamble, or incomprehensible

**Overall Quality Guidelines:**
* **Excellent**: All dimensions score 0.8+, meets all agent requirements, publication-ready
* **Good**: Most dimensions score 0.6+, minor improvements needed, mostly follows requirements
* **Fair**: Mixed performance, significant improvements needed, partially follows requirements
* **Poor**: Multiple dimensions below 0.5, major revision required, fails key requirements

**Output Format:**
Your evaluation *must* be structured as a nested JSON dictionary with the following top-level key: "Summary Evaluation". Please return NULL if any of the inputs are empty or invalid.

The value for "Summary Evaluation" should be a dictionary containing:
- "Accuracy" (float 0-1)
- "Completeness" (float 0-1) 
- "Conciseness" (float 0-1)
- "Clarity" (float 0-1)
- "Overall Quality" (string: "excellent", "good", "fair", or "poor")
- "Explanation" (string with detailed reasoning)

Make sure you return a valid JSON string. Pay special attention to quotes, commas and special characters in the JSON string. Make sure to escape all special characters and quotes in the JSON string.

**Example:**
```json
{{
  "Summary Evaluation": {{
    "Accuracy": 0.9,
    "Completeness": 0.8,
    "Conciseness": 0.9,
    "Clarity": 0.9,
    "Overall Quality": "good",
    "Explanation": "The summary is factually accurate and grounded in the article content with no hallucinations. It captures the main announcement and key company involved, though it misses one significant financial figure mentioned. The summary is exactly 3 sentences, meeting the length requirement perfectly. The writing is clear, professional, and contains no headings or preamble as required."
  }}
}}
```

**Now, proceed with the evaluation using the provided article and AI summary.**

Original Article:
Title: {title}

Body: {body}

--------------------
AI-Generated Summary to Evaluate:
{summary}

--------------------
Rating:
"""


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