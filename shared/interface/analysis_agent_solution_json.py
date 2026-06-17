from __future__ import annotations

from typing import Any, List

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

CATEGORY_BIN_KEYS = (
    "plotting",
    "manual_data_inspection",
    "statistical_summaries",
    "context_equations_and_metrics",
    "pretrained_knowledge",
    "generic_feature_engineering",
    "model_based_estimation",
)
JUSTIFICATION_FIELDS = tuple(f"{key}_justification" for key in CATEGORY_BIN_KEYS)
STEP_ID_FIELDS = tuple(f"{key}_step_ids" for key in CATEGORY_BIN_KEYS)
ANALYSIS_REQUIRED_FIELDS = (
    ("is_correct", "Agent JSON missing 'is_correct'"),
    ("correct_answer_flawed_reasoning", "Agent JSON missing 'correct_answer_flawed_reasoning'"),
    ("correct_answer_flawed_reasoning_step_ids", "Agent JSON missing 'correct_answer_flawed_reasoning_step_ids'"),
    ("correct_answer_flawed_reasoning_motivation", "Agent JSON missing 'correct_answer_flawed_reasoning_motivation'"),
    ("category_bins", "Agent JSON missing 'category_bins'"),
    ("used_packages", "Agent JSON missing 'used_packages'"),
    ("cheating", "Agent JSON missing 'cheating'"),
    ("cheating_notes", "Agent JSON missing 'cheating_notes'"),
)
OUTPUT_REQUIRED_FIELDS = (
    ("bypass", "Agent JSON missing 'bypass'"),
    ("iterations", "Agent JSON missing 'iterations'"),
    ("observation_output_chars", "Agent JSON missing 'observation_output_chars'"),
)


def is_int(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return value.is_integer()
    return False


def normalize_step_ids(value: object, field_name: str) -> str:
    if isinstance(value, str):
        return value
    if is_int(value):
        return str(int(value))
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif is_int(item):
                parts.append(str(int(item)))
            else:
                raise TypeError(
                    f"category_bins '{field_name}' must be a string, int, or list of ints"
                )
        return ",".join(parts)
    raise TypeError(
        f"category_bins '{field_name}' must be a string, int, or list of ints"
    )


class CategoryBins(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    plotting: int
    plotting_justification: str
    plotting_step_ids: str
    manual_data_inspection: int
    manual_data_inspection_justification: str
    manual_data_inspection_step_ids: str
    statistical_summaries: int
    statistical_summaries_justification: str
    statistical_summaries_step_ids: str
    context_equations_and_metrics: int
    context_equations_and_metrics_justification: str
    context_equations_and_metrics_step_ids: str
    pretrained_knowledge: int
    pretrained_knowledge_justification: str
    pretrained_knowledge_items: List[str]
    pretrained_knowledge_step_ids: str
    generic_feature_engineering: int
    generic_feature_engineering_justification: str
    generic_feature_engineering_generated_features: List[str]
    generic_feature_engineering_step_ids: str
    model_based_estimation: int
    model_based_estimation_justification: str
    model_based_estimation_step_ids: str

    @model_validator(mode="before")
    @classmethod
    def _require_keys(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise TypeError("Agent JSON 'category_bins' must be an object")
        missing_bins = set(CATEGORY_BIN_KEYS).difference(value.keys())
        if missing_bins:
            raise ValueError(f"category_bins missing keys: {sorted(missing_bins)}")
        missing_justifications = set(JUSTIFICATION_FIELDS).difference(value.keys())
        if missing_justifications:
            raise ValueError(
                "category_bins missing justification keys: "
                f"{sorted(missing_justifications)}"
            )
        missing_step_ids = set(STEP_ID_FIELDS).difference(value.keys())
        if missing_step_ids:
            raise ValueError(
                f"category_bins missing step_ids keys: {sorted(missing_step_ids)}"
            )
        if "pretrained_knowledge_items" not in value:
            raise ValueError("category_bins missing key: pretrained_knowledge_items")
        if "generic_feature_engineering_generated_features" not in value:
            raise ValueError(
                "category_bins missing key: generic_feature_engineering_generated_features"
            )
        return value

    @field_validator(*CATEGORY_BIN_KEYS, mode="before")
    @classmethod
    def _validate_bins(cls, value: Any, info: Any) -> int:
        key = info.field_name
        if not is_int(value):
            raise TypeError(f"category_bins '{key}' must be an integer")
        normalized = int(value)
        if normalized < 0 or normalized > 3:
            raise ValueError(f"category_bins '{key}' must be between 0 and 3")
        return normalized

    @field_validator(*JUSTIFICATION_FIELDS, mode="before")
    @classmethod
    def _validate_justifications(cls, value: Any, info: Any) -> str:
        if not isinstance(value, str):
            raise TypeError(f"category_bins '{info.field_name}' must be a string")
        return value

    @field_validator(*STEP_ID_FIELDS, mode="before")
    @classmethod
    def _normalize_step_ids(cls, value: Any, info: Any) -> str:
        return normalize_step_ids(value, info.field_name)

    @field_validator("pretrained_knowledge_items", mode="before")
    @classmethod
    def _validate_pretrained_knowledge_items(cls, value: Any) -> List[str]:
        if not isinstance(value, list):
            raise TypeError("category_bins 'pretrained_knowledge_items' must be a list")
        if not all(isinstance(item, str) for item in value):
            raise TypeError(
                "category_bins 'pretrained_knowledge_items' must be a list of strings"
            )
        return value

    @field_validator("generic_feature_engineering_generated_features", mode="before")
    @classmethod
    def _validate_generic_feature_engineering_generated_features(
        cls, value: Any
    ) -> List[str]:
        if not isinstance(value, list):
            raise TypeError(
                "category_bins 'generic_feature_engineering_generated_features' must be a list"
            )
        if not all(isinstance(item, str) for item in value):
            raise TypeError(
                "category_bins 'generic_feature_engineering_generated_features' must be a list of strings"
            )
        return value


class AnalysisAgentDecision(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    is_correct: bool | None
    correct_answer_flawed_reasoning: bool | None
    correct_answer_flawed_reasoning_step_ids: List[int]
    correct_answer_flawed_reasoning_motivation: str
    category_bins: CategoryBins
    used_packages: List[str]
    cheating: bool
    cheating_notes: str

    @model_validator(mode="before")
    @classmethod
    def _require_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("Agent response JSON must be an object")
        for field_name, message in ANALYSIS_REQUIRED_FIELDS:
            if field_name not in value:
                raise ValueError(message)
        return value

    @field_validator("is_correct", mode="before")
    @classmethod
    def _validate_is_correct(cls, value: Any) -> bool | None:
        if value is None or isinstance(value, bool):
            return value
        raise TypeError("Agent JSON 'is_correct' must be true, false, or null")

    @field_validator("correct_answer_flawed_reasoning", mode="before")
    @classmethod
    def _validate_correct_answer_flawed_reasoning(cls, value: Any) -> bool | None:
        if value is None or isinstance(value, bool):
            return value
        raise TypeError("Agent JSON 'correct_answer_flawed_reasoning' must be true, false, or null")

    @field_validator("correct_answer_flawed_reasoning_step_ids", mode="before")
    @classmethod
    def _validate_correct_answer_flawed_reasoning_steps(cls, value: Any) -> List[int]:
        if not isinstance(value, list):
            raise TypeError("Agent JSON 'correct_answer_flawed_reasoning_step_ids' must be a list")
        if not all(
            isinstance(step, int) and not isinstance(step, bool)
            for step in value
        ):
            raise TypeError(
                "Agent JSON 'correct_answer_flawed_reasoning_step_ids' must be a list of integers"
            )
        return value

    @field_validator("correct_answer_flawed_reasoning_motivation", mode="before")
    @classmethod
    def _validate_correct_answer_flawed_reasoning_motivation(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("Agent JSON 'correct_answer_flawed_reasoning_motivation' must be a string")
        return value

    @field_validator("category_bins", mode="before")
    @classmethod
    def _validate_category_bins(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise TypeError("Agent JSON 'category_bins' must be an object")
        return value

    @field_validator("used_packages", mode="before")
    @classmethod
    def _validate_used_packages(cls, value: Any) -> List[str]:
        if not isinstance(value, list):
            raise TypeError("Agent JSON 'used_packages' must be a list")
        if not all(isinstance(item, str) for item in value):
            raise TypeError("Agent JSON 'used_packages' must be a list of strings")
        return value

    @field_validator("cheating", mode="before")
    @classmethod
    def _validate_cheating(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise TypeError("Agent JSON 'cheating' must be a boolean")
        return value

    @field_validator("cheating_notes", mode="before")
    @classmethod
    def _validate_cheating_notes(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise TypeError("Agent JSON 'cheating_notes' must be a string")
        return value


class AnalysisAgentSolution(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True)

    bypass: bool
    iterations: int
    observation_output_chars: int

    @model_validator(mode="before")
    @classmethod
    def _require_fields(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            raise ValueError("Agent response JSON must be an object")
        for field_name, message in OUTPUT_REQUIRED_FIELDS:
            if field_name not in value:
                raise ValueError(message)
        bypass = value["bypass"]
        if not isinstance(bypass, bool):
            raise TypeError("Agent JSON 'bypass' must be a boolean")
        if bypass:
            forbidden = {
                field_name
                for field_name, _message in ANALYSIS_REQUIRED_FIELDS
                if field_name in value
            }
            if forbidden:
                raise ValueError(
                    "Agent JSON must omit analysis fields when bypass=true: "
                    f"{sorted(forbidden)}"
                )
        else:
            analysis_payload = dict(value)
            for field_name, _message in OUTPUT_REQUIRED_FIELDS:
                analysis_payload.pop(field_name, None)
            AnalysisAgentDecision.model_validate(analysis_payload)
        return value

    @field_validator("bypass", mode="before")
    @classmethod
    def _validate_bypass(cls, value: Any) -> bool:
        if not isinstance(value, bool):
            raise TypeError("Agent JSON 'bypass' must be a boolean")
        return value

    @field_validator("iterations", "observation_output_chars", mode="before")
    @classmethod
    def _validate_metric_ints(cls, value: Any, info: Any) -> int:
        if not is_int(value):
            raise TypeError(f"Agent JSON '{info.field_name}' must be an integer")
        normalized = int(value)
        if normalized < 0:
            raise ValueError(f"Agent JSON '{info.field_name}' must be >= 0")
        return normalized
