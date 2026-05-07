from __future__ import annotations

import json
import re
from typing import Any, List, Literal, Mapping, Optional, Sequence

from shared.tsenv_combinations import TIME0_BASELINE_AGENT_FACING_LABEL

_OPTIONAL_BLOCK_RE = re.compile(r"\[(.*?)\]", flags=re.DOTALL)
_MULTIPLE_BLANK_LINES_RE = re.compile(r"\n{3,}")
_TRAILING_WHITESPACE_RE = re.compile(r"[ \t]+\n")
_TSENV_PROMPT_PLACEHOLDER_RE = re.compile(r"\{([^{}\n]+)\}")
FINAL_ANSWER_KEY = "final_answer"
TSENV_DOCUMENTED_PROMPT_FIELDS = [
    "sample_source",
    "environment_description",
    "observed_columns",
    "intervention_semantics",
    "label_space",
    "no_change_guidance",
    "task_artifact",
    "prediction_format",
    "fewshot_context",
    "mode_specific_requirements",
    "evaluation",
    "runtime_constraints",
]
TSENV_LEGACY_PROMPT_FIELDS = [
    "first_sentence",
    "model_description",
    "shared_description",
    "task_instruction",
]


def _default_prompt_field_entries(question_text: Mapping[str, Any]) -> List[tuple[str, str]]:
    fields = (
        TSENV_DOCUMENTED_PROMPT_FIELDS
        if any(field in question_text for field in TSENV_DOCUMENTED_PROMPT_FIELDS)
        else TSENV_LEGACY_PROMPT_FIELDS
    )
    return [
        (field, "\n\n" if index < len(fields) - 1 else "")
        for index, field in enumerate(fields)
    ]


def tsenv_prompt_field_entries(question_text: Mapping[str, Any]) -> List[tuple[str, str]]:
    field_order_raw = question_text.get("ordered_field_agent_prompt")
    if not (isinstance(field_order_raw, list) and field_order_raw):
        return _default_prompt_field_entries(question_text)

    entries: List[tuple[str, str]] = []
    for index, raw_entry in enumerate(field_order_raw):
        if isinstance(raw_entry, (list, tuple)):
            if not raw_entry:
                continue
            field = str(raw_entry[0]).strip()
            separator = str(raw_entry[1]) if len(raw_entry) > 1 else "\n\n"
        else:
            field = str(raw_entry).strip()
            separator = "\n\n"
        if not field:
            continue
        if index == len(field_order_raw) - 1 and not isinstance(raw_entry, (list, tuple)):
            separator = ""
        entries.append((field, separator))
    return entries or _default_prompt_field_entries(question_text)


def _count_to_english_words(value: int) -> str:
    try:
        from num2words import num2words  # type: ignore[import-not-found]
    except Exception:
        return str(value)
    try:
        return str(num2words(int(value), lang="en"))
    except Exception:
        return str(value)


def classification_results_payload() -> str:
    return "{\"final_answer\": \"<class_label>\"}"


def tsenv_classification_results_payload() -> str:
    return "{\"change_time\": <t>, \"final_answer\": \"<class_label>\"}"


def tsenv_direct_results_payload() -> str:
    return "{\"<filename>.parquet\": [\"<possible_answer_1>\",...,\"<possible_answer_n>\"], ...}"


def tsenv_open_ended_results_payload() -> str:
    return "{\"<filename>.parquet\": \"<your answer>\", ...}"


def tsenv_code_results_payload() -> str:
    return "{\"rule_file\": \"rule.py\"}"


_TOOL_AND_INTERNET_LINE = (
    "If you need to create any intermediate file, image, or artifact to solve "
    "the task, please do so in the current working directory.\n"
    "You are free to use any approach you deem useful to solve the task; "
    "however, note that internet access is disabled."
)


InstructionKind = Literal["classification", "anomaly"]


def _general_instruction_lines() -> List[str]:
    return ["", _TOOL_AND_INTERNET_LINE]


def _format_multiple_choices(multiple_choices: Sequence[object]) -> str:
    try:
        return json.dumps(list(multiple_choices))
    except TypeError:
        return json.dumps([str(choice) for choice in multiple_choices])


def _format_natural_quoted_list(values: Sequence[object]) -> str:
    quoted = [json.dumps(str(value)) for value in values]
    if not quoted:
        return ""
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def tsenv_parameter_question_sentence(
    multiple_choices: Sequence[object] = (),
) -> str:
    choices = [str(choice) for choice in multiple_choices]
    candidate_choices = [
        choice for choice in choices if choice != TIME0_BASELINE_AGENT_FACING_LABEL
    ]
    num_scored_choices = len(choices)
    return (
        "For each test sample, return the smallest set of labels that you believe contains the true label with high confidence, based on the observed time series.\n"
        "For each sample, either no parameter changed or exactly one candidate parameter from "
        f"{_format_natural_quoted_list(candidate_choices)} changed during the observed simulation interval.\n"
        f"If you cannot detect any parameter change, return [\"{TIME0_BASELINE_AGENT_FACING_LABEL}\"].\n"
        "If a parameter changed, assume it changed abruptly at a single time point and produced an observable effect within the recorded interval.\n"
        "You may assume that, in the corresponding noiseless trajectory, the correct changed parameter would be clearly distinguishable from "
        f"the other candidate parameters and from the \"{TIME0_BASELINE_AGENT_FACING_LABEL}\" case.\n\n"
        "Always return a non-empty list of unique labels.\n"
        "If you are confident, only return a list with one label.\n"
        "When uncertain, you may return multiple distinct allowed labels.\n\n"
        "Scoring:\n"
        f"Let N = {num_scored_choices}, the total number of allowed labels including "
        f"\"{TIME0_BASELINE_AGENT_FACING_LABEL}\".\n"
        "Let m be the number of labels returned for a given sample.\n\n"
        "If the correct label is included among the returned labels, the score is:\n\n"
        "score = (N - m) / (N - 1)\n\n"
        "Otherwise, the score is 0.\n"
        "Note that there is only one correct answer per question."
    )


def classification_instruction_lines(
    multiple_choices: Sequence[object] = (),
) -> List[str]:
    formatted_choices = _format_multiple_choices(multiple_choices)
    return [
        f"Save the solution in a file called results.json with this format {classification_results_payload()}",
        f"where <class_label> must be one of: {formatted_choices}.",
        *_general_instruction_lines(),
    ]


def tsenv_direct_classification_instruction_lines(
    multiple_choices: Sequence[object] = (),
) -> List[str]:
    formatted_choices = _format_multiple_choices(multiple_choices)
    return [
        "Output format instructions:",
        "",
        "Save a `results.json` file in the following format:",
        "{",
        '  "<filename>.parquet": ["<possible_answer_1>",...,"<possible_answer_n>"],',
        "  ...",
        "}",
        "",
        "Requirements:",
        "1. The keys must exactly match the test sample filenames.",
        f"2. Each `<possible_answer_1>` must be one of the choices in {formatted_choices}.",
        *_general_instruction_lines(),
    ]


def tsenv_code_rule_instruction_lines(
    multiple_choices: Sequence[object] = (),
) -> List[str]:
    formatted_choices = _format_multiple_choices(multiple_choices)
    return [
        "Output format:",
        "1. Create `rule.py` with a function `predict(df) -> list[str]`.",
        "",
        "Requirements:",
        "- `rule.py` is executed on every parquet file under `test_samples/`.",
        "- Use the same `predict` function for all test samples.",
        f"- The returned list must be one of: {formatted_choices}.",
        "- Do not provide per-sample hardcoded outputs.",
        "- `rule.py` must be self-contained and must not depend on any other local file.",
    ]


def tsenv_direct_predictions_instruction_lines(
    multiple_choices: Sequence[object] = (),
) -> List[str]:
    return tsenv_direct_classification_instruction_lines(multiple_choices)[:-2]


def tsenv_direct_predictions_instruction_block(
    multiple_choices: Sequence[object] = (),
) -> str:
    return join_message_lines(
        tsenv_direct_predictions_instruction_lines(multiple_choices)
    )


def tsenv_code_rule_instruction_block(
    multiple_choices: Sequence[object] = (),
) -> str:
    return join_message_lines(tsenv_code_rule_instruction_lines(multiple_choices))


def _tsenv_split_description_and_question(instruction_human_format: str) -> tuple[str, str]:
    text = join_message_lines([instruction_human_format]).strip()
    if not text:
        return "", ""

    match = re.search(
        r"(?:^|\n\n)(For each test sample, determine which single outcome is best supported by the observed time series\.|For each test sample, determine whether there is evidence that one of the candidate parameters |For each test sample, exactly one of the following parameters changes suddenly once during the simulation:|For each test sample, exactly one of the following parameters changes suddenly once during the simulation\.|In this simulation, exactly one parameter from |Exactly one parameter from |At most one parameter from |At most one parameter in |One of the parameters in |One of the outcomes in )",
        text,
    )
    if match is None:
        return "", text

    question_start = match.start(1)
    description = text[:question_start].strip()
    question = _tsenv_normalize_question_text(text[question_start:].strip())
    return description, question


def _tsenv_normalize_question_text(question_text: str) -> str:
    text = str(question_text or "").replace("\r\n", "\n").replace("\r", "\n")
    q1_re = re.compile(
        r"^\s*(?:\(\s*1\s*\)\s*)?At what time does the change first become detectable in the data\?\s*$",
        flags=re.IGNORECASE,
    )
    q2_re = re.compile(
        r"^\s*(?:\(\s*2\s*\)\s*)?Which parameter was changed\?\s*$",
        flags=re.IGNORECASE,
    )
    count_re = re.compile(
        r"^You are given \d+ training and \d+ test samples generated by a simulator\.$"
    )
    removable_prefixes = (
        "The data you need to check is available at `dataframe.parquet`.",
        "The data you need to check is available in `test_samples/*.parquet`.",
        "The data to be classified is stored in the test_samples folder.",
        "The labeled examples are stored in the train_samples folder, with ",
        "Save the solution in a file called results.json with this format ",
        "Save `results.json` with:",
        "where <t> must be a value from the `time` column",
        "where <class_label> must be one of:",
        "- The root JSON object must contain exactly one entry per parquet file under `test_samples/`.",
        "- Keys must exactly match those task-local sample paths.",
        "- Each value must be one of:",
        "- `rule.py` must be self-contained and must not depend on any other local file.",
        "- Do not provide per-sample hardcoded outputs.",
        "1. Create `rule.py` following the environment profile `solutions/rule.py` interface.",
        "1. Create `rule.py` with a function `predict(df) -> list[str]`.",
        "- `rule.py` is executed on every parquet file under `test_samples/`.",
        "- Use the same `predict` function for all test samples.",
        "- The returned list must be one of:",
        "2. Save `results.json` with:",
    )
    normalized: List[str] = []
    for raw_line in text.split("\n"):
        stripped = raw_line.strip()
        if not stripped:
            normalized.append("")
            continue
        if stripped in {
            "Questions:",
            "Output format:",
            "Evaluation contract:",
            "If you need to create any intermediate file, image, or artifact to solve the task, please do so in the current working directory.",
            "You are free to use any approach you deem useful to solve the task; however, note that internet access is disabled.",
        }:
            continue
        if q1_re.match(stripped):
            continue
        if q2_re.match(stripped):
            normalized.append("Which parameter was changed?")
            continue
        if count_re.match(stripped):
            continue
        if any(stripped.startswith(prefix) for prefix in removable_prefixes):
            continue
        normalized.append(raw_line.rstrip())
    return join_message_lines(normalized)


def _tsenv_instruction_data_location(*, shot_count_per_class: int) -> str:
    if shot_count_per_class <= 0:
        return "The data to be classified is stored in the test_samples folder."
    return (
        "The labeled examples are stored in the train_samples folder, with "
        f"{int(shot_count_per_class)} labeled examples per class.\n"
        "The label is included in the file name. "
        "The data to be classified is stored in the test_samples folder."
    )


def _tsenv_instruction_format(
    *,
    eval_mode: Literal["direct", "code"],
    multiple_choices: Sequence[object],
) -> str:
    formatted_choices = _format_multiple_choices(multiple_choices)
    if eval_mode == "direct":
        return join_message_lines(
            [
                "Output format instructions:",
                "",
                "Save a `results.json` file in the following format:",
                "{",
                '  "<filename>.parquet": ["<possible_answer_1>",...,"<possible_answer_n>"],',
                "  ...",
                "}",
                "",
                "Requirements:",
                "1. The keys must exactly match the test sample filenames.",
                f"2. Each `<possible_answer_1>` must be one of the choices in {formatted_choices}.",
            ]
        )
    return tsenv_code_rule_instruction_block(multiple_choices)


def tsenv_classification_prompt_parts(
    instruction_human_format: str,
    multiple_choices: Sequence[object],
    *,
    train_sample_count: int,
    eval_mode: Literal["direct", "code"],
) -> dict[str, str]:
    description, question = _tsenv_split_description_and_question(instruction_human_format)
    parts = {
        "description": description,
        "question": question,
        "instruction_data_location": _tsenv_instruction_data_location(
            shot_count_per_class=int(train_sample_count)
        ),
        "instruction_format": _tsenv_instruction_format(
            eval_mode=eval_mode,
            multiple_choices=multiple_choices,
        ),
        "general_instruction": _TOOL_AND_INTERNET_LINE,
    }
    return parts


def tsenv_render_prompt_parts(parts: dict[str, str]) -> str:
    ordered_keys = (
        "description",
        "question",
        "instruction_data_location",
        "instruction_format",
        "general_instruction",
    )
    return join_message_lines([parts[key] for key in ordered_keys if str(parts.get(key) or "").strip()])


def _stringify_tsenv_prompt_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value))
    if isinstance(value, Mapping):
        return json.dumps(dict(value), sort_keys=True)
    return str(value or "")


def _resolve_mapping_path(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = mapping
    for part in path:
        key = str(part).strip()
        if not key:
            raise KeyError("empty placeholder path segment")
        if isinstance(value, Mapping):
            if key not in value:
                raise KeyError(key)
            value = value[key]
            continue
        raise KeyError(key)
    return value


def _question_text_for_placeholder(
    *,
    target_slug: str | None,
    current_question_text: Mapping[str, Any],
    current_question_slug: str | None,
    questions_by_id: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any]:
    if not target_slug or target_slug == current_question_slug:
        return current_question_text
    if questions_by_id is None:
        raise KeyError(target_slug)
    target_question = questions_by_id.get(target_slug)
    if not isinstance(target_question, Mapping):
        raise KeyError(target_slug)
    target_question_text = target_question.get("question_text")
    if not isinstance(target_question_text, Mapping):
        raise KeyError(f"{target_slug}.question_text")
    return target_question_text


def _placeholder_target_slug(
    raw_slug: str | None,
    *,
    current_question_slug: str | None,
) -> str | None:
    target_slug = str(raw_slug or "").strip()
    if target_slug.startswith("<") and target_slug.endswith(">"):
        target_slug = target_slug[1:-1].strip()
    if target_slug == "question_slug":
        return current_question_slug
    return target_slug or current_question_slug


def _resolve_tsenv_question_text_placeholder(
    *,
    body: str,
    target_slug: str | None,
    field_path: str,
    current_question_text: Mapping[str, Any],
    current_question_slug: str | None,
    questions_by_id: Mapping[str, Mapping[str, Any]] | None,
    questions_metadata: Mapping[str, Any] | None,
    stack: tuple[str, ...],
) -> str:
    if not field_path:
        raise KeyError(body)
    lookup_key = f"{target_slug or '<current>'}.question_text.{field_path}"
    if lookup_key in stack:
        raise ValueError(f"Recursive tsENV prompt placeholder: {lookup_key}")
    source_text = _question_text_for_placeholder(
        target_slug=target_slug,
        current_question_text=current_question_text,
        current_question_slug=current_question_slug,
        questions_by_id=questions_by_id,
    )
    try:
        resolved_value = _resolve_mapping_path(source_text, field_path.split("."))
    except KeyError as exc:
        raise KeyError(f"Could not resolve tsENV prompt placeholder {{{body}}}") from exc
    return _resolve_tsenv_prompt_placeholders(
        resolved_value,
        current_question_text=current_question_text,
        current_question_slug=current_question_slug,
        questions_by_id=questions_by_id,
        questions_metadata=questions_metadata,
        stack=(*stack, lookup_key),
    )


def _resolve_tsenv_questions_namespace_placeholder(
    *,
    body: str,
    current_question_text: Mapping[str, Any],
    current_question_slug: str | None,
    questions_by_id: Mapping[str, Mapping[str, Any]] | None,
    questions_metadata: Mapping[str, Any] | None,
    stack: tuple[str, ...],
) -> str:
    tail = body.removeprefix("questions.").strip()
    if not tail:
        raise KeyError(body)
    target_slug, separator, field_path = tail.partition(".question_text.")
    if separator:
        return _resolve_tsenv_question_text_placeholder(
            body=body,
            target_slug=_placeholder_target_slug(
                target_slug,
                current_question_slug=current_question_slug,
            ),
            field_path=field_path,
            current_question_text=current_question_text,
            current_question_slug=current_question_slug,
            questions_by_id=questions_by_id,
            questions_metadata=questions_metadata,
            stack=stack,
        )
    if questions_metadata is None:
        raise KeyError(body)
    lookup_key = f"questions.{tail}"
    if lookup_key in stack:
        raise ValueError(f"Recursive tsENV prompt placeholder: {lookup_key}")
    try:
        resolved_value = _resolve_mapping_path(questions_metadata, tail.split("."))
    except KeyError as exc:
        raise KeyError(f"Could not resolve tsENV prompt placeholder {{{body}}}") from exc
    return _resolve_tsenv_prompt_placeholders(
        resolved_value,
        current_question_text=current_question_text,
        current_question_slug=current_question_slug,
        questions_by_id=questions_by_id,
        questions_metadata=questions_metadata,
        stack=(*stack, lookup_key),
    )


def _resolve_tsenv_prompt_placeholder_body(
    body: str,
    *,
    current_question_text: Mapping[str, Any],
    current_question_slug: str | None,
    questions_by_id: Mapping[str, Mapping[str, Any]] | None,
    questions_metadata: Mapping[str, Any] | None,
    stack: tuple[str, ...],
) -> str | None:
    if body.startswith("questions."):
        return _resolve_tsenv_questions_namespace_placeholder(
            body=body,
            current_question_text=current_question_text,
            current_question_slug=current_question_slug,
            questions_by_id=questions_by_id,
            questions_metadata=questions_metadata,
            stack=stack,
        )
    if body.startswith("<question_slug>.question_text."):
        return _resolve_tsenv_question_text_placeholder(
            body=body,
            target_slug=current_question_slug,
            field_path=body.removeprefix("<question_slug>.question_text."),
            current_question_text=current_question_text,
            current_question_slug=current_question_slug,
            questions_by_id=questions_by_id,
            questions_metadata=questions_metadata,
            stack=stack,
        )
    if body.startswith("question_text."):
        return _resolve_tsenv_question_text_placeholder(
            body=body,
            target_slug=current_question_slug,
            field_path=body.removeprefix("question_text."),
            current_question_text=current_question_text,
            current_question_slug=current_question_slug,
            questions_by_id=questions_by_id,
            questions_metadata=questions_metadata,
            stack=stack,
        )
    target_slug, separator, field_path = body.partition(".question_text.")
    if not separator:
        return None
    return _resolve_tsenv_question_text_placeholder(
        body=body,
        target_slug=_placeholder_target_slug(
            target_slug,
            current_question_slug=current_question_slug,
        ),
        field_path=field_path,
        current_question_text=current_question_text,
        current_question_slug=current_question_slug,
        questions_by_id=questions_by_id,
        questions_metadata=questions_metadata,
        stack=stack,
    )


def _resolve_tsenv_prompt_placeholders(
    value: Any,
    *,
    current_question_text: Mapping[str, Any],
    current_question_slug: str | None,
    questions_by_id: Mapping[str, Mapping[str, Any]] | None,
    questions_metadata: Mapping[str, Any] | None,
    stack: tuple[str, ...] = (),
) -> str:
    text = _stringify_tsenv_prompt_value(value)
    if "question_text." not in text and "{questions." not in text:
        return text.strip()

    def _replace(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        resolved = _resolve_tsenv_prompt_placeholder_body(
            body,
            current_question_text=current_question_text,
            current_question_slug=current_question_slug,
            questions_by_id=questions_by_id,
            questions_metadata=questions_metadata,
            stack=stack,
        )
        return match.group(0) if resolved is None else resolved

    return _TSENV_PROMPT_PLACEHOLDER_RE.sub(_replace, text).strip()


def render_tsenv_agent_prompt(
    question_text: Mapping[str, Any],
    *,
    question_slug: str | None = None,
    questions_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    questions_metadata: Mapping[str, Any] | None = None,
) -> str:
    prompt_parts: List[str] = []
    pending_separator = ""
    for field, separator in tsenv_prompt_field_entries(question_text):
        if field not in question_text:
            if prompt_parts:
                pending_separator = separator
            continue
        rendered = _resolve_tsenv_prompt_placeholders(
            question_text.get(field),
            current_question_text=question_text,
            current_question_slug=str(question_slug or "").strip() or None,
            questions_by_id=questions_by_id,
            questions_metadata=questions_metadata,
        )
        if rendered:
            if prompt_parts:
                prompt_parts.append(pending_separator)
            prompt_parts.append(rendered)
            pending_separator = separator
        elif prompt_parts:
            pending_separator = separator
    return "".join(prompt_parts)


def tsenv_direct_prompt_stem_from_agent_prompt(agent_prompt: str) -> str:
    description, question = _tsenv_split_description_and_question(str(agent_prompt or ""))
    return join_message_lines([description, question])


def tsenv_code_prompt_stem_from_agent_prompt(agent_prompt: str) -> str:
    description, question = _tsenv_split_description_and_question(str(agent_prompt or ""))
    return join_message_lines([description, question])


def agent_prompt_tsenv_code_classification(
    instruction_human_format: str,
    multiple_choices: Sequence[object],
    *,
    train_sample_count: int,
    test_sample_count: int,
) -> str:
    del test_sample_count
    return tsenv_render_prompt_parts(
        tsenv_classification_prompt_parts(
            instruction_human_format,
            multiple_choices,
            train_sample_count=train_sample_count,
            eval_mode="code",
        )
    )


def agent_prompt_tsenv_direct_classification(
    instruction_human_format: str,
    multiple_choices: Sequence[object],
    *,
    train_sample_count: int,
    test_sample_count: int,
) -> str:
    del test_sample_count
    return tsenv_render_prompt_parts(
        tsenv_classification_prompt_parts(
            instruction_human_format,
            multiple_choices,
            train_sample_count=train_sample_count,
            eval_mode="direct",
        )
    )


def anomaly_instruction_lines() -> List[str]:
    return [*_general_instruction_lines()]


def instruction_lines(
    kind: InstructionKind, *, multiple_choices: Sequence[object] | None = None
) -> List[str]:
    if kind == "classification":
        if multiple_choices is None:
            raise ValueError("multiple_choices is required for classification instructions.")
        return classification_instruction_lines(multiple_choices)
    if kind == "anomaly":
        return anomaly_instruction_lines()
    raise ValueError(f"Unknown instruction kind: {kind}")


def apply_optional_bracket_blocks(text: str, *, include: bool) -> str:
    """Expand optional blocks written as `[ ... ]`.

    - If `include=True`, keep the contents and drop the surrounding brackets.
    - If `include=False`, remove the entire block.
    """
    rendered = str(text or "")
    if "[" not in rendered or "]" not in rendered:
        return rendered.strip()

    def _replace(match: re.Match) -> str:
        # Preserve JSON arrays like `["a", "b"]` which are used to render choice lists.
        # Optional blocks are a prompt-authoring convenience, not a general bracket syntax.
        try:
            parsed = json.loads(match.group(0))
        except Exception:
            parsed = None
        if isinstance(parsed, list):
            return match.group(0)
        return match.group(1) if include else ""

    rendered = _OPTIONAL_BLOCK_RE.sub(_replace, rendered)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    return rendered.strip()


def join_message_lines(lines: Sequence[object]) -> str:
    """Join prompt fragments into a single prompt string with stable whitespace.

    Guarantees:
    - No fragment starts with a newline.
    - No runs of >=2 empty lines (i.e. no '\\n\\n\\n').
    - No trailing whitespace at line ends.
    """

    parts: List[str] = []
    for line in lines:
        text = str(line or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.lstrip("\n")
        text = _TRAILING_WHITESPACE_RE.sub("\n", text)
        text = "\n".join(piece.rstrip() for piece in text.splitlines())
        parts.append(text)

    rendered = "\n".join(parts)
    rendered = rendered.lstrip("\n")
    rendered = _MULTIPLE_BLANK_LINES_RE.sub("\n\n", rendered)
    rendered = _TRAILING_WHITESPACE_RE.sub("\n", rendered)
    return rendered.rstrip()


def fill_question_template(
    template: str,
    *,
    time: object = None,
    choices: object = None,
    signals: Sequence[str] = (),
    signal_text: object = None,
) -> str:
    text = str(template or "").strip()
    if not text:
        return ""
    text = apply_optional_bracket_blocks(text, include=bool(signals))
    if time is not None and ("{time}" in text or "{}" in text):
        text = text.replace("{time}", str(time))
        text = text.replace("{}", str(time))
    if "{number_of_signals}" in text or "{num_signals}" in text or "{signal_count}" in text:
        count = len([item for item in signals if str(item).strip()])
        text = (
            text.replace("{number_of_signals}", str(count))
            .replace("{num_signals}", str(count))
            .replace("{signal_count}", str(count))
        )
    if "{number_of_candidates}" in text or "{num_candidates}" in text or "{candidate_count}" in text:
        count = 0
        if isinstance(choices, (list, tuple)):
            count = len([item for item in choices if str(item).strip()])
        elif choices is not None:
            raw = str(choices)
            count = len([item for item in raw.split(",") if item.strip()])
        text = (
            text.replace("{number_of_candidates}", str(count))
            .replace("{num_candidates}", str(count))
            .replace("{candidate_count}", str(count))
        )
    if choices is not None and (
        "{choices}" in text
        or "{options}" in text
        or "{multiple_choices}" in text
        or "{multipe_choices}" in text
    ):
        if isinstance(choices, (list, tuple)):
            raw = _format_multiple_choices(choices)
        else:
            raw = str(choices)
        text = (
            text.replace("{choices}", raw)
            .replace("{options}", raw)
            .replace("{multiple_choices}", raw)
            .replace("{multipe_choices}", raw)
        )
    if "{signals}" in text:
        text = text.replace("{signals}", ", ".join(str(item) for item in signals))
    if "{signal_text}" in text:
        text = text.replace("{signal_text}", str(signal_text or ""))
    return text


def tsenv_instruction_human_with_description(question_text: str, description: Optional[str]) -> str:
    question_text = str(question_text or "").strip()
    description = None if description is None else str(description).strip()
    if not description:
        return question_text
    return "\n".join([description, "", question_text])

def tsenv_instruction_with_description(question_text: str, description: Optional[str]) -> str:
    return tsenv_instruction_human_with_description(question_text, description)

def tsenv_instruction_with_description_and_signals(
    *,
    question_text: str,
    description: Optional[str],
    signals: Sequence[object],
    signal_display_names: dict[str, str] | None = None,
) -> str:
    question_text = str(question_text or "").strip()
    description = None if description is None else str(description).strip()
    signals_block = tsenv_signals_block(signals or (), signal_display_names or {})
    if not description:
        return "\n".join([question_text]).strip() # Removed signal for now. We are embedding into into the description
    if question_text.startswith("In this simulation, exactly one parameter from "):
        question_text = question_text.replace(
            "In this simulation, exactly one parameter from ",
            "Exactly one parameter from ",
            1,
        )
    return "\n".join([description, "", question_text]).strip() # # Removed signal for now. We are embedding into into the description

def tsenv_instruction_human_with_description_and_signals(
    *,
    question_text: str,
    description: Optional[str],
    signals: Sequence[object],
    signal_display_names: dict[str, str] | None = None,
) -> str:
    return tsenv_instruction_with_description_and_signals(
        question_text=question_text,
        description=description,
        signals=signals,
        signal_display_names=signal_display_names,
    )

def tsenv_signals_block(
    signals: Sequence[object],
    signal_display_names: dict[str, str] | None = None,
) -> str:
    resolved_signals = [str(item).strip() for item in (signals or ()) if str(item).strip()]
    signal_display_names = signal_display_names or {}
    lines = ["Each simulation provides the following signals:"]
    for signal in resolved_signals:
        description = str(signal_display_names.get(signal) or "").strip()
        if description:
            lines.append(f"* {signal} — {description}")
        else:
            lines.append(signal)
    return "\n".join(lines)

def few_shot_pairs_explanation_block(
    *,
    few_shot_count: int,
    data_ref: str,
    label_ref: str,
    label_semantics: str = "0 (good)/1 (anomaly)",
    include_time_note: bool = False,
) -> str:
    if few_shot_count <= 0:
        raise ValueError("few_shot_count must be > 0")
    count_words = _count_to_english_words(few_shot_count)
    pair = "pair" if few_shot_count == 1 else "pairs"
    data_sentence = "The data you need to check is available at `dataframe.parquet`."
    time_note = ""
    return "\n".join(
        [   
            f"{data_sentence}{time_note} Additionally "
            f"You have available {count_words} labelled example {pair} of parquet files:",
            f"  - {data_ref}: full time series (signals)",
            f"  - {label_ref}: full time series labels with columns `time` and `label` ({label_semantics})",
            f"Anything else is not necessary nor useful."
        ]
    )

def append_few_shot_pairs_explanation(
    instruction: str,
    *,
    few_shot_count: int,
    data_ref: str,
    label_ref: str,
    label_semantics: str = "0 (good)/1 (anomaly)",
) -> str:
    if not instruction:
        raise ValueError("instruction must be non-empty")
    base = instruction.rstrip()
    block = few_shot_pairs_explanation_block(
        few_shot_count=few_shot_count,
        data_ref=data_ref,
        label_ref=label_ref,
        label_semantics=label_semantics,
    )
    return f"{base}\n\n{block}\n"


def agent_prompt_classification(
    instruction_human_format: str,
    multiple_choices: Sequence[object],
) -> str:
    return join_message_lines(
        [
            str(instruction_human_format or "").strip(),
            "",
            *instruction_lines("classification", multiple_choices=multiple_choices),
        ]
    )


def agent_prompt_anomaly(
    instruction_human_format: str,
    *,
    extra_lines: Sequence[str] = (),
) -> str:
    message_lines: List[str] = [str(instruction_human_format or "").strip(), *[str(line) for line in extra_lines]]
    message_lines.extend(instruction_lines("anomaly"))
    return join_message_lines(message_lines)

def shot_context_line(*, benchmark: str, shot_key: str, few_shot_per_class: int) -> str:
    shot_level = (shot_key or "").strip().lower()
    is_tsenv = "tsenv" in (benchmark or "").lower()
    time_note = ""
    data_location = (
        "The data you need to check is available in `test_samples/*.parquet`."
        if is_tsenv
        else "The data you need to check is available at `dataframe.parquet`."
    )
    
    if shot_level == "zero_shot":
        if is_tsenv:
            return "The data to be classified is stored in the test_samples folder."
        return f"{data_location}{time_note} Anything else is not necessary nor useful."
    if shot_level in ["one_shot", "few_shot", "many_shots", "many_shot"]:
        count_words = _count_to_english_words(few_shot_per_class)
        noun = "example" if few_shot_per_class == 1 else "examples"
        if is_tsenv:
            text = (
                "The labeled examples are stored in the train_samples folder, with "
                f"{count_words} labeled examples per class.\n"
                "The label is included in the file name. "
                "The data to be classified is stored in the test_samples folder."
            )
        else:
            text = (
                f"{data_location}{time_note} Additionally, you have {count_words} {noun} per class. "
                "The class label is encoded in the filename. Anything else is not necessary nor useful."
            )

    return text


def shot_level_from_benchmark_type(benchmark_type: str) -> Optional[str]:
    lowered = (benchmark_type or "").strip().lower()
    for shot in ("zero_shot", "one_shot", "few_shot"):
        if shot in lowered:
            return shot
    return None


def ucr_instruction_human_format(
    *,
    description: str = "",
    use_description: bool = False,
    benchmark_type: str = "",
    few_shot_per_class: int,
) -> str:
    lines = ["Determine which class this time series belongs to."]
    if use_description and description:
        lines.extend([description, ""])
    shot_context = shot_context_line(
        benchmark=benchmark_type,
        shot_key=shot_level_from_benchmark_type(benchmark_type) or "",
        few_shot_per_class=few_shot_per_class,
    )
    if shot_context:
        lines.append(shot_context)

    return join_message_lines(lines)


def ucr_instruction_agent_format(
    multiple_choices: Sequence[object],
    *,
    description: str = "",
    use_description: bool = False,
    benchmark_type: str = "",
    few_shot_per_class: int,
) -> str:
    shot_line = shot_context_line(
        benchmark=benchmark_type,
        shot_key=shot_level_from_benchmark_type(benchmark_type) or "",
        few_shot_per_class=few_shot_per_class,
    )
    if use_description:
        assert description
    instruction_choices: Sequence[object] = multiple_choices
    if not (use_description and description):
        instruction_choices = [str(idx + 1) for idx, _ in enumerate(multiple_choices)]
    if use_description:
        return join_message_lines(
            [
                description,
                "",
                "Classify the time series into one of the provided classes.",
                "",
                shot_line,
                "",
                *instruction_lines("classification", multiple_choices=instruction_choices),
            ]
        )
    return join_message_lines(
        [   
            "Classify the time series into one of the provided classes.",
            "",
            shot_line,
            "",
            *instruction_lines("classification", multiple_choices=instruction_choices),
        ]
    )


def tsb_ad_instruction_human_format(
    *,
    description: str = "",
    use_description: bool = False,
    is_with_hint: bool = False,
    extra_hint: Optional[dict[str, float]] = None,
) -> str:
    prefix = ""
    if use_description and description:
        prefix = f"Dataset description:\n{description.strip()}\n\n"
    description_lead = "Based on the dataset description, " if use_description and description else ""
    if is_with_hint:
        if extra_hint is None:
            raise ValueError("extra_hint is required when is_with_hint is True")
        file_count = int(extra_hint["tuning_file_count"])
        avg_count = round(extra_hint["tuning_avg_anomaly_count"], 2)
        avg_len = round(extra_hint["tuning_avg_anomaly_len"], 2)
        initial_message = (
            f"{description_lead}There is at least one anomaly, your task is to identify it. "
            f"Note that on average, on other {file_count} related time series "
            f"there were {avg_count} anomalies with an average segment length of {avg_len}."
        )
    else:
        initial_message = (
            f"{description_lead}Identify all the anomalies (zero, one or more) in this time series. "
        )

    message = initial_message + (
        "For each anomaly return a list with start_idx, end_idx, and confidence "
        "(1 max confidence that is an anomaly)"
    )

    return f"{prefix}{message}"


def tsb_ad_append_complete_hint(instruction: str, anomaly_start: int) -> str:
    if not instruction:
        raise ValueError("instruction must be non-empty")
    base = instruction.rstrip()
    return (
        f"{base} Baseline data/label files are available in train_paths; "
        f"it contains a known anomaly that starts at index {anomaly_start} "
    )


def tsb_ad_append_train_paths_explanation(instruction: str, *, few_shot_count: int) -> str:
    if not instruction:
        raise ValueError("instruction must be non-empty")

    block = few_shot_pairs_explanation_block(
        few_shot_count=few_shot_count,
        data_ref="`data_<k>.parquet`",
        label_ref="`label_<k>.parquet`",
        label_semantics="0 (good)/1 (anomaly)",
    )

    base = instruction.rstrip()
    save_marker = "Save `/app/results.json` with"
    if save_marker not in base:
        return f"{base}\n\n{block}\n"

    lines = base.splitlines()
    insert_at: Optional[int] = None
    for idx, line in enumerate(lines):
        if line.strip().startswith(save_marker):
            insert_at = idx
            break
    if insert_at is None:
        return f"{base}\n\n{block}\n"

    block_lines = block.splitlines()
    prefix = lines[:insert_at]
    suffix = lines[insert_at:]

    needs_leading_blank = not (prefix and not prefix[-1].strip())
    needs_trailing_blank = not (suffix and not suffix[0].strip())
    injected: List[str] = []
    if needs_leading_blank:
        injected.append("")
    injected.extend(block_lines)
    if needs_trailing_blank:
        injected.append("")

    return join_message_lines([*prefix, *injected, *suffix])


def tsb_ad_instruction_agent_format(
    *,
    final_answer_key: str,
    description: str = "",
    use_description: bool = False,
    is_with_hint: bool = False,
    extra_hint: Optional[dict[str, float]] = None,
) -> str:
    assert not use_description, (
        "tsb_ad_instruction_agent_format: descriptions are not supported; "
        "expected use_description=False"
    )
    assert not str(description or "").strip(), (
        "tsb_ad_instruction_agent_format: descriptions are not supported; "
        "expected description to be empty"
    )
    description_lead = "Based on the dataset description, " if use_description and description else ""
    if is_with_hint:
        if extra_hint is None:
            raise ValueError("extra_hint is required when is_with_hint is True")
        file_count = int(extra_hint["tuning_file_count"])
        avg_count = round(extra_hint["tuning_avg_anomaly_count"], 2)
        avg_len = round(extra_hint["tuning_avg_anomaly_len"], 2)
        initial_message = (
            f"{description_lead}There is at least one anomaly, your task is to identify it. "
            f"Note that on average, on other {file_count} related time series "
            f"there were {avg_count} anomalies with an average segment length of {avg_len}. "
        )
    else:
        initial_message = "Identify anomalous contiguous regions in the provided time series."

    message_lines: List[str] = []
    if use_description and description:
        message_lines.extend(["Dataset description:", description.strip(), ""])
    message_lines.append(initial_message)
    results_payload = json.dumps({final_answer_key: "<windows>"}).replace('"<windows>"', "<windows>")
    message_lines += [
        "",
        f"Save `/app/results.json` with {results_payload}.",
        "Where: `<windows>` is a list of windows `[[start_idx, end_idx, confidence], ...]` where:",
        "- Indices are 0-based, and `end_idx` is inclusive.",
        "- `confidence` is a float in [0, 1] (1 = highest confidence anomaly).",
        "- A point anomaly should be represented as `[i, i, confidence]`.",
        *instruction_lines("anomaly"),
    ]

    return join_message_lines(message_lines)
