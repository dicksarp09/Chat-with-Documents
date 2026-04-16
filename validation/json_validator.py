import logging
from typing import Type, Optional, Any, Dict, List
from pydantic import BaseModel, ValidationError, field_validator
from pydantic_settings import BaseSettings

from schemas.output_schema import (
    DocumentAnalysisOutput,
    QueryOutput,
    ActionItem,
    RiskItem,
    ObligationItem,
    KeyPoint,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationResult:
    def __init__(
        self,
        is_valid: bool,
        data: Optional[BaseModel] = None,
        errors: Optional[List[str]] = None,
        raw_data: Optional[Dict] = None,
    ):
        self.is_valid = is_valid
        self.data = data
        self.errors = errors or []
        self.raw_data = raw_data

    def to_dict(self) -> Dict[str, Any]:
        if self.data:
            return self.data.model_dump()
        return {"valid": self.is_valid, "errors": self.errors}


class JsonValidator:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    def validate_document_analysis(self, data: Dict[str, Any]) -> ValidationResult:
        try:
            validated = DocumentAnalysisOutput(**data)
            return ValidationResult(is_valid=True, data=validated, raw_data=data)
        except ValidationError as e:
            errors = [str(err) for err in e.errors()]
            logger.warning(f"Validation errors: {errors}")
            return ValidationResult(is_valid=False, errors=errors, raw_data=data)
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return ValidationResult(is_valid=False, errors=[str(e)], raw_data=data)

    def validate_query_output(self, data: Dict[str, Any]) -> ValidationResult:
        try:
            validated = QueryOutput(**data)
            return ValidationResult(is_valid=True, data=validated, raw_data=data)
        except ValidationError as e:
            errors = [str(err) for err in e.errors()]
            logger.warning(f"Validation errors: {errors}")
            return ValidationResult(is_valid=False, errors=errors, raw_data=data)
        except Exception as e:
            logger.error(f"Unexpected validation error: {e}")
            return ValidationResult(is_valid=False, errors=[str(e)], raw_data=data)

    def validate_with_fallback(
        self, data: Dict[str, Any], output_type: str = "document_analysis"
    ) -> ValidationResult:
        if output_type == "document_analysis":
            return self.validate_document_analysis(data)
        elif output_type == "query":
            return self.validate_query_output(data)
        else:
            return ValidationResult(is_valid=True, raw_data=data)

    def fix_common_errors(
        self, data: Dict[str, Any], output_type: str
    ) -> Dict[str, Any]:
        fixed = data.copy()

        if "actions" in fixed:
            for action in fixed["actions"]:
                if isinstance(action, dict):
                    if "priority" in action:
                        priority = str(action["priority"]).lower()
                        if priority not in ["high", "medium", "low"]:
                            action["priority"] = "medium"

                    if "evidence" not in action:
                        action["evidence"] = []
                    elif not isinstance(action["evidence"], list):
                        action["evidence"] = [str(action["evidence"])]

                if isinstance(action, str):
                    fixed["actions"] = [
                        a for a in fixed["actions"] if not isinstance(a, str)
                    ]
                    break

        if "risks" in fixed:
            for risk in fixed["risks"]:
                if isinstance(risk, dict):
                    if "severity" in risk:
                        severity = str(risk["severity"]).lower()
                        if severity not in ["high", "medium", "low"]:
                            risk["severity"] = "medium"

                    if "evidence" not in risk:
                        risk["evidence"] = []
                    elif not isinstance(risk["evidence"], list):
                        risk["evidence"] = [str(risk["evidence"])]

        if "obligations" in fixed:
            for ob in fixed["obligations"]:
                if isinstance(ob, dict):
                    if "evidence" not in ob:
                        ob["evidence"] = []
                    elif not isinstance(ob["evidence"], list):
                        ob["evidence"] = [str(ob["evidence"])]

        if "key_points" in fixed:
            for kp in fixed["key_points"]:
                if isinstance(kp, dict):
                    if "evidence" not in kp:
                        kp["evidence"] = []
                    elif not isinstance(kp["evidence"], list):
                        kp["evidence"] = [str(kp["evidence"])]

        if output_type == "query" and "sources" in fixed:
            if not isinstance(fixed["sources"], list):
                fixed["sources"] = [str(fixed["sources"])]

        return fixed


_validator_instance: Optional[JsonValidator] = None


def get_validator() -> JsonValidator:
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = JsonValidator()
    return _validator_instance


def validate_output(
    data: Dict[str, Any], output_type: str = "document_analysis"
) -> ValidationResult:
    validator = get_validator()
    return validator.validate_with_fallback(data, output_type)
