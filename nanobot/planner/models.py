
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"


class ValidationStatus(str, Enum):
    NOT_CHECKED = "not_checked"
    PASSED = "passed"
    FAILED = "failed"


@dataclass
class CriterionResult:
    criterion_id: str
    passed: bool
    feedback: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    passed: bool
    feedback: list[str] = field(default_factory=list)
    criterion_results: list[CriterionResult] = field(default_factory=list)


@dataclass
class AcceptanceCriterion:
    criterion_id: str
    type: str
    description: str
    parameters: dict[str, Any]


@dataclass
class Task:
    task_id: str
    description: str
    dependencies: list[str] = field(default_factory=list)
    dependents: list[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    acceptance_criteria: list[AcceptanceCriterion] = field(default_factory=list)
    validation_result: ValidationResult | None = None
    validation_status: ValidationStatus = ValidationStatus.NOT_CHECKED
