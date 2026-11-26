
from abc import ABC, abstractmethod
from pathlib import Path
from ..models import AcceptanceCriterion, CriterionResult


class Checker(ABC):
    @abstractmethod
    def validate(
        self,
        criterion: AcceptanceCriterion,
        workspace_root: Path,
    ) -> CriterionResult:
        pass
