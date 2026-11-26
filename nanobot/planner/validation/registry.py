from .base import Checker
from .checkers.file_system import FileContainsChecker,FileExistsChecker,DirectoryExistsChecker
from .checkers.code import JsonValidChecker,PythonSyntaxChecker
class CheckerRegistry:
    def __init__(self):
        self._checkers: dict[str, Checker] = {}

    def register(self,criterion_type:str,checker:Checker)->None:
        if self.has_checker(criterion_type):
            raise ValueError("该类型Checker已存在")
        self._checkers[criterion_type]=checker

    def get_checker(self,criterion_type:str)->Checker:
        if not self.has_checker(criterion_type):
            raise ValueError("该类型Checker不存在")
        return self._checkers[criterion_type]

    def has_checker(self,criterion_type:str)->bool:
        return criterion_type in self._checkers

def create_default_registry() -> CheckerRegistry:
    cr = CheckerRegistry()
    cr.register("file_exists", FileExistsChecker())
    cr.register("directory_exists", DirectoryExistsChecker())
    cr.register("file_contains", FileContainsChecker())
    cr.register("python_syntax", PythonSyntaxChecker())
    cr.register("json_valid", JsonValidChecker())
    return cr


    