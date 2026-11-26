
import sys
import types
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.planner.models import TaskStatus, ValidationStatus


def make_module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def load_workflow_module():
    planner_module = make_module(
        "nanobot.planner.planner",
        Planner=MagicMock,
    )
    executor_module = make_module(
        "nanobot.planner.executor",
        DAGExecutor=MagicMock,
    )
    adapter_module = make_module(
        "nanobot.planner.adapter",
        SubagentTaskExecutor=MagicMock,
    )
    validation_package = make_module("nanobot.planner.validation")
    validation_package.__path__ = []
    validator_module = make_module(
        "nanobot.planner.validation.validator",
        TaskResultValidator=MagicMock,
    )
    registry_module = make_module(
        "nanobot.planner.validation.registry",
        create_default_registry=MagicMock,
    )
    workspace_module = make_module(
        "nanobot.security.workspace_access",
        current_workspace_scope=MagicMock(return_value=None),
    )
    fake_modules = {
        "nanobot.planner.planner": planner_module,
        "nanobot.planner.executor": executor_module,
        "nanobot.planner.adapter": adapter_module,
        "nanobot.planner.validation": validation_package,
        "nanobot.planner.validation.validator": validator_module,
        "nanobot.planner.validation.registry": registry_module,
        "nanobot.security.workspace_access": workspace_module,
    }
    with patch.dict(sys.modules, fake_modules):
        sys.modules.pop("nanobot.planner.workflow", None)
        module = importlib.import_module("nanobot.planner.workflow")
    return module


workflow_module = load_workflow_module()
DAGWorkflow = workflow_module.DAGWorkflow


def make_task(
    task_id,
    status,
    dependents=None,
    result=None,
    error=None,
    validation_status=ValidationStatus.NOT_CHECKED,
    validation_result=None,
):
    return SimpleNamespace(
        task_id=task_id,
        description=f"任务 {task_id}",
        status=status,
        dependents=dependents or [],
        result=result,
        error=error,
        validation_status=validation_status,
        validation_result=validation_result,
    )


def make_graph(tasks):
    return SimpleNamespace(
        tasks={task.task_id: task for task in tasks}
    )


class TestDAGWorkflow(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.manager = SimpleNamespace(
            workspace=Path("/tmp/workspace")
        )
        self.runtime = MagicMock()
        self.workflow = DAGWorkflow(self.manager, 3)

    async def run_workflow(self, graph, paras=None, workspace_scope=None):
        with (
            patch.object(workflow_module, "Planner") as planner_cls,
            patch.object(workflow_module, "SubagentTaskExecutor") as adapter_cls,
            patch.object(workflow_module, "DAGExecutor") as executor_cls,
            patch.object(workflow_module, "TaskResultValidator") as validator_cls,
            patch.object(workflow_module, "create_default_registry") as registry_factory,
            patch.object(
                workflow_module,
                "current_workspace_scope",
                return_value=workspace_scope,
            ),
        ):
            planner_cls.return_value.plan = AsyncMock(return_value=graph)
            executor_cls.return_value.execute = AsyncMock()
            result = await self.workflow.run(paras or {}, self.runtime)
        return result, adapter_cls, executor_cls, validator_cls, registry_factory

    async def test_all_tasks_completed(self):
        tasks = [
            make_task(
                "T1",
                TaskStatus.COMPLETED,
                dependents=["T2"],
                result="结果1",
            ),
            make_task(
                "T2",
                TaskStatus.COMPLETED,
                result="结果2",
            ),
        ]
        result, _, _, _, _ = await self.run_workflow(make_graph(tasks))
        self.assertTrue(result["success"])
        self.assertEqual(len(result["process_results"]), 1)
        self.assertEqual(len(result["terminal_results"]), 1)
        self.assertEqual(result["incomplete_tasks"], [])

    async def test_partial_failure_preserves_validation_feedback(self):
        validation_result = SimpleNamespace(
            passed=False,
            feedback=["缺少必要的 CRUD 接口"],
            criterion_results=[],
        )
        tasks = [
            make_task(
                "T1",
                TaskStatus.COMPLETED,
                result="已完成",
            ),
            make_task(
                "T2",
                TaskStatus.FAILED,
                validation_status=ValidationStatus.FAILED,
                validation_result=validation_result,
            ),
        ]
        result, _, _, _, _ = await self.run_workflow(make_graph(tasks))
        self.assertFalse(result["success"])
        self.assertEqual(len(result["terminal_results"]), 1)
        self.assertEqual(len(result["incomplete_tasks"]), 1)
        failed_task = result["incomplete_tasks"][0]
        self.assertEqual(failed_task["task_id"], "T2")
        self.assertEqual(
            failed_task["feedback"],
            ["缺少必要的 CRUD 接口"],
        )

    async def test_all_tasks_failed_or_blocked(self):
        tasks = [
            make_task(
                "T1",
                TaskStatus.FAILED,
                error="执行异常",
            ),
            make_task(
                "T2",
                TaskStatus.BLOCKED,
            ),
        ]
        result, _, _, _, _ = await self.run_workflow(make_graph(tasks))
        self.assertFalse(result["success"])
        self.assertEqual(result["terminal_results"], [])
        self.assertEqual(len(result["incomplete_tasks"]), 2)
        self.assertEqual(
            result["incomplete_tasks"][0]["error"],
            "执行异常",
        )
        self.assertEqual(
            result["incomplete_tasks"][1]["fail_reason"],
            "因前置依赖失败而未执行",
        )

    async def test_cancelled_task_is_incomplete(self):
        tasks = [
            make_task(
                "T1",
                TaskStatus.CANCELLED,
            )
        ]
        result, _, _, _, _ = await self.run_workflow(make_graph(tasks))
        self.assertFalse(result["success"])
        self.assertEqual(len(result["incomplete_tasks"]), 1)
        self.assertEqual(
            result["incomplete_tasks"][0]["fail_reason"],
            "任务已取消",
        )

    async def test_empty_graph_is_not_success(self):
        result, _, _, _, _ = await self.run_workflow(make_graph([]))
        self.assertFalse(result["success"])
        self.assertEqual(result["terminal_results"], [])
        self.assertEqual(result["process_results"], [])
        self.assertEqual(result["incomplete_tasks"], [])

    async def test_disabled_tools_and_workspace_are_passed(self):
        tasks = [
            make_task(
                "T1",
                TaskStatus.COMPLETED,
                result="完成",
            )
        ]
        graph = make_graph(tasks)
        disabled_tools = ["web_search"]
        scope = SimpleNamespace(
            project_path=Path("/tmp/project")
        )
        result, adapter_cls, executor_cls, validator_cls, registry_factory = (
            await self.run_workflow(
                graph,
                paras={"disabled_tools": disabled_tools},
                workspace_scope=scope,
            )
        )
        adapter_cls.assert_called_once_with(
            self.manager,
            graph,
            disabled_tools,
        )
        executor_kwargs = executor_cls.call_args.kwargs
        self.assertEqual(
            executor_kwargs["workspace_root"],
            scope.project_path,
        )
        self.assertEqual(
            executor_kwargs["max_concurrency"],
            3,
        )
        registry_factory.assert_called_once()
        validator_cls.assert_called_once_with(
            registry_factory.return_value
        )
        self.assertTrue(result["success"])

    async def test_workspace_falls_back_to_manager_workspace(self):
        tasks = [
            make_task(
                "T1",
                TaskStatus.COMPLETED,
            )
        ]
        graph = make_graph(tasks)
        _, _, executor_cls, _, _ = await self.run_workflow(
            graph,
            workspace_scope=None,
        )
        self.assertEqual(
            executor_cls.call_args.kwargs["workspace_root"],
            self.manager.workspace,
        )
