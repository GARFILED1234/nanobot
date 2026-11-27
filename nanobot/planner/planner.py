
import json
from .graph import TaskGraph
from .models import Task,AcceptanceCriterion
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template

SUPPORTED_CRITERION_TYPES = {
    "file_exists",
    "directory_exists",
    "file_contains",
    "python_syntax",
    "json_valid",
}

CRITERION_REQUIRED_PARAMETERS = {
    "file_exists": {"path"},
    "directory_exists": {"path"},
    "file_contains": {"path", "text"},
    "python_syntax": {"path"},
    "json_valid": {"path"},
}

class Planner:
    def __init__(self,runtime:LLMRuntime):
        self.runtime=runtime
    
    async def plan(self,paras):
        if not isinstance(paras,dict):
            raise ValueError("Planner输入错误：参数必须是字典")
        if "goal" not in paras:
            raise ValueError("Planner输入错误：缺少goal字段")
        goal=paras["goal"]
        if not isinstance(goal,str) or not (goal:=goal.strip()):
            raise ValueError("Planner输入错误：goal必须是非空字符串")
        paras["goal"]=goal
        p=paras.get("parameters")
        if p is not None:
            if not isinstance(p,dict):
                raise ValueError("Planner输入错误：parameters必须是字典")
        system_prompt=self._build_system_prompt()
        para = json.dumps(paras, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": para},
        ]
        response = await self.runtime.provider.chat_with_retry(
            messages=messages,
            model=self.runtime.model,
            max_tokens=self.runtime.generation.max_tokens,
            temperature=self.runtime.generation.temperature,
            reasoning_effort=self.runtime.generation.reasoning_effort,
        )
        content = response.content
        print(f"[Planner Debug] response.content type: {type(content)}")
        print(f"[Planner Debug] response.content: {content!r}", flush=True)
        if not isinstance(content,str) or not (content := content.strip()):
            raise ValueError("Planner规划失败：LLM返回内容为空或不是字符串")
        
        try:
            tasks = self._parse_response(content)
        except ValueError as err:
            repair_messages = messages + [
                {"role": "assistant", "content": content},
                {
                    "role": "user",
                    "content": (
                        "你刚才生成的 Planner JSON 未通过格式校验。\n"
                        f"具体错误：{err}\n"
                        "请根据错误修复之前的 JSON。\n"
                        "要求：\n"
                        "1. 只修复不符合要求的内容，尽量保留原有任务、描述和依赖关系。\n"
                        "2. 必须遵守系统提示词中的所有规划规则。\n"
                        "3. 必须输出完整 JSON，不能只输出修改片段。\n"
                        "4. 只输出 JSON，不要添加 Markdown 代码块或解释。"
                    )
                }
            ]
            repair_response = await self.runtime.provider.chat_with_retry(
                messages=repair_messages,
                model=self.runtime.model,
                max_tokens=self.runtime.generation.max_tokens,
                temperature=self.runtime.generation.temperature,
                reasoning_effort=self.runtime.generation.reasoning_effort,
            )
            repaired_content = repair_response.content
            if not isinstance(repaired_content, str) or not (repaired_content := repaired_content.strip()):
                raise ValueError(f"Planner自动修复失败：LLM返回内容为空或不是字符串，首次错误：{err}")
            try:
                tasks = self._parse_response(repaired_content)
            except ValueError as repair_err:
                raise ValueError(
                    f"Planner自动修复失败：修复后的内容仍未通过校验。首次错误：{err}；修复错误：{repair_err}"
                ) from repair_err
        graph=self._build_task_graph(tasks)
        return graph

    def _build_system_prompt(self):
        return render_template("agent/planner.md")

    def _parse_response(self,response):
        if not isinstance(response, str):
            raise ValueError("Planner解析失败：LLM返回内容必须是字符串")
        try:
            lines = response.splitlines()
            if (
                len(lines) >= 3
                and lines[0].strip().startswith("```")
                and lines[-1].strip() == "```"
            ):
                response = "\n".join(lines[1:-1]).strip()
            r_dict=json.loads(response)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"Planner JSON解析失败：{err.msg}，第{err.lineno}行第{err.colno}列"
            ) from err
        if not isinstance(r_dict, dict):
            raise ValueError("Planner JSON结构错误：最外层必须是对象")
        tasks=r_dict.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("Planner JSON结构错误：tasks必须是数组")
        if not tasks:
            raise ValueError("Planner规划失败：tasks数组不能为空")
        checked=set()
        for index,t in enumerate(tasks, start=1):
            if not isinstance(t,dict):
                raise ValueError(f"Planner任务错误：第{index}个任务必须是对象")
            task_id=t.get("task_id")
            deps=t.get("dependencies")
            des=t.get("description")
            if not isinstance(task_id,str):
                raise ValueError(f"Planner任务错误：第{index}个任务的task_id必须是字符串")
            if not isinstance(des,str):
                raise ValueError(f"Planner任务错误：第{index}个任务的description必须是字符串")
            if not (task_id := task_id.strip()):
                raise ValueError(f"Planner任务错误：第{index}个任务的task_id不能为空")
            if task_id in checked:
                raise ValueError(f"Planner任务错误：任务ID重复，task_id={task_id}")
            if not (des := des.strip()):
                raise ValueError(f"Planner任务错误：{task_id}的description不能为空")
            if not isinstance(deps,list):
                raise ValueError(f"Planner任务错误：{task_id}的dependencies必须是数组")
            acceptance_criteria=t.get("acceptance_criteria")
            if acceptance_criteria is None or not isinstance(acceptance_criteria,list):
                raise ValueError(f"Planner任务错误：{task_id}的acceptance_criteria必须是数组")
            if not acceptance_criteria:
                raise ValueError(f"Planner任务错误：{task_id}的验收标准不能为空")
            new_acceptance=[]
            a_checked=set() 
            for criterion_index,a in enumerate(acceptance_criteria, start=1):
                if not isinstance(a,dict):
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}的第{criterion_index}条验收标准必须是对象"
                    )
                criterion_id=a.get("criterion_id")
                criterion_type=a.get("type")
                description=a.get("description")
                parameters=a.get("parameters")
                if not isinstance(criterion_id,str):
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}的第{criterion_index}条criterion_id必须是字符串"
                    )
                if not isinstance(criterion_type,str):
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}的第{criterion_index}条type必须是字符串"
                    )
                if not isinstance(description,str):
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}的第{criterion_index}条description必须是字符串"
                    )
                if not isinstance(parameters,dict):
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}的第{criterion_index}条parameters必须是对象"
                    )
                criterion_id=criterion_id.strip()
                if not criterion_id:
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}的第{criterion_index}条criterion_id不能为空"
                    )
                if criterion_id in a_checked:
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}/{criterion_id}重复"
                    )
                a_checked.add(criterion_id)
                criterion_type=criterion_type.strip()
                description=description.strip()
                if not criterion_type:
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}/{criterion_id}的type不能为空"
                    )
                if criterion_type not in SUPPORTED_CRITERION_TYPES:
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}/{criterion_id}不支持验收类型{criterion_type}"
                    )
                if not description:
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}/{criterion_id}的description不能为空"
                    )
                new_parameters={}
                for p,v in parameters.items():
                    if not isinstance(p,str):
                        raise ValueError(
                            f"Planner验收标准错误：{task_id}/{criterion_id}的参数名称必须是字符串"
                        )
                    p=p.strip()
                    if not p:
                        raise ValueError(
                            f"Planner验收标准错误：{task_id}/{criterion_id}存在空参数名称"
                        )
                    if p in new_parameters:
                        raise ValueError(
                            f"Planner验收标准错误：{task_id}/{criterion_id}参数名称规范化后重复，参数名={p}"
                        )
                    new_parameters[p]=v
                actual = set(new_parameters.keys())
                required = CRITERION_REQUIRED_PARAMETERS.get(criterion_type)
                if actual != required:
                    missing = required - actual
                    extra = actual - required
                    details = []
                    if missing:
                        details.append(f"缺少参数：{', '.join(sorted(missing))}")
                    if extra:
                        details.append(f"不允许的参数：{', '.join(sorted(extra))}")
                    raise ValueError(
                        f"Planner验收标准错误：{task_id}/{criterion_id}的parameters不符合{criterion_type}要求；"
                        + "；".join(details)
                    )
                a1=AcceptanceCriterion(criterion_id=criterion_id,type=criterion_type,description=description,parameters=new_parameters)
                new_acceptance.append(a1)
            t["task_id"]=task_id
            t["description"]=des
            t["acceptance_criteria"]=new_acceptance
            checked.add(task_id)
            deps_checked=set()
            new_deps=[]
            for d in deps:
                if not isinstance(d,str) or not (d:=d.strip()):
                    raise ValueError(f"Planner任务错误：{task_id}的dependencies包含非字符串或空依赖ID")
                if d in deps_checked:
                    raise ValueError(f"Planner任务错误：{task_id}的dependencies包含重复依赖，依赖ID={d}")
                new_deps.append(d)
                deps_checked.add(d)
            t["dependencies"]=new_deps
        return tasks


    def _build_task_graph(self,tasks):
        graph = TaskGraph()
        for t in tasks:
            task=Task(task_id=t["task_id"].strip(),description=t["description"].strip(),acceptance_criteria=t["acceptance_criteria"])
            graph.add_task(task)
        for t in tasks:
            t_id=t["task_id"].strip()
            for d in t["dependencies"]:
                graph.add_dependency(t_id,d.strip())
        graph.validate()
        return graph
