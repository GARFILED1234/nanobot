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
            raise ValueError("输入格式不正确")
        if "goal" not in paras:
            raise ValueError("goal不存在")
        goal=paras["goal"]
        if not isinstance(goal,str) or not (goal:=goal.strip()):
            raise ValueError("输入有误")
        paras["goal"]=goal
        p=paras.get("parameters")
        if p is not None:
            if not isinstance(p,dict):
                raise ValueError("输入格式不正确")
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
            raise ValueError("llm返回结果为空")
        tasks=self._parse_response(content)
        graph=self._build_task_graph(tasks)
        return graph

    def _build_system_prompt(self):
        return render_template("agent/planner.md")

    def _parse_response(self,response):
        if not isinstance(response, str):
            raise ValueError("llm返回格式有误")
        try:
            lines = response.splitlines()
            if (
                len(lines) >= 3
                and lines[0].strip().startswith("```")
                and lines[-1].strip() == "```"
            ):
                response = "\n".join(lines[1:-1]).strip()
            r_dict=json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("llm返回格式有误")
        if not isinstance(r_dict, dict):
            raise ValueError("llm返回格式有误")
        tasks=r_dict.get("tasks")
        if not isinstance(tasks, list):
            raise ValueError("llm返回格式有误")
        if not tasks:
            raise ValueError("无任务")
        checked=set()
        for t in tasks:
            if not isinstance(t,dict):
                raise ValueError("llm返回格式有误")
            task_id=t.get("task_id")
            deps=t.get("dependencies")
            des=t.get("description")
            if not isinstance(task_id,str):
                raise ValueError("llm返回格式有误")
            if not isinstance(des,str):
                raise ValueError("llm返回格式有误")
            if not (task_id := task_id.strip()) or task_id in checked or not (des := des.strip()):
                raise ValueError("llm返回数据有误")
            if not isinstance(deps,list):
                raise ValueError("llm返回格式有误")
            acceptance_criteria=t.get("acceptance_criteria")
            if acceptance_criteria is None or not isinstance(acceptance_criteria,list):
                raise ValueError("llm返回格式有误")
            if not acceptance_criteria:
                raise ValueError("任务验收标准不能为空")
            new_acceptance=[]
            a_checked=set() 
            for a in acceptance_criteria:
                if not isinstance(a,dict):
                    raise ValueError("llm返回格式有误")
                criterion_id=a.get("criterion_id")
                criterion_type=a.get("type")
                description=a.get("description")
                parameters=a.get("parameters")
                if not isinstance(criterion_id,str) or not isinstance(criterion_type,str) or not isinstance(description,str) or not isinstance(parameters,dict):
                    raise ValueError("llm返回格式有误")
                criterion_id=criterion_id.strip()
                if criterion_id in a_checked:
                    raise ValueError("llm返回数据有误")
                a_checked.add(criterion_id)
                criterion_type=criterion_type.strip()
                description=description.strip()
                if not criterion_id or not criterion_type or criterion_type not in SUPPORTED_CRITERION_TYPES or not description:
                    raise ValueError("llm返回格式有误")
                new_parameters={}
                for p,v in parameters.items():
                    if not isinstance(p,str):
                        raise ValueError("llm返回格式有误")
                    p=p.strip()
                    if not p:
                        raise ValueError("llm返回格式有误")
                    if p in new_parameters:
                        raise ValueError("验收标准参数名称重复")
                    new_parameters[p]=v
                actual = set(new_parameters.keys())
                if actual !=CRITERION_REQUIRED_PARAMETERS.get(criterion_type):
                    raise ValueError("验收标准参数不符合要求")
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
                    raise ValueError("llm返回格式有误")
                if d in deps_checked:
                    raise ValueError("llm返回数据有误")
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

