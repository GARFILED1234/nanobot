import json
from .graph import TaskGraph
from .models import Task
from nanobot.utils.llm_runtime import LLMRuntime
from nanobot.utils.prompt_templates import render_template

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
            t["task_id"]=task_id
            t["description"]=des
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
            task=Task(t["task_id"].strip(),t["description"].strip())
            graph.add_task(task)
        for t in tasks:
            t_id=t["task_id"].strip()
            for d in t["dependencies"]:
                graph.add_dependency(t_id,d.strip())
        graph.validate()
        return graph

