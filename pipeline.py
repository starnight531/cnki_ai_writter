from llm import LLM
from prompts.outline_prompt import *
from prompts.content_prompt import *
from prompts.cite_judge_prompt import *
import json 
MODEL = "deepseek-chat"   # "Qwen3-14B"
LESS_CITE_NUM = 1
def paser_think(string):
    res = string
    if "</think>" in string:
        think, res = string.split("</think>")
        if len(res) == 0:
            res = think.replace("<think>", "")
    return res

def extract_json(res):
    json_string = ""
    lines = res.split("\n")
    in_json_block = False
    for line in lines:
        if "```json" in line:
            in_json_block = True
            continue
        elif "```" in line and in_json_block:
            in_json_block = False
        if in_json_block:
            json_string += line
    
    # 移除可能的多余空格和换行符，但保留JSON结构
    json_string = json_string.strip()
    
    try:
        json_data = json.loads(json_string)
        return json_data
    except json.JSONDecodeError as e:
        print(f"JSON解析失败: {e}\n原始内容: {res}")
        return None
    
# 模拟 RAG 搜索，返回固定内容
def rag_search(query):
    llm = LLM("https://api.deepseek.com/v1")
    rag_mock = """假装你是一个搜索引擎，编造一些和用户输入<!-用户输入-!>相关的参考文献列表，以json的形式返回，必须有的键是标题、正文。长度在300字内。形式类似下面：
```json
[
{
"标题": "《低空经济：未来城市新引擎》（2023）",
"内容": "定义低空经济为 100-1000 米空域内航空器运营及相关产业的经济形态，提出其对智慧城市建设的战略意义。"
},
{
"标题": "国际航空运输协会《2024 全球低空经济白皮书》",
"内容": "统计显示全球低空经济规模已达 2.3 万亿美元，预测 2030 年将突破 5 万亿美元，无人机物流、城市空中交通成增长主力。"
},
{
"标题": "《低空飞行器关键技术发展报告》（2022）",
"内容": "系统阐述电池技术、导航系统、空管通信等核心技术突破对低空经济发展的支撑作用。"
}
]
```

### 开始输出：
"""
    rag_mock = rag_mock.replace("<!-用户输入-!>", query)
    res = llm.generate_response(rag_mock, MODEL)
    res = paser_think(res)
    res = extract_json(res)
    return res


class OutlineAgent:
    def __init__(self, llm, model):
        self.llm = llm
        self.model = model
        self.level1_title = ""
        self.outline = ""
        self.references = []

    def generate_first_level_title(self, user_input):
        user_input = outline_prompt_v2["level_1_title_prompt"].replace("<!-文章要求-!>", user_input)
        self.level1_title = self.llm.generate_response(user_input, self.model)
        self.level1_title = paser_think(self.level1_title)
        return self.level1_title

    def generate_subtitles(self, user_input_reference = None):
        references = rag_search(self.level1_title)
        if user_input_reference:
            references.append("一定参考用户提供的参考文献：\n"+user_input_reference)

        references = [f"""##### 参考文章序号{i}：\n{ref}""" for i, ref in enumerate(references, start=1)]
        self.references = references
        references_text = "\n\n".join(references)

        user_input = outline_prompt_v2["level_n_title_prompt"].replace("<!-参考文献-!>", references_text).replace("<!-一级大纲-!>", self.level1_title)

        self.outline = self.llm.generate_response(user_input, self.model).split('\n')
        self.outline = paser_think(self.outline)
        if "参考文献序号" in self.outline[-1]:
            # 编写正则表达式，抽取这一行中的数字，可能是多个数字，用逗号分隔，例如：1, 2, 3
            import re
            self.used_reference_nums = re.findall(r'\d+', self.outline[-1])
            self.outline = self.outline[:-1]

        if isinstance(self.outline, list):
            self.outline = "\n".join(self.outline)
        return self.outline


    def revise_first_level_title(self, user_input = None):
        references_text = "\n\n".join(self.references)

        user_input = outline_prompt_v2["revise_level_1_title_prompt"].replace("<!-大纲-!>", self.outline).replace("<!-参考文献-!>", references_text)
        self.outline_diction = self.llm.generate_response(user_input, self.model)
        self.outline_diction = paser_think(self.outline_diction)
        self.outline_diction = extract_json(self.outline_diction)
        return self.outline_diction

    def prune_reference(self):
        used_references = []
        for reference_num in self.used_reference_nums:
            used_references = used_references + self.references[int(reference_num)-1]
        self.references = used_references

class CiteJudgeAgent:
    def __init__(self, llm, model):
        self.llm = llm
        self.model = model
        self.used_references = []
        self.unused_references = []

    def judge_usable_references(self, references, subtitle, start=1):
        # user_input = f"在这些参考文献 {references} 中，有多少可以用于编写小标题 '{subtitle}' 的内容，请直接给出数量"
        references = [f"""##### 参考文章序号{i}：\n{ref}""" for i, ref in enumerate(references, start=start)]
        references_text = "\n\n".join(references)

        user_input = judge_prompt_v1.replace("<!-参考文献-!>", references_text).replace("<!-当前章节标题-!>", subtitle)

        res = self.llm.generate_response(user_input, self.model)
        res = paser_think(res)
        res = extract_json(res)
        for ref_num in res:
            self.used_references.append(references[ref_num-1])

        for idx, ref in references:
            if idx+1 not in res:
                self.unused_references.append(references[idx])

        return res, self.used_references

        
# class ContentAgent:
#     def __init__(self, llm, model, outline_diction):
#         self.llm = llm
#         self.model = model
#         self.outline_diction = outline_diction
#         self.multi_round_session = []   # 可以考虑pick llm，删除user，作为上文

#     def generate_content(self, subtitle, previous_content, references):
        
#         return self.llm.generate_response(user_input, self.model)

class ContentAgent:
    def __init__(self, llm, model, article_requirements, outline_dict, references, used_reference_nums, content_system_prompt, content_round_prompt):
        self.llm = llm
        self.model = model
        self.article_requirements = article_requirements
        self.outline_dict = outline_dict  # 大纲字典树
        self.references = references
        self.used_reference_nums = used_reference_nums

        self.multi_round_session = []
        self.generated_contents = {}  # 存储已生成的内容 {section_title: content}
        self.content_system_prompt = content_system_prompt
        self.content_round_prompt = content_round_prompt
        self.initialize_session()

        self.citejudgeAgent = CiteJudgeAgent(self.llm, self.model)

    def prune_reference(self):
        used_references = []
        for reference_num in self.used_reference_nums:
            used_references = used_references + self.references[int(reference_num)-1]
        self.references = used_references

    def _outline_to_string(self, outline, indent: int = 0) -> str:
        """将大纲字典转换为字符串表示"""
        result = []
        for title, children in outline.items():
            result.append("    " * indent + title)
            if children:  # 如果有子标题
                result.append(self._outline_to_string(children, indent + 1))
        return "\n".join(result)
    
    def initialize_session(self, add_title_and_outline = False):
        """初始化对话会话"""
        system_prompt = self.content_system_prompt.replace("<!-文章要求-!>", self.article_requirements).replace("<!-大纲-!>", self._outline_to_string(self.outline_dict))
        self.multi_round_session = [
            {"role": "system", "content": system_prompt}
        ]
        if add_title_and_outline:
            self.multi_round_session.append({"role": "user", "content": "### 请你输出标题和大纲作为文章的第一，二页"})
            add_title_and_outline = self.llm.generate_response(self.multi_round_session, self.model)
            self.multi_round_session.append({"role": "assistant", "content": add_title_and_outline})
        
    def judge_and_fetch_usable_references(self, subtitle):
        # 查看上文用过的参考文献是否可以用于写作当前章节
        useful_reference_nums, useful_references = self.citejudgeAgent.judge_usable_references(self.references, subtitle)
        # RAG找新的参考文献
        if len(useful_reference_nums) < LESS_CITE_NUM:
            new_references = rag_search(subtitle)
            new_useful_reference_nums, new_useful_references = self.citejudgeAgent.judge_usable_references(new_references, subtitle, len(self.references))
            self.references = self.references + new_useful_references
        return useful_references + new_useful_references
    
    def generate_content(self, subtitle: str, references: str = "", previous_user_content = None) -> str:
        """生成单个章节内容"""
        # 添加用户输入到会话历史
        user_prompt = self.content_round_prompt.replace("<!-章节标题-!>", subtitle)
        if references:
            user_prompt = user_prompt.replace("<!-参考文献-!>", references)

        self.multi_round_session.append({"role": "user", "content": user_prompt})
        
        # 获取本章节内容
        response = self.llm.generate_response(self.multi_round_session, self.model)
        
        # 添加本章节内容到会话历史
        self.multi_round_session.append({"role": "assistant", "content": response})
        
        # 存储章节内容
        self.generated_contents[subtitle] = response
        
        return response
    
    def generate_article(self, references_map = None) -> str:
        """
        遍历大纲字典树并生成完整文章
        :param references_map: 各章节对应的参考文献字典 {section_title: references}
        :return: 生成的完整文章
        """
        if references_map is None:
            references_map = {}
        
        article_parts = []
        
        def dfs(node, parent_content = None):
            """深度优先遍历大纲字典树"""
            for title, children in node.items():
                # 获取当前章节的参考文献
                # references = references_map.get(title, "")
                # 获取可用于当前章节的参考文献，需要新的参考文献时，更新参考文献引用库
                references = self.judge_and_fetch_usable_references(title)
                # 生成当前章节内容
                content = self.generate_content(
                    subtitle=title,
                    references=references,
                    previous_user_content=parent_content
                )
                article_parts.append(f"## {title}\n{content}\n")
                
                # 递归处理子节点
                if children:
                    dfs(children, content)
        
        # 从根开始遍历
        dfs(self.outline_dict)
        return "\n".join(article_parts)



# 主函数实现智能创作系统
def intelligent_writing_system(query="请你编写一篇讲解低空经济的文章。不超过1000字。"):
    llm = LLM("https://api.deepseek.com/v1")

    outline_agent = OutlineAgent(llm, MODEL)
    outline_agent.generate_first_level_title(query)
    outline_agent.generate_subtitles()
    outline_agent.revise_first_level_title()
    outline_agent.prune_reference()
    """
    def __init__(self, llm, model, article_requirements, outline_dict, references, used_reference_nums, content_system_prompt, content_round_prompt):
        self.llm = llm
        self.model = model
        self.article_requirements = article_requirements
        self.outline_dict = outline_dict  # 大纲字典树
        self.references = references
        self.used_reference_nums = used_reference_nums

        self.multi_round_session = []
        self.generated_contents = {}  # 存储已生成的内容 {section_title: content}
        self.content_system_prompt = content_system_prompt
        self.content_round_prompt = content_round_prompt
        self.initialize_session()
    """
    content_agent = ContentAgent(llm, MODEL, 
                                article_requirements=query,
                                outline_dict=outline_agent.outline_diction,
                                references=outline_agent.references,
                                used_reference_nums=outline_agent.used_reference_nums,
                                content_system_prompt=content_system_prompt_v1,
                                content_round_prompt=content_round_prompt_v1
                                )
    cite_judge_agent = CiteJudgeAgent(llm, MODEL)

    # 大模型生成大纲
    prompt = "请你依据低空经济为标题撰写一篇文章"
    first_level_title = outline_agent.generate_first_level_title(prompt)
    subtitles = outline_agent.generate_subtitles(first_level_title)
    first_level_title = outline_agent.revise_first_level_title(first_level_title, subtitles)

    print("最终大纲:")
    print(first_level_title)
    for subtitle in subtitles:
        print(f"  {subtitle}")

    # 内容生成
    full_content = ""
    all_references = rag_search(first_level_title)
    used_references = []
    for subtitle in subtitles:
        usable_references = cite_judge_agent.judge_usable_references(all_references, subtitle)
        if usable_references < 2:
            new_references = rag_search(subtitle)
            references = new_references
        else:
            references = [ref for ref in all_references if ref not in used_references]

        content = content_agent.generate_content(subtitle, full_content, references)
        full_content += content + "\n"

        # 删除已用的参考文献
        used_references.extend(references)

    print("生成的完整内容:")
    print(full_content)


if __name__ == "__main__":
    intelligent_writing_system()

