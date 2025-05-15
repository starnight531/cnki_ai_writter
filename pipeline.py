from llm import LLM
from .prompts.outline_prompt import *
from .prompts.content_prompt import *
from .prompts.cite_judge_prompt import *

# 模拟 RAG 搜索，返回固定内容
def rag_search(query):
    return [f"固定参考文献 - {query}"]


class OutlineAgent:
    def __init__(self, llm, model):
        self.llm = llm
        self.model = model
        self.level1_title = ""
        self.outline = ""
        self.references = []

    def generate_first_level_title(self, user_input):
        user_input = outline_prompt_v1["level_1_title_prompt"].replace("<!-文章要求-!>", user_input)
        self.level1_title = self.llm.generate_response(user_input, self.model)
        return self.level1_title

    def generate_subtitles(self, user_input = None):
        references = rag_search(self.level1_title)
        if user_input:
            references.append("一定参考用户提供的参考文献：\n"+user_input)
        
        references = [f"""##### 参考文章{i}：\n{ref}""" for i, ref in enumerate(references, start=1)]
        self.references = references
        references_text = "\n\n".join(references)

        user_input = outline_prompt_v1["level_n_title_prompt"].replace("<!-参考文献-!>", references_text)

        self.outline = self.llm.generate_response(user_input, self.model).split('\n')
        return self.outline


    def revise_first_level_title(self, user_input = None):
        user_input = outline_prompt_v1["revise_level_1_title_prompt"].replace("<!-大纲-!>", self.outline).replace("<!-参考文献-!>", self.references)
        self.outline_diction = self.llm.generate_response(user_input, self.model)
        return self.outline_diction


# class ContentAgent:
#     def __init__(self, llm, model, outline_diction):
#         self.llm = llm
#         self.model = model
#         self.outline_diction = outline_diction
#         self.multi_round_session = []   # 可以考虑pick llm，删除user，作为上文

#     def generate_content(self, subtitle, previous_content, references):
        
#         return self.llm.generate_response(user_input, self.model)

class ContentAgent:
    def __init__(self, llm, model, article_requirements, outline_dict, content_system_prompt, content_round_prompt):
        self.llm = llm
        self.model = model
        self.article_requirements = article_requirements
        self.outline_dict = outline_dict  # 大纲字典树
        self.multi_round_session = []
        self.generated_contents = {}  # 存储已生成的内容 {section_title: content}
        self.content_system_prompt = content_system_prompt
        self.content_round_prompt = content_round_prompt
        self.initialize_session()
    
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
       
    def generate_content(self, subtitle: str, references: str = "", previous_content = None) -> str:
        """生成单个章节内容"""
        # 添加用户输入到会话历史
        user_prompt = self.content_round_prompt.replace("<!-章节标题-!>", subtitle)
        if references:
            user_prompt = user_prompt.replace("<!-参考文献-!>", references)

        self.multi_round_session.append({"role": "user", "content": user_prompt})
        
        # 获取模型响应
        response = self.llm.generate_response(self.multi_round_session, self.model)
        
        # 添加模型响应到会话历史
        self.multi_round_session.append({"role": "assistant", "content": response})
        
        # 存储生成的内容
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
                references = references_map.get(title, "")
                
                # 生成当前章节内容
                content = self.generate_content(
                    subtitle=title,
                    references=references,
                    previous_content=parent_content
                )
                article_parts.append(f"## {title}\n\n{content}\n")
                
                # 递归处理子节点
                if children:
                    dfs(children, content)
        
        # 从根开始遍历
        dfs(self.outline_dict)
        return "\n".join(article_parts)



class CiteJudgeAgent:
    def __init__(self, llm, model):
        self.llm = llm
        self.model = model

    def judge_usable_references(self, references, subtitle):
        user_input = f"在这些参考文献 {references} 中，有多少可以用于编写小标题 '{subtitle}' 的内容，请直接给出数量"
        try:
            return int(self.llm.generate_response(user_input, self.model))
        except ValueError:
            return 0


# 主函数实现智能创作系统
def intelligent_writing_system():
    models = ["R1-distill-Qwen2.5-1.5B", "R1-distill-Qwen2.5-7B_GGUF", "Qwen2.5-3B-Instruct", "QwQ-32B"]
    # 这里可根据需要选择模型
    model = models[3]
    llm = LLM("http://10.25.160.22:10425/v1")

    outline_agent = OutlineAgent(llm, model)
    content_agent = ContentAgent(llm, model)
    cite_judge_agent = CiteJudgeAgent(llm, model)

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

