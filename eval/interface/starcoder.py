from typing import Dict, Union, List, Tuple
from transformers import AutoTokenizer

class StarCoderInterface:
    def __init__(self, model_id):
        self.total_budget = 4096
        self.suffix_budget = int(self.total_budget * 0.15)
        self.rag_budget = int(self.total_budget * 0.5)
        self.max_rag_num = 1

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.FIM_PRE = "<fim_prefix>"
        self.FIM_SUF = "<fim_suffix>"
        self.FIM_MID = "<fim_middle>"

        self.prefix_structure = [("before_cursor", -1)]
        self.suffix_structure = [("after_cursor", -1)]
        self.rag_structure = [("rag", -1)]
    
    
    def gen_prompt(self, context) -> str:
        prefix_init = ""
        suffix_init = ""

        prefix_input = {
            idx: context[idx] for idx, budget in self.prefix_structure
        }
        suffix_input = {
            idx: context[idx] for idx, budget in self.suffix_structure
        }

        rag_init = ""
        rag_input = {}
        if context.get("rag"):
            rag_input["rag"] = context["rag"]

        prompt_rag, budget = self.manage_prompt_size(
            rag_init,
            rag_input,
            self.rag_structure,
            self.total_budget,
            self.tokenizer,
        )
        prompt_prefix, budget = self.manage_prompt_size(
            prefix_init,
            prefix_input,
            self.prefix_structure,
            budget - self.suffix_budget,
            self.tokenizer,
        )
        prompt_suffix, budget = self.manage_prompt_size(
            suffix_init,
            suffix_input,
            self.suffix_structure,
            budget + self.suffix_budget,
            self.tokenizer,
        )

        # suffix_first = False
        # if suffix_first:
        #     prompt_input = FIM_PRE + FIM_SUF + prompt_suffix + FIM_MID + prompt_prefix
        # else:
        #     prompt_input = FIM_PRE + prompt_prefix + FIM_SUF + prompt_suffix + FIM_MID

        prompt = (
            self.FIM_PRE
            + prompt_rag
            + prompt_prefix
            + self.FIM_SUF
            + prompt_suffix
            + self.FIM_MID
        )
        return prompt
    
    def convert_to_commit(self, code):
        lines = code.split("\n")
        lines = ["//" + line for line in lines]
        code = "\n".join(lines)
        return code
    
    def manage_prompt_size(
        self,
        init_context: str,
        context_dict,
        structure: List[Tuple[str, int]],
        total_budget: int,
        tokenizer,
    ) -> Tuple[str, int]:
        # tokenizer应该放到外面去做，方便拓展到别的模型。
        tokenized_context = self.tokenizer.tokenize(init_context)
        for key, key_budget in structure:
            if key not in context_dict:
                continue
            context_budget = total_budget - len(tokenized_context)
            if key_budget >= 0:
                context_budget = min(context_budget, key_budget)
            if key == "rag":
                context_content: List = context_dict[key]
                context_content = context_content[: self.max_rag_num]
                context_budget_cpy = context_budget
                for content in context_content:
                    code = self.convert_to_commit(content["code"])
                    file_name = content["path"].split("/")[-1]
                    context = (
                        f"""// Compare this snippet from {file_name}: \n{code}\n\n"""
                    )
                    tokenized_content = self.tokenizer.tokenize(context)
                    if len(tokenized_content) > context_budget_cpy:
                        break
                    tokenized_context.extend(tokenized_content) 
                    context_budget_cpy -= len(tokenized_content)
            else:
                context_content: str = context_dict[key]
                tokenized_content = self.tokenizer.tokenize(context_content)
                if key == "before_cursor":
                    tokenized_content = tokenized_content[-context_budget:]
                else:
                    tokenized_content = tokenized_content[:context_budget]
                tokenized_context.extend(tokenized_content)

        if len(tokenized_context) == 0:
            return "", total_budget

        context_input = tokenizer.convert_tokens_to_string(tokenized_context)
        budget = total_budget - len(tokenized_context)
        return context_input, budget
