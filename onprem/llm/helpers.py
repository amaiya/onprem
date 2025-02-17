"""Helper utilities for using LLMs"""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/00_llm.helpers.ipynb.

# %% auto 0
__all__ = ['SUBQUESTION_PROMPT', 'FOLLOWUP_PROMPT', 'TITLE_PROMPT', 'TABLE_PROMPT_EXACT', 'TABLE_PROMPT', 'parse_json_markdown',
           'parse_code_markdown', 'decompose_question', 'needs_followup', 'Title', 'extract_title', 'TableSummary',
           'caption_table_text', 'caption_tables']

# %% ../../nbs/00_llm.helpers.ipynb 3
from ..utils import SafeFormatter
import json
import yaml
from typing import List, Any, Union
from pydantic import BaseModel, Field
from langchain_core.documents import Document


# %% ../../nbs/00_llm.helpers.ipynb 4
def _marshal_llm_to_json(output: str) -> str:
    """
    Extract a substring containing valid JSON or array from a string.

    **Args:**
    
        - output: A string that may contain a valid JSON object or array surrounded by extraneous characters or information.

    **Returns:**
        
        - A string containing a valid JSON object or array.
    """
    output = output.strip()

    left_square = output.find("[")
    left_brace = output.find("{")

    if left_square < left_brace and left_square != -1:
        left = left_square
        right = output.rfind("]")
    else:
        left = left_brace
        right = output.rfind("}")

    return output[left : right + 1]


def parse_json_markdown(text: str) -> Any:
    """
    Parse json embedded in markdown into dictionary
    """
    if "```json" in text:
        text = text.split("```json")[1].strip().strip("```").strip()

    json_string = _marshal_llm_to_json(text)

    try:
        json_obj = json.loads(json_string)
    except json.JSONDecodeError as e_json:
        try:
            # NOTE: parsing again with pyyaml
            #       pyyaml is less strict, and allows for trailing commas
            #       right now we rely on this since guidance program generates
            #       trailing commas
            json_obj = yaml.safe_load(json_string)
        except yaml.YAMLError as e_yaml:
            raise OutputParserException(
                f"Got invalid JSON object. Error: {e_json} {e_yaml}. "
                f"Got JSON string: {json_string}"
            )
        except NameError as exc:
            raise ImportError("Please pip install PyYAML.") from exc

    return json_obj

def parse_code_markdown(text: str, only_last: bool) -> List[str]:
    """
    Parsing embedded code out of markdown string
    """
    # Regular expression pattern to match code within triple-backticks
    pattern = r"```(.*?)```"

    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the last matched group if requested
    code = matches[-1] if matches and only_last else matches

    # If empty we optimistically assume the output is the code
    if not code:
        # we want to handle cases where the code may start or end with triple
        # backticks
        # we also want to handle cases where the code is surrounded by regular
        # quotes
        # we can't just remove all backticks due to JS template strings

        candidate = text.strip()

        if candidate.startswith('"') and candidate.endswith('"'):
            candidate = candidate[1:-1]

        if candidate.startswith("'") and candidate.endswith("'"):
            candidate = candidate[1:-1]

        if candidate.startswith("`") and candidate.endswith("`"):
            candidate = candidate[1:-1]

        # For triple backticks we split the handling of the start and end
        # partly because there can be cases where only one and not the other
        # is present, and partly because we don't need to be so worried
        # about it being a string in a programming language
        if candidate.startswith("```"):
            candidate = re.sub(r"^```[a-zA-Z]*", "", candidate)

        if candidate.endswith("```"):
            candidate = candidate[:-3]
        code = [candidate.strip()]

    return code

# %% ../../nbs/00_llm.helpers.ipynb 5
SUBQUESTION_PROMPT = """\
Given a user question, output a list of relevant sub-questions \
in json markdown that when composed can help answer the full user question.
Only return the JSON response, with no additional text or explanations.

# Example 1
<User Question>
Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021

<Output>
```json
{
    "items": [
        {
            "sub_question": "What is the revenue growth of Uber",
        },
        {
            "sub_question": "What is the EBITDA of Uber",
        },
        {
            "sub_question": "What is the revenue growth of Lyft",
        },
        {
            "sub_question": "What is the EBITDA of Lyft",
        }
    ]
}
```

# Example 2
<User Question>
{query_str}

<Output>
"""

def decompose_question(question:str, llm, parse=True, **kwargs):
    """
    Decompose a question into subquestions
    """
    prompt = SafeFormatter({'query_str': question}).format(SUBQUESTION_PROMPT)
    json_string = llm.prompt(prompt)
    json_dict = parse_json_markdown(json_string)
    subquestions = [d['sub_question'] for d in json_dict['items']]
    return subquestions
    

# %% ../../nbs/00_llm.helpers.ipynb 6
FOLLOWUP_PROMPT = """\
Given a question, answer "yes" only if the question is complex and follow-up questions are needed or "no" if not.
Always respond with "no" for short questions that are less than 8 words.
Answer only with either "yes" or "no" with no additional text or explanations.

# Example 1
<User Question>
Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021

<Output>
yes

# Example 2
<User Question>
How is the Coast Guard using artificial intelligence?

<Output>
No

# Example 3
<User Question
What is AutoGluon?

<Output>
No

# Example 4
<User Question>
{query_str}

<Output>
"""

def needs_followup(question:str, llm, parse=True, **kwargs):
    """
    Decide if follow-up questions are needed
    """
    prompt = SafeFormatter({'query_str': question}).format(FOLLOWUP_PROMPT)
    output = llm.prompt(prompt)
    return "yes" in output.lower()

# %% ../../nbs/00_llm.helpers.ipynb 7
TITLE_PROMPT = """\
Context: {context_str}.\n\nGive a title that summarizes what the context describes. \
Title: """

class Title(BaseModel):
    title: str = Field(description="title of text")

def extract_title(docs_or_text:Union[List[Document], str], llm, max_words=1024, retries=1, **kwargs):
    """
    Extract or infer the title for the given text

    **Args**
      - docs_or_text: Either a list of LangChain Document objects or a single text string
      - llm: An onprem.LLM instance
      - max_words: Maximum words to consider
      - retries: Number of tries to correctly extract title
    """
    if not docs_or_text:
        raise ValueError('docs_or_text cannot be empty')
    if isinstance(docs_or_text, list):
        text = ""
        for doc in docs_or_text:
            if not doc.page_content.strip() or len(doc.page_content.strip()) < 32:
                continue
            text = doc.page_content.strip()
            break
    else:
        text = docs_or_text
    text = " ".join(text.split()[:max_words])
    for i in range(retries+1):
        obj = llm.pydantic_prompt(TITLE_PROMPT.replace("{context_str}", text), pydantic_model=Title)
        if not isinstance(obj, str):
            break
    return "" if isinstance(obj, str) else obj.title

# %% ../../nbs/00_llm.helpers.ipynb 8
from ..utils import CAPTION_STR

TABLE_PROMPT_EXACT= """\
{context_str}\n\nWhat is this table about? Give a very concise summary (imagine you are adding a new caption and summary for this table), \
and output the real/existing table title/caption if context provided."""
TABLE_PROMPT= """\
{context_str}\n\nWhat is this table about? Give a very concise summary (imagine you are adding a new caption and summary for this table)."""
class TableSummary(BaseModel):
    summary: str = Field(description="concise summary or caption of table")

def caption_table_text(table_text:str, llm, max_chars=4096, retries=1, attempt_exact=False, **kwargs):
    """
    Caption table text
    """
    table_text = table_text[:max_chars]
    for i in range(retries+1):
        prompt = TABLE_PROMPT_EXACT if attempt_exact else TABLE_PROMPT
        obj = llm.pydantic_prompt(prompt.replace("{context_str}", table_text), pydantic_model=Title)
        if not isinstance(obj, str):
            break
    return "" if isinstance(obj, str) else obj.title
    
def caption_tables(docs:List[Document], llm, max_chars=4096, max_tables=3, retries=1, 
                   attempt_exact=False,
                   only_caption_missing=False, **kwargs):
    """
    Given a list of Documents, auto-caption or summarize any tables within list.

    **Args**
      - docs_or_text: A list of LangChain Document objects
      - llm: An onprem.LLM instance
      - max_chars: Maximum characters to consider
      - retries: Number of tries to correctly auto-caption table
      - attempt_exact: Try to exact existing caption if it exists.
      - only_caption_missing: Only caption tables without a caption
    """
    if not docs:
        raise ValueError('docs_or_text cannot be empty')
    n_processed = 0
    for doc in docs:
        if not 'table' in doc.metadata or not doc.metadata['table']:
            continue
        if only_caption_missing and CAPTION_STR in doc.page_content:
            continue
        doc.page_content = caption_table_text(doc.page_content, 
                                              llm=llm, max_chars=max_chars, retries=retries) + '\n\n' + doc.page_content        
        n_processed +=1
        if n_processed >= max_tables:
            break
    return
