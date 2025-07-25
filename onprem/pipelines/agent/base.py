"""Agent-based task execution pipeline that wraps smolagents to provide autonomous task completion capabilities using local LLMs."""

# AUTOGENERATED! DO NOT EDIT! File to edit: ../../../nbs/04_pipelines.agent.base.ipynb.

# %% auto 0
__all__ = ['Agent']

# %% ../../../nbs/04_pipelines.agent.base.ipynb 3
import onprem
from typing import Callable
import warnings
from ...ingest.stores.base import VectorStore

from smolagents import PythonInterpreterTool, WebSearchTool, VisitWebpageTool, Tool as SA_Tool
from smolagents import ToolCallingAgent, CodeAgent
from .model import AgentModel
from . import tools as tool_utils

class Agent:
    """
    Pipeline for agent-based task execution using smolagents.
    Extra kwargs are supplied directly to agent instantation.
    
    Args:
        llm (LLM): An onprem LLM instance to use for agent reasoning
        agent_type (str, optional): Type of agent to use ('tool_calling' or 'code')
        max_steps (int, optional): Maximum number of steps the agent can take
        tools (dict, optional): a dictionary of tools for agent to use
    """
    

    
    def __init__(
        self, 
        llm: onprem.LLM,
        agent_type: str = "tool_calling",
        max_steps: int = 20,
        tools:dict = {},
        **kwargs,
    ):

        self.model = AgentModel(llm)
        self.tools = tools or {}
        self.agent_type = agent_type
        self.max_steps = max_steps
        self.kwargs = kwargs
                     
        # Initialize the agent
        self._init_agent()
    
    def _init_agent(self):
        """Initialize the appropriate smolagents agent type."""
        # Convert the LLM to a smolagents-compatible model
        
        # Get the tool list
        tool_list = list(self.tools.values())
        
        # Initialize the agent based on the agent type
        if self.agent_type == "tool_calling":
            self.agent = ToolCallingAgent(
                tools=tool_list,
                model=self.model,
                max_steps = self.max_steps,
                **self.kwargs
             )
        elif self.agent_type == "code":
            self.agent = CodeAgent(
                tools=tool_list,
                model=self.model,
                max_steps = self.max_steps,
                **self.kwargs
             )
        else:
            raise ValueError(f"Unsupported agent type: {self.agent_type}")
    
   
    def add_tool(self, name: str, tool_instance): # tool_instance is SA_Tool
        """
        Add a tool to the agent.
        
        Args:
            name (str): The name of the tool
            tool_instance (Tool): The tool instance
        """
        self.tools[name] = tool_instance
        self._init_agent()  # Reinitialize the agent with the new tool

    def add_default_tool(
        self, name:str
    ):
        """
        Create one of the built-in tools
                 
        Returns:
            Tool: The created tool
        """
        if name not in tool_utils.DEFAULT_TOOLS:
            raise ValueError(f'{name} is not one of the built-in tools: {tool_utils.DEFAULT_TOOLS.keys()}')
        self.add_tool(name, tool_utils.DEFAULT_TOOLS[name])
    
    def add_websearch_tool(
        self, 
    ):
        """
        Create a tool to perform Web searches.
                 
        Returns:
            Tool: The created tool
        """
        name = 'websearch'
        self.add_default_tool(name)

    
    def add_webview_tool(
        self, 
    ):
        """
        Create a tool to visit Web page
                 
        Returns:
            Tool: The created tool
        """
        name = 'webview'
        self.add_default_tool(name)

    
    def add_python_tool(
        self, 
    ):
        """
        Create a tool to access Python interpreter
                 
        Returns:
            Tool: The created tool
        """
        name = 'python'
        self.add_default_tool(name)

    
    def add_function_tool(
        self, 
        func: Callable, 
    ):
        """
        Create a tool from a function and its documentation.
        
        Args:
            func (Callable): The function to wrap as a tool
            name (str, optional): The name of the tool (defaults to function name)
            description (str, optional): The description of the tool (defaults to function docstring)
            
        Returns:
            Tool: The created tool
        """
        from types import FunctionType, MethodType

        if not isinstance(func, SA_Tool) and (isinstance(func, (FunctionType, MethodType)) or hasattr(func, "__call__")):
            name = func.__name__
            tool = tool_utils.createtool(func)
        else:
            raise ValueError(f'{func} is not a callable method or function')
        self.add_tool(name, tool)


    def add_vectorstore_tool(
        self, 
        name: str, 
        store: VectorStore, 
        description: str = "Search a vector database for relevant information",
    ):
        """
        Create a tool from a VectorStore instance.
        
        Args:
            name (str): The name of the vector store tool
            store (VectorStore): The vector store instance
            description (str, optional): The description of the vector store  
        Returns:
            Tool: The created tool
        """
        self.add_tool(name, tool_utils.VectorStoreTool(name, description, store))


    def add_mcp_tool(self, url:str):
        """
        Add tool to access MCP server
        """
        import mcpadapt.core
        from mcpadapt.smolagents_adapter import SmolAgentsAdapter
        
        mcp_tools = mcpadapt.core.MCPAdapt({"url": url}, SmolAgentsAdapter()).tools()
        for i, mcp_tool in enumerate(mcp_tools):
            self.add_tool(f'{url} (i)', mcp_tool)


    def run(self, task: str) -> str:
        """
        Run the agent on a given task.
        
        Args:
            task (str): The task description
            
        Returns:
            str: The agent's response
        """
        if not self.tools and self.agent_type == "tool_calling":
            raise ValueError('No tools have been added to agent. Please add at least one tool using on eof the Agent.add_* methods.')
            
        result = self.agent.run(task)
        return '\n'.join(result) if isinstance(result, list) else result
