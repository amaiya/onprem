"""
Workflow execution engine.

This module contains the WorkflowEngine class that handles loading, validation,
and execution of YAML-based workflows.
"""

import yaml
import os
from typing import Dict, List, Any, Optional
import traceback

from .base import BaseNode
from .exceptions import WorkflowValidationError, NodeExecutionError
from .registry import NODE_REGISTRY


class WorkflowEngine:
    """Executes YAML-defined workflows with validation."""
    
    def __init__(self):
        self.nodes: Dict[str, BaseNode] = {}
        self.connections: List[Dict[str, str]] = []
        self.execution_order: List[str] = []
        self._shared_llm_cache: Dict[str, Any] = {}  # Cache for shared LLM instances
    
    def get_shared_llm(self, llm_config: Dict[str, Any]) -> Any:
        """Get or create a shared LLM instance based on configuration."""
        if not llm_config:
            raise WorkflowValidationError("Empty LLM configuration provided")
        
        # Create a cache key from the LLM configuration
        cache_key = str(sorted(llm_config.items()))
        
        if cache_key not in self._shared_llm_cache:
            try:
                from ..llm.base import LLM
                self._shared_llm_cache[cache_key] = LLM(**llm_config)
            except Exception as e:
                raise WorkflowValidationError(f"Failed to initialize shared LLM with config {llm_config}: {str(e)}")
        
        return self._shared_llm_cache[cache_key]
    
    def load_workflow_from_yaml(self, yaml_path: str) -> None:
        """Load workflow definition from YAML file."""
        if not os.path.exists(yaml_path):
            raise WorkflowValidationError(f"Workflow file not found: {yaml_path}")
        
        try:
            with open(yaml_path, 'r') as f:
                workflow_def = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise WorkflowValidationError(f"Invalid YAML: {str(e)}")
        
        self._parse_workflow(workflow_def)
        self._validate_workflow()
        self._determine_execution_order()
    
    def load_workflow_from_dict(self, workflow_def: Dict[str, Any]) -> None:
        """Load workflow definition from dictionary."""
        self._parse_workflow(workflow_def)
        self._validate_workflow()
        self._determine_execution_order()
    
    def _parse_workflow(self, workflow_def: Dict[str, Any]) -> None:
        """Parse workflow definition and create node instances."""
        self.nodes.clear()
        self.connections.clear()
        
        # Parse nodes
        nodes_def = workflow_def.get("nodes", {})
        for node_id, node_config in nodes_def.items():
            node_type = node_config.get("type")
            if node_type not in NODE_REGISTRY:
                raise WorkflowValidationError(f"Unknown node type: {node_type}")
            
            node_class = NODE_REGISTRY[node_type]
            node = node_class(node_id, node_config.get("config", {}), self)
            
            if not node.validate_config():
                raise WorkflowValidationError(f"Invalid configuration for node {node_id}")
            
            self.nodes[node_id] = node
        
        # Parse connections
        connections_def = workflow_def.get("connections", [])
        for conn in connections_def:
            if not all(key in conn for key in ["from", "from_port", "to", "to_port"]):
                raise WorkflowValidationError("Connection missing required fields: from, from_port, to, to_port")
            self.connections.append(conn)
    
    def _validate_workflow(self) -> None:
        """Validate node connections and types."""
        if not self.nodes:
            raise WorkflowValidationError("Workflow must contain at least one node")
        
        for conn in self.connections:
            source_node = self.nodes.get(conn["from"])
            target_node = self.nodes.get(conn["to"])
            
            if not source_node:
                raise WorkflowValidationError(f"Source node not found: {conn['from']}")
            if not target_node:
                raise WorkflowValidationError(f"Target node not found: {conn['to']}")
            
            # Validate port types
            source_outputs = source_node.get_output_types()
            target_inputs = target_node.get_input_types()
            
            source_port = conn["from_port"]
            target_port = conn["to_port"]
            
            if source_port not in source_outputs:
                raise WorkflowValidationError(
                    f"Source node {conn['from']} has no output port '{source_port}'. "
                    f"Available: {list(source_outputs.keys())}"
                )
            
            if target_port not in target_inputs:
                raise WorkflowValidationError(
                    f"Target node {conn['to']} has no input port '{target_port}'. "
                    f"Available: {list(target_inputs.keys())}"
                )
            
            # Validate type compatibility
            source_type = source_outputs[source_port]
            target_type = target_inputs[target_port]
            
            if source_type != target_type:
                raise WorkflowValidationError(
                    f"Type mismatch: {conn['from']}.{source_port} ({source_type}) -> "
                    f"{conn['to']}.{target_port} ({target_type})"
                )
            
            # Validate node type compatibility using the registry
            if not source_node.can_connect_to(target_node):
                source_type = source_node.NODE_TYPE or "Unknown"
                target_type = target_node.NODE_TYPE or "Unknown"
                
                # Generate helpful error message
                if source_node.is_terminal():
                    raise WorkflowValidationError(
                        f"{source_type} node '{source_node.node_id}' is terminal and cannot connect to other nodes"
                    )
                else:
                    from .base import NODE_TYPES
                    valid_targets = NODE_TYPES.get(source_type)
                    valid_types = valid_targets.can_connect_to if valid_targets else []
                    raise WorkflowValidationError(
                        f"{source_type} node '{source_node.node_id}' cannot connect to {target_type} node '{target_node.node_id}'. "
                        f"Valid target types: {valid_types}"
                    )
    
    def _determine_execution_order(self) -> None:
        """Determine execution order using topological sort."""
        # Build dependency graph
        dependencies = {node_id: set() for node_id in self.nodes}
        dependents = {node_id: set() for node_id in self.nodes}
        
        for conn in self.connections:
            dependencies[conn["to"]].add(conn["from"])
            dependents[conn["from"]].add(conn["to"])
        
        # Topological sort
        self.execution_order = []
        no_deps = [node_id for node_id, deps in dependencies.items() if not deps]
        
        while no_deps:
            current = no_deps.pop(0)
            self.execution_order.append(current)
            
            for dependent in dependents[current]:
                dependencies[dependent].remove(current)
                if not dependencies[dependent]:
                    no_deps.append(dependent)
        
        if len(self.execution_order) != len(self.nodes):
            raise WorkflowValidationError("Workflow contains cycles")
    
    def execute(self, verbose: bool = True) -> Dict[str, Any]:
        """Execute the workflow and return results."""
        if not self.execution_order:
            raise WorkflowValidationError("Workflow not loaded or invalid")
        
        results = {}
        node_outputs = {}
        
        for node_id in self.execution_order:
            node = self.nodes[node_id]
            
            if verbose:
                print(f"Executing node: {node_id}")
            
            # Collect inputs from connected nodes
            inputs = {}
            for conn in self.connections:
                if conn["to"] == node_id:
                    source_output = node_outputs[conn["from"]]
                    inputs[conn["to_port"]] = source_output[conn["from_port"]]
            
            # Execute node
            try:
                output = node.execute(inputs)
                node_outputs[node_id] = output
                results[node_id] = output
                
                if verbose:
                    if "documents" in output:
                        print(f"  -> Processed {len(output['documents'])} documents")
                    elif "status" in output:
                        print(f"  -> {output['status']}")
            
            except Exception as e:
                error_msg = f"Failed to execute node {node_id}: {str(e)}"
                if verbose:
                    print(f"  -> ERROR: {error_msg}")
                traceback.print_exc()
                raise NodeExecutionError(error_msg)
        
        return results
