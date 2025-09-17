"""
Exporter node implementations for workflow pipelines.

This module contains exporter nodes that export processed results to various
formats including CSV, Excel, and JSON.
"""

from typing import Dict, Any

from .base import ExporterNode
from .exceptions import NodeExecutionError


class CSVExporterNode(ExporterNode):
    """Exports results to CSV format."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = inputs.get("results", [])
        if not results:
            return {"status": "No results to export"}
        
        output_path = self.config.get("output_path", "results.csv")
        columns = self.config.get("columns", None)  # If None, use all keys from first result
        
        try:
            import csv
            
            def clean_text_for_csv(text):
                """Clean text content for proper CSV formatting."""
                if text is None:
                    return ""
                text = str(text)
                # Normalize line endings and replace problematic characters
                text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
                # Replace multiple spaces with single space
                text = ' '.join(text.split())
                # Limit length to prevent Excel issues
                if len(text) > 32000:  # Excel cell limit is ~32,767 characters
                    text = text[:32000] + "... [truncated]"
                return text
            
            # Determine columns
            if columns is None:
                columns = list(results[0].keys()) if results else []
            
            # Flatten nested metadata if present
            flattened_results = []
            for result in results:
                flat_result = {}
                for key, value in result.items():
                    if key == 'metadata' and isinstance(value, dict):
                        # Flatten metadata with prefix
                        for meta_key, meta_value in value.items():
                            flat_result[f"meta_{meta_key}"] = clean_text_for_csv(meta_value)
                    else:
                        flat_result[key] = clean_text_for_csv(value)
                flattened_results.append(flat_result)
            
            # Update columns to include flattened metadata
            if flattened_results:
                all_columns = set()
                for result in flattened_results:
                    all_columns.update(result.keys())
                if columns is None or 'metadata' in columns:
                    columns = sorted(list(all_columns))
            
            # Write CSV with proper quoting for Excel compatibility
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:  # utf-8-sig for Excel BOM
                writer = csv.DictWriter(
                    csvfile, 
                    fieldnames=columns,
                    quoting=csv.QUOTE_ALL,  # Quote all fields for better Excel compatibility
                    lineterminator='\n'     # Use Unix line endings consistently
                )
                writer.writeheader()
                for result in flattened_results:
                    # Only include columns that exist in the result
                    row = {col: result.get(col, "") for col in columns}
                    writer.writerow(row)
            
            return {"status": f"Exported {len(results)} results to {output_path}"}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to export CSV: {str(e)}")


class ExcelExporterNode(ExporterNode):
    """Exports results to Excel format."""
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        results = inputs.get("results", [])
        if not results:
            return {"status": "No results to export"}
        
        output_path = self.config.get("output_path", "results.xlsx")
        sheet_name = self.config.get("sheet_name", "Results")
        
        try:
            import pandas as pd
            
            # Flatten results similar to CSV exporter
            flattened_results = []
            for result in results:
                flat_result = {}
                for key, value in result.items():
                    if key == 'metadata' and isinstance(value, dict):
                        for meta_key, meta_value in value.items():
                            flat_result[f"meta_{meta_key}"] = meta_value
                    else:
                        flat_result[key] = value
                flattened_results.append(flat_result)
            
            # Create DataFrame and export
            df = pd.DataFrame(flattened_results)
            df.to_excel(output_path, sheet_name=sheet_name, index=False)
            
            return {"status": f"Exported {len(results)} results to {output_path}"}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to export Excel: {str(e)}")


class JSONExporterNode(ExporterNode):
    """Exports results to JSON format."""
    
    def get_input_types(self) -> Dict[str, str]:
        # Accept both List[Dict] (normal) and Dict (from aggregators)
        return {"results": "List[Dict]", "result": "Dict"}
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Handle both List[Dict] (normal) and Dict (from aggregators)
        if "results" in inputs:
            results = inputs["results"]
            if not results:
                return {"status": "No results to export"}
        elif "result" in inputs:
            # Single result from aggregator - wrap in list
            result = inputs["result"]
            if not result:
                return {"status": "No result to export"}
            results = [result]
        else:
            return {"status": "No results to export"}
        
        output_path = self.config.get("output_path", "results.json")
        pretty_print = self.config.get("pretty_print", True)
        
        try:
            import json
            
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                if pretty_print:
                    json.dump(results, jsonfile, indent=2, ensure_ascii=False)
                else:
                    json.dump(results, jsonfile, ensure_ascii=False)
            
            return {"status": f"Exported {len(results)} results to {output_path}"}
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to export JSON: {str(e)}")


class JSONResponseExporterNode(ExporterNode):
    """Exports extracted JSON responses from LLM processing, removing metadata and extracting only the JSON content."""
    
    def get_input_types(self) -> Dict[str, str]:
        # Accept both List[Dict] (normal) and Dict (from aggregators)
        return {"results": "List[Dict]", "result": "Dict"}
    
    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Handle both List[Dict] (normal) and Dict (from aggregators)
        if "results" in inputs:
            results = inputs["results"]
            if not results:
                return {"status": "No results to export"}
        elif "result" in inputs:
            # Single result from aggregator - wrap in list
            result = inputs["result"]
            if not result:
                return {"status": "No result to export"}
            results = [result]
        else:
            return {"status": "No results to export"}
        
        output_path = self.config.get("output_path", "extracted_responses.json")
        pretty_print = self.config.get("pretty_print", True)
        response_field = self.config.get("response_field", "response")  # Field containing the JSON response
        
        try:
            # Import JSON extraction helper
            from onprem.llm.helpers import extract_json
            import json
            
            extracted_responses = []
            
            for result in results:
                # Get the response text (try different possible fields)
                response_text = None
                for field in [response_field, "response", "aggregated_response", "output"]:
                    if field in result and result[field]:
                        response_text = result[field]
                        break
                
                if not response_text:
                    # Skip results without response text
                    continue
                
                # Extract JSON from the response text
                try:
                    extracted_json = extract_json(response_text)
                    if extracted_json:
                        # If single JSON object, add it directly
                        if isinstance(extracted_json, dict):
                            extracted_responses.append(extracted_json)
                        # If list of JSON objects, extend the list
                        elif isinstance(extracted_json, list):
                            extracted_responses.extend(extracted_json)
                        else:
                            # Fallback: wrap in object
                            extracted_responses.append({"extracted_data": extracted_json})
                    else:
                        # If no JSON found, include the raw response
                        extracted_responses.append({"raw_response": response_text})
                        
                except Exception as e:
                    # If JSON extraction fails, include the raw response with error info
                    extracted_responses.append({
                        "raw_response": response_text,
                        "extraction_error": str(e)
                    })
            
            # Export the extracted JSON responses
            with open(output_path, 'w', encoding='utf-8') as jsonfile:
                if pretty_print:
                    json.dump(extracted_responses, jsonfile, indent=2, ensure_ascii=False)
                else:
                    json.dump(extracted_responses, jsonfile, ensure_ascii=False)
            
            return {"status": f"Exported {len(extracted_responses)} JSON responses to {output_path}"}
            
        except Exception as e:
            raise NodeExecutionError(f"Node {self.node_id}: Failed to export JSON responses: {str(e)}")