
from typing import List, Dict, Any
import re

class BrainResponseFormatter:
    """Handles the formatting of ALU Brain responses for contextual presentation"""
    
    def format_for_context(self, results: List[Dict[str, Any]]) -> str:
        """Format search results as context for the prompt engine with enhanced presentation"""
        if not results:
            return "No relevant information found in the ALU knowledge base."
            
        formatted_context = "# ALU Knowledge Base Results\n\n"
        
        for i, result in enumerate(results):
            entry = result['entry']
            category = result['category']
            
            # Format category nicely
            formatted_category = category.replace('_', ' ').title()
            
            # Format header based on entry type and category
            entry_type = entry.get('type', 'short_response')
            formatted_context += f"## {i+1}. {formatted_category} - {self._get_type_label(entry_type)}\n\n"
            
            # Format the question/topic
            formatted_context += f"**Question:** {entry.get('question', 'Information')}\n\n"
            
            # Format answer based on entry type
            answer = entry.get('answer', '')
            
            if entry_type == 'link_response' and 'links' in entry:
                formatted_context += f"{answer}\n\n"
                formatted_context += "**Relevant Links:**\n"
                for link in entry.get('links', []):
                    formatted_context += f"- [{link.get('text', 'Link')}]({link.get('url', '')})\n"
            
            elif entry_type == 'table_response' and 'table' in entry:
                formatted_context += f"{answer}\n\n"
                table = entry.get('table', {})
                if 'headers' in table and 'rows' in table:
                    # Format as markdown table
                    headers = table.get('headers', [])
                    rows = table.get('rows', [])
                    
                    # Create table header
                    formatted_context += "| " + " | ".join(headers) + " |\n"
                    formatted_context += "| " + " | ".join(["---" for _ in headers]) + " |\n"
                    
                    # Create table rows
                    for row in rows:
                        formatted_context += "| " + " | ".join(row) + " |\n"
            
            elif entry_type == 'statistical_response' and 'statistics' in entry:
                formatted_context += f"{answer}\n\n"
                formatted_context += "**Key Statistics:**\n"
                for stat in entry.get('statistics', []):
                    formatted_context += f"- **{stat.get('metric', '')}:** {stat.get('value', '')}\n"
            
            elif entry_type == 'date_response' and 'dates' in entry:
                formatted_context += f"{answer}\n\n"
                formatted_context += "**Important Dates:**\n"
                
                # Sort dates by deadline if possible
                dates = sorted(entry.get('dates', []), 
                               key=lambda x: x.get('deadline', ''), 
                               reverse=False)
                               
                for date_item in dates:
                    formatted_context += f"- **{date_item.get('round', '')}:** {date_item.get('deadline', '')}\n"
            
            elif entry_type == 'procedural_response' and 'steps' in entry:
                formatted_context += f"{answer}\n\n"
                formatted_context += "**Process Steps:**\n"
                for i, step in enumerate(entry.get('steps', [])):
                    formatted_context += f"{i+1}. {step}\n"
            
            else:
                # Enhanced formatting for text-based responses
                paragraphs = answer.split('\n\n')
                
                # Add paragraph structure
                for paragraph in paragraphs:
                    # Format bullet points if they exist
                    if '\n- ' in paragraph:
                        bullet_parts = paragraph.split('\n- ')
                        formatted_context += f"{bullet_parts[0]}\n\n"
                        for bullet in bullet_parts[1:]:
                            formatted_context += f"- {bullet}\n"
                    # Format numbered lists if they exist
                    elif re.search(r'\n\d+\. ', paragraph):
                        list_parts = re.split(r'(\n\d+\. )', paragraph)
                        formatted_context += f"{list_parts[0]}\n"
                        for i in range(1, len(list_parts), 2):
                            if i+1 < len(list_parts):
                                formatted_context += f"{list_parts[i]}{list_parts[i+1]}"
                    else:
                        formatted_context += f"{paragraph}\n\n"
            
            # Add metadata if available and relevant
            if 'metadata' in entry and entry['metadata']:
                metadata = entry['metadata']
                formatted_context += "\n**Source Information:**\n"
                
                if metadata.get('source'):
                    formatted_context += f"- **Source:** {metadata['source']}\n"
                    
                if metadata.get('lastUpdated'):
                    formatted_context += f"- **Last Updated:** {metadata['lastUpdated']}\n"
                    
                if metadata.get('author'):
                    formatted_context += f"- **Author:** {metadata['author']}\n"
                    
                if metadata.get('department'):
                    formatted_context += f"- **Department:** {metadata['department']}\n"
            
            formatted_context += "\n---\n\n"
        
        # Add guidance for using this information
        formatted_context += "When crafting your response, use the above information to provide accurate details about ALU. Format your response with clear headings, bullet points where appropriate, and maintain a professional tone. Include relevant links if available."
        
        return formatted_context
    
    def _get_type_label(self, entry_type: str) -> str:
        """Convert entry type to a human-readable label"""
        type_labels = {
            'link_response': 'Resource Links',
            'table_response': 'Tabular Data',
            'statistical_response': 'Statistics', 
            'date_response': 'Important Dates',
            'procedural_response': 'Process Guide',
            'long_response': 'Detailed Explanation',
            'short_response': 'Quick Answer'
        }
        
        return type_labels.get(entry_type, 'Information')
    
    def format_as_markdown(self, result: Dict[str, Any]) -> str:
        """Format a single search result as a standalone markdown document"""
        if not result or 'entry' not in result:
            return "No information available."
            
        entry = result['entry']
        category = result['category'].replace('_', ' ').title()
        entry_type = entry.get('type', 'short_response')
        
        markdown = f"# {entry.get('question', 'ALU Information')}\n\n"
        markdown += f"## {self._get_type_label(entry_type)} from {category}\n\n"
        
        # Format the answer content with markdown
        answer = entry.get('answer', '')
        if answer:
            # Process answer to enhance markdown formatting
            answer = self._enhance_markdown_formatting(answer)
            markdown += f"{answer}\n\n"
        
        # Format specialized content based on type
        if entry_type == 'link_response' and 'links' in entry:
            markdown += "### Related Resources\n\n"
            for link in entry.get('links', []):
                markdown += f"- [{link.get('text', 'Link')}]({link.get('url', '')})\n"
                
        elif entry_type == 'statistical_response' and 'statistics' in entry:
            markdown += "### Key Figures\n\n"
            for stat in entry.get('statistics', []):
                markdown += f"- **{stat.get('metric', '')}:** {stat.get('value', '')}\n"
                
        elif entry_type == 'procedural_response' and 'steps' in entry:
            markdown += "### Step-by-Step Process\n\n"
            for i, step in enumerate(entry.get('steps', [])):
                markdown += f"{i+1}. {step}\n"
                
        elif entry_type == 'date_response' and 'dates' in entry:
            markdown += "### Important Dates\n\n"
            for date_item in entry.get('dates', []):
                markdown += f"- **{date_item.get('round', '')}:** {date_item.get('deadline', '')}\n"
        
        # Add metadata footer
        if 'metadata' in entry and entry['metadata']:
            markdown += "\n---\n\n"
            markdown += "*Source information:* "
            
            metadata = entry['metadata']
            info_parts = []
            
            if metadata.get('source'):
                info_parts.append(f"Source: {metadata['source']}")
                
            if metadata.get('lastUpdated'):
                info_parts.append(f"Updated: {metadata['lastUpdated']}")
                
            markdown += " | ".join(info_parts)
        
        return markdown
    
    def _enhance_markdown_formatting(self, text: str) -> str:
        """Enhance text with better markdown formatting"""
        # Format section headers
        text = re.sub(r'(?m)^([A-Z][A-Za-z\s]+):$', r'### \1', text)
        
        # Format bold terms (terms followed by colon in sentences)
        text = re.sub(r'([A-Z][a-zA-Z\s]+):(\s)', r'**\1:**\2', text)
        
        # Format bullet lists (lines starting with - or *)
        text = re.sub(r'(?m)^[-*]\s*(.*)', r'- \1', text)
        
        # Format numbered lists (lines starting with 1. 2. etc)
        text = re.sub(r'(?m)^(\d+)\.\s*(.*)', r'\1. \2', text)
        
        return text
