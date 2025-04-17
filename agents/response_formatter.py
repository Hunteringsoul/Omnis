from typing import Dict, Any
import re

def format_structured_response(response: str, agent_type: str, query: str = "") -> str:
    """
    Format the response based on the question type and content.
    Formats responses line by line without automatically adding points.
    """
    # Return original response for coding agent
    if agent_type == "coding":
        return response
    
    # Check if the response contains code blocks
    if "```" in response:
        return response
    
    # Function to format numbered points
    def format_numbered_points(text):
        # Split text into lines
        lines = text.split('\n')
        formatted_lines = []
        current_number = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check if line starts with a number and markdown formatting
            number_match = re.match(r'^\d+\.\s*\*\*', line)
            if number_match:
                # Extract the content after the number and formatting
                content = line[number_match.end():].strip()
                # Remove markdown formatting
                content = re.sub(r'\*\*', '', content)
                formatted_lines.append(f"{current_number}. {content}")
                current_number += 1
            else:
                # If line doesn't start with a number, add it as is
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    # Check if response contains numbered points with markdown formatting
    if re.search(r'\d+\.\s*\*\*', response):
        return format_numbered_points(response)
    
    # Check if response is already structured with bullet points or numbered lists
    if re.search(r'^[\s]*[-•*]\s|^\d+\.\s', response, re.MULTILINE):
        # Just ensure proper line breaks without adding points
        lines = response.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)
    
    # For regular responses, just ensure proper line breaks
    # Split by sentences but don't add numbering
    sentences = re.split(r'(?<=[.!?])\s+', response)
    formatted_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            formatted_sentences.append(sentence)
    
    return '\n'.join(formatted_sentences)

def format_5w_response(response: str, agent_type: str) -> str:
    """
    Format the response in the 5W format (Who, What, When, Where, Why).
    This is kept for backward compatibility.
    """
    # Return original response for coding agent
    if agent_type == "coding":
        return response
    
    # Define regex patterns for 5W
    who_pattern = r"(who|whose|whom|person|people|entity|entities)"
    what_pattern = r"(what|which|thing|things|item|items|object|objects)"
    when_pattern = r"(when|time|date|period|duration|how long|how old)"
    where_pattern = r"(where|place|location|area|region|site|venue)"
    why_pattern = r"(why|reason|purpose|cause|motivation|explanation)"
    
    # Extract information for each W
    who_info = re.findall(who_pattern, response, re.IGNORECASE)
    what_info = re.findall(what_pattern, response, re.IGNORECASE)
    when_info = re.findall(when_pattern, response, re.IGNORECASE)
    where_info = re.findall(where_pattern, response, re.IGNORECASE)
    why_info = re.findall(why_pattern, response, re.IGNORECASE)
    
    # Format the response with 5W structure
    formatted_response = ""
    
    if who_info:
        formatted_response += f"👤 Who: {', '.join(set(who_info))}\n\n"
    
    if what_info:
        formatted_response += f"📝 What: {', '.join(set(what_info))}\n\n"
    
    if when_info:
        formatted_response += f"⏱️ When: {', '.join(set(when_info))}\n\n"
    
    if where_info:
        formatted_response += f"📍 Where: {', '.join(set(where_info))}\n\n"
    
    if why_info:
        formatted_response += f"❓ Why: {', '.join(set(why_info))}\n\n"
    
    # If no 5W information found, return the original response with a note
    if not formatted_response:
        return f"{response}\n\nNote: This response could not be structured in the 5W format."
    
    return formatted_response 