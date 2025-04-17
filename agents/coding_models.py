from pydantic import BaseModel, Field
from typing import List, Optional

class CodeResponse(BaseModel):
    """Structured response for code generation"""
    code: str = Field(..., description="The generated code")
    language: str = Field(..., description="Programming language of the code")
    imports: List[str] = Field(default_factory=list, description="Required imports")
    explanation: Optional[str] = Field(None, description="Brief explanation of the code")
    complexity: Optional[str] = Field(None, description="Time and space complexity if applicable")

class CodeExplanation(BaseModel):
    """Structured response for code explanation"""
    purpose: str = Field(..., description="Overall purpose of the code")
    components: List[str] = Field(..., description="Key components and their functions")
    algorithms: Optional[List[str]] = Field(None, description="Important algorithms or patterns used")
    improvements: Optional[List[str]] = Field(None, description="Potential improvements or optimizations")

class CodeDebug(BaseModel):
    """Structured response for code debugging"""
    issues: List[str] = Field(..., description="Identified issues or bugs")
    root_cause: str = Field(..., description="Root cause of the issues")
    fixes: List[str] = Field(..., description="Suggested fixes")
    prevention: List[str] = Field(..., description="Prevention tips for future") 