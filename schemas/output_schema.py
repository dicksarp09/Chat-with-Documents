from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ActionItem(BaseModel):
    task: str = Field(..., description="The action task to be performed")
    priority: Priority = Field(..., description="Priority level of the action")
    reason: str = Field(..., description="Why this action is needed")
    evidence: List[str] = Field(
        default_factory=list, description="Node IDs providing evidence"
    )


class RiskItem(BaseModel):
    description: str = Field(..., description="Description of the risk")
    severity: str = Field(..., description="Severity level: high, medium, low")
    category: str = Field(..., description="Category of the risk")
    evidence: List[str] = Field(
        default_factory=list, description="Node IDs providing evidence"
    )


class ObligationItem(BaseModel):
    description: str = Field(..., description="Description of the obligation")
    party: str = Field(..., description="Party responsible")
    deadline: Optional[str] = Field(None, description="Deadline if specified")
    evidence: List[str] = Field(
        default_factory=list, description="Node IDs providing evidence"
    )


class KeyPoint(BaseModel):
    text: str = Field(..., description="The key point text")
    category: str = Field(..., description="Category of the key point")
    evidence: List[str] = Field(
        default_factory=list, description="Node IDs providing evidence"
    )


class DocumentSummary(BaseModel):
    summary: str = Field(..., description="Overall document summary")
    document_type: Optional[str] = Field(
        None, description="Type of document if identifiable"
    )
    key_theme: Optional[str] = Field(None, description="Main theme or subject")
    evidence: List[str] = Field(
        default_factory=list, description="Node IDs providing evidence"
    )


class DocumentAnalysisOutput(BaseModel):
    summary: str = Field(..., description="Document summary")
    key_points: List[KeyPoint] = Field(
        default_factory=list, description="Extracted key points"
    )
    risks: List[RiskItem] = Field(default_factory=list, description="Identified risks")
    obligations: List[ObligationItem] = Field(
        default_factory=list, description="Extracted obligations"
    )
    actions: List[ActionItem] = Field(
        default_factory=list, description="Suggested actions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class QueryOutput(BaseModel):
    answer: str = Field(..., description="Generated answer based on query")
    summary: str = Field(..., description="Summary of relevant context")
    key_points: List[KeyPoint] = Field(
        default_factory=list, description="Key points from context"
    )
    risks: List[RiskItem] = Field(
        default_factory=list, description="Risks found in context"
    )
    obligations: List[ObligationItem] = Field(
        default_factory=list, description="Obligations in context"
    )
    actions: List[ActionItem] = Field(
        default_factory=list, description="Suggested actions based on context"
    )
    sources: List[str] = Field(default_factory=list, description="Source node IDs used")


class SectionData(BaseModel):
    title: str = ""
    level: int = 1
    content: str = ""
    children: List[Dict[str, Any]] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    sections: List[SectionData] = Field(default_factory=list)
    full_text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChunkNode(BaseModel):
    id: str
    text: str
    level: int
    doc_id: str
    section_title: str = ""
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentStructure(BaseModel):
    doc_id: str
    nodes: List[ChunkNode] = Field(default_factory=list)
    total_chunks: int = 0


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    file_type: str
    uploaded_at: datetime
    chunk_count: int
    status: str = "processed"


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    chunk_count: int
    message: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    doc_id: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class AnalyzeRequest(BaseModel):
    doc_id: Optional[str] = None
    file_path: Optional[str] = None


class UploadResponse(BaseModel):
    dataset_id: str = Field(..., min_length=1)
    type: str = Field(..., pattern="^(csv|document)$")
    success: bool = True
    status: str = "processed"
    message: str = ""

    class Config:
        json_schema_extra = {
            "example": {
                "dataset_id": "ds_abc123",
                "type": "document",
                "success": True,
                "status": "processed",
                "message": "",
            }
        }


class QueryResponse(BaseModel):
    answer: str = Field(..., min_length=1)
    sources: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def validate_not_empty_answer(cls, v):
        if not v or not v.strip():
            raise ValueError("Answer cannot be empty")
        return v.strip()


class APIError(BaseModel):
    error: str
    detail: Optional[str] = None
    type: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "detail": "Query must be at least 1 character",
                "type": "validation_error",
            }
        }
