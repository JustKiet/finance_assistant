from pydantic import BaseModel, Field
from typing import List, Optional, Literal
import json

class CashFlowCumulativeUnit(BaseModel):
    year: int = Field(..., description="Year")
    value: float = Field(..., description="Value, if the value is wrapped in parentheses, it is negative")

class CashFlowData(BaseModel):
    id: int = Field(..., description="ID")
    subtitle: int = Field(..., description="Subtitle")
    ytd_cumulative: List[CashFlowCumulativeUnit] = Field(..., description="Year To Date Cumulative Values")

class CashFlowBulletPoint(BaseModel):
    bullet_point: str = Field(..., description="Bullet Point Title")
    bullet_data: CashFlowData = Field(..., description="Bullet Data Values")

class CashFlowSubSection(BaseModel):
    subsection_title: str = Field(..., description="Subsection Title")
    subsection_data: List[CashFlowBulletPoint] = Field(..., description="Subsection Data")

class CashFlowSection(BaseModel):
    section_title: str = Field(..., description="Section Title")
    section_data: List[CashFlowSubSection] = Field(..., description="Section Data")
    net_cash_flow: CashFlowData = Field(..., description="Net Cash Flow")

class SeperateCashFlowStatement(BaseModel):
    title: str = Field(..., description="Title of the Cash Flow Statement")
    description: str = Field(..., description="Description of the Cash Flow Statement")
    period_end_date: str = Field(..., description="Period End Date")
    calculation_unit: str = Field(..., description="Calculation Unit used in the Cash Flow Statement")
    quarter: Literal["Q1", "Q2", "Q3", "Q4"] = Field(..., description="Quarter")
    year: int = Field(..., description="Year of the Cash Flow Statement")
    data: List[CashFlowSection] = Field(..., description="Data within the Cash Flow Statement")

#print(json.dumps(SeperateCashFlowStatement.model_json_schema(), indent=4))