"""苍穹签到系统 — Pydantic 数据模型（请求/响应结构）。

用于 FastAPI 自动校验请求体并生成接口文档。
"""
from pydantic import BaseModel, Field


class StudentCreate(BaseModel):
    name: str = Field(..., min_length=1, description="学生姓名")
    student_no: str = Field(..., min_length=1, description="学号（唯一）")
    class_name: str = Field("", description="班级")


class Student(StudentCreate):
    id: int
    created_at: str
    faces_total: int = 0


class CourseCreate(BaseModel):
    name: str = Field(..., min_length=1, description="课程名称")
    class_name: str = Field("", description="班级")


class Course(CourseCreate):
    id: int
    created_at: str


class FaceRegister(BaseModel):
    image: str = Field(..., description="Base64 编码的人脸照片（可含 data:image 前缀）")


class ProbeRequest(BaseModel):
    """签到前的单帧探测：只看有没有可识别的人脸，绝不写签到。"""
    image: str = Field(..., description="Base64 编码的单帧画面")
    course_id: int = Field(..., description="课程 ID")


class ChallengeRequest(BaseModel):
    course_id: int = Field(..., description="课程 ID")


class VerifyRequest(BaseModel):
    """一次签到的完整提交：一次性令牌 + 连续帧序列 + 指令前的基线帧。"""
    course_id: int = Field(..., description="课程 ID")
    challenge_id: str = Field(..., min_length=8, description="服务端签发的一次性挑战令牌")
    frames: list[str] = Field(..., min_length=1, description="按时间顺序的 Base64 帧序列")
    ref: str | None = Field(
        default=None,
        description="显示动作指令**之前**拍的一帧，用作动作判据的中性基线。"
                    "缺失时服务端退回序列首帧（动作占满整段时可能误判）。",
    )


class ManualAttendance(BaseModel):
    """手工补签（老师在本机操作，用于动作校验连续失败时兜底）。"""
    course_id: int = Field(..., description="课程 ID")
    student_id: int = Field(..., description="学生 ID")
