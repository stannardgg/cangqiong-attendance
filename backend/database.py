"""苍穹签到系统 — SQLite 数据层。

使用 Python 标准库 sqlite3，零额外依赖。
人脸特征向量以 BLOB 形式存储（numpy float32 数组序列化为字节）。

所有数据库操作通过 db_session() 上下文管理器执行，保证连接在
正常或异常情况下都被正确提交/回滚/关闭，避免「database is locked」。
"""
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "attendance.db")


def get_conn() -> sqlite3.Connection:
    """获取数据库连接（自动建 data 目录，设置外键与锁等待）。"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row  # 让查询结果支持按列名访问
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


@contextmanager
def db_session():
    """事务上下文：正常退出提交，异常回滚，最终必定关闭连接。"""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """建表（幂等，可重复执行）。"""
    with db_session() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                student_no  TEXT NOT NULL UNIQUE,
                class_name  TEXT DEFAULT '',
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS courses (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                class_name  TEXT DEFAULT '',
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS faces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                embedding   BLOB NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                course_id   INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
                student_id  INTEGER NOT NULL REFERENCES students(id) ON DELETE CASCADE,
                date        TEXT NOT NULL,
                check_in_at TEXT NOT NULL,
                status      TEXT NOT NULL DEFAULT 'present',
                UNIQUE(course_id, student_id, date)
            );
            """
        )


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _row_to_dict(row: sqlite3.Row) -> dict:
    return dict(row)


# ---- 学生 CRUD ----

def create_student(name: str, student_no: str, class_name: str = "") -> dict:
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO students (name, student_no, class_name, created_at) VALUES (?, ?, ?, ?)",
            (name, student_no, class_name, now_str()),
        )
        row = conn.execute("SELECT * FROM students WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _row_to_dict(row)


def list_students(class_name: str | None = None) -> list[dict]:
    with db_session() as conn:
        sql = """
            SELECT s.*, (SELECT COUNT(*) FROM faces f WHERE f.student_id = s.id) AS faces_total
            FROM students s
        """
        if class_name:
            sql += " WHERE s.class_name = ?"
            rows = conn.execute(sql + " ORDER BY s.student_no", (class_name,)).fetchall()
        else:
            rows = conn.execute(sql + " ORDER BY s.student_no").fetchall()
        return [_row_to_dict(r) for r in rows]


def get_student(student_id: int) -> dict | None:
    with db_session() as conn:
        row = conn.execute(
            """
            SELECT s.*, (SELECT COUNT(*) FROM faces f WHERE f.student_id = s.id) AS faces_total
            FROM students s WHERE s.id = ?
            """,
            (student_id,),
        ).fetchone()
        return _row_to_dict(row) if row else None


def update_student(student_id: int, name: str | None = None,
                   student_no: str | None = None, class_name: str | None = None) -> bool:
    fields, values = [], []
    if name is not None:
        fields.append("name = ?"); values.append(name)
    if student_no is not None:
        fields.append("student_no = ?"); values.append(student_no)
    if class_name is not None:
        fields.append("class_name = ?"); values.append(class_name)
    if not fields:
        return False
    values.append(student_id)
    with db_session() as conn:
        cur = conn.execute(f"UPDATE students SET {', '.join(fields)} WHERE id = ?", values)
        return cur.rowcount > 0


def delete_student(student_id: int) -> bool:
    with db_session() as conn:
        cur = conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        return cur.rowcount > 0


# ---- 课程 CRUD ----

def create_course(name: str, class_name: str = "") -> dict:
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO courses (name, class_name, created_at) VALUES (?, ?, ?)",
            (name, class_name, now_str()),
        )
        row = conn.execute("SELECT * FROM courses WHERE id = ?", (cur.lastrowid,)).fetchone()
        return _row_to_dict(row)


def list_courses() -> list[dict]:
    with db_session() as conn:
        rows = conn.execute("SELECT * FROM courses ORDER BY id DESC").fetchall()
        return [_row_to_dict(r) for r in rows]


def get_course(course_id: int) -> dict | None:
    with db_session() as conn:
        row = conn.execute("SELECT * FROM courses WHERE id = ?", (course_id,)).fetchone()
        return _row_to_dict(row) if row else None


def update_course(course_id: int, name: str | None = None, class_name: str | None = None) -> bool:
    fields, values = [], []
    if name is not None:
        fields.append("name = ?"); values.append(name)
    if class_name is not None:
        fields.append("class_name = ?"); values.append(class_name)
    if not fields:
        return False
    values.append(course_id)
    with db_session() as conn:
        cur = conn.execute(f"UPDATE courses SET {', '.join(fields)} WHERE id = ?", values)
        return cur.rowcount > 0


def delete_course(course_id: int) -> bool:
    with db_session() as conn:
        cur = conn.execute("DELETE FROM courses WHERE id = ?", (course_id,))
        return cur.rowcount > 0


# ---- 人脸特征 ----

def add_face(student_id: int, embedding: np.ndarray) -> int:
    """为学生保存一条 512 维人脸特征（float32 序列化为 BLOB），返回特征记录 id。"""
    blob = np.asarray(embedding, dtype=np.float32).tobytes()
    with db_session() as conn:
        cur = conn.execute(
            "INSERT INTO faces (student_id, embedding, created_at) VALUES (?, ?, ?)",
            (student_id, blob, now_str()),
        )
        return cur.lastrowid


def list_face_embeddings(student_id: int) -> list[np.ndarray]:
    """返回某学生的全部人脸特征（512 维 numpy 数组列表）。"""
    with db_session() as conn:
        rows = conn.execute(
            "SELECT embedding FROM faces WHERE student_id = ?", (student_id,)
        ).fetchall()
        return [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]


def list_all_face_embeddings() -> list[tuple[int, np.ndarray]]:
    """返回全部人脸特征 [(student_id, embedding)]，用于签到比对。"""
    with db_session() as conn:
        rows = conn.execute("SELECT student_id, embedding FROM faces").fetchall()
        return [(r["student_id"], np.frombuffer(r["embedding"], dtype=np.float32)) for r in rows]


def count_faces(student_id: int) -> int:
    with db_session() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM faces WHERE student_id = ?", (student_id,)).fetchone()
        return row["c"]


def delete_faces(student_id: int) -> int:
    """删除某学生全部人脸特征，返回删除条数。"""
    with db_session() as conn:
        cur = conn.execute("DELETE FROM faces WHERE student_id = ?", (student_id,))
        return cur.rowcount


# ---- 签到记录 ----

def mark_attendance(course_id: int, student_id: int, status: str = "present") -> bool:
    """记录出勤（同课程同日期同学生唯一），返回是否为新记录。"""
    with db_session() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO attendance (course_id, student_id, date, check_in_at, status) "
            "VALUES (?, ?, ?, ?, ?)",
            (course_id, student_id, today_str(), now_str(), status),
        )
        return cur.rowcount > 0


def list_attendance(course_id: int, date: str | None = None) -> list[dict]:
    """返回某课程某日期的签到记录（含学生姓名/学号/班级）。"""
    date = date or today_str()
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.course_id, a.student_id, a.date, a.check_in_at, a.status,
                   s.name, s.student_no, s.class_name
            FROM attendance a
            JOIN students s ON s.id = a.student_id
            WHERE a.course_id = ? AND a.date = ?
            ORDER BY a.check_in_at
            """,
            (course_id, date),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]


def clear_attendance(course_id: int, date: str | None = None) -> int:
    """清空某课程的签到记录（指定日期则只清该日），返回删除条数。

    用于「重新签到」场景：清掉后学生可再次被识别为出勤。
    """
    with db_session() as conn:
        if date:
            cur = conn.execute(
                "DELETE FROM attendance WHERE course_id = ? AND date = ?",
                (course_id, date),
            )
        else:
            cur = conn.execute("DELETE FROM attendance WHERE course_id = ?", (course_id,))
        return cur.rowcount
