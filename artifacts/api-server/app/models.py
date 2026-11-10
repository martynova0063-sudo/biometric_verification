from datetime import date, datetime
from uuid import uuid4

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, JSON, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class Employee(Base):
    __tablename__ = "employees"

    guid: Mapped[str] = mapped_column(String(36), primary_key=True)
    personnel_number: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    sex: Mapped[str] = mapped_column(String(20))
    surname: Mapped[str] = mapped_column(String(120), index=True)
    given_name: Mapped[str] = mapped_column(String(120))
    patronymic: Mapped[str] = mapped_column(String(120))
    birth_date: Mapped[str] = mapped_column(String(80))
    organization: Mapped[str] = mapped_column(String(200), index=True)
    is_photo: Mapped[bool] = mapped_column(Boolean, default=False)
    is_test: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    photo_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    sid_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    first_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    last_name: Mapped[str | None] = mapped_column(String(160), nullable=True)
    profile_image_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class EmployeePhotoUpload(Base):
    __tablename__ = "employee_photo_uploads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    employee_guid: Mapped[str] = mapped_column(String(36), index=True)
    user_id: Mapped[str] = mapped_column(String(255), index=True)
    object_path: Mapped[str] = mapped_column(Text, unique=True)
    content_type: Mapped[str] = mapped_column(String(80))
    expected_size: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)


class EmployeePhotoArchive(Base):
    __tablename__ = "employee_photo_archives"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    employee_guid: Mapped[str] = mapped_column(
        String(36), ForeignKey("employees.guid", ondelete="RESTRICT"), index=True
    )
    source_photo_path: Mapped[str] = mapped_column(Text)
    object_path: Mapped[str] = mapped_column(Text, unique=True)
    archived_by: Mapped[str] = mapped_column(String(255), index=True)
    content_type: Mapped[str] = mapped_column(String(80))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class HealthResult(Base):
    __tablename__ = "health_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    employee_ref: Mapped[str] = mapped_column(String(120), index=True)
    event_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    health_id: Mapped[float] = mapped_column(Float)
    category: Mapped[str] = mapped_column(String(30), index=True)
    status: Mapped[str] = mapped_column(String(30), default="calculated")
    model_version: Mapped[str] = mapped_column(String(40))
    components: Mapped[dict] = mapped_column(JSON)
    measurements: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    evidence: Mapped[list] = mapped_column(JSON)
    disclaimer: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)


class VerificationSession(Base):
    __tablename__ = "verification_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    employee_ref: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True)
    match_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    liveness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    decision: Mapped[str] = mapped_column(String(30))
    algorithm_version: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    audit_ref: Mapped[str] = mapped_column(String(80))
    attack_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    is_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ground_truth_match: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    is_live: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    ground_truth_liveness: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    cosine_distance: Mapped[float | None] = mapped_column(Float, nullable=True)

    @property
    def embedding_comparison(self) -> dict | None:
        details = self.details if isinstance(self.details, dict) else {}
        comparison = details.get("embedding_comparison")
        return comparison if isinstance(comparison, dict) else None

    @property
    def embedding_interpretation(self) -> dict | None:
        details = self.details if isinstance(self.details, dict) else {}
        interpretation = details.get("embedding_interpretation")
        return interpretation if isinstance(interpretation, dict) else None

    @property
    def verification_elapsed_seconds(self) -> float | None:
        details = self.details if isinstance(self.details, dict) else {}
        elapsed = details.get("verification_elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            return float(elapsed)
        return None


class BiometricMetricsLog(Base):
    __tablename__ = "biometric_metrics_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    calculation_date: Mapped[date] = mapped_column(Date, nullable=False)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    far: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    frr: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    cer: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    apcer: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    bpcer: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    acer: Mapped[float] = mapped_column(Numeric(5, 4), nullable=False)
    total_samples: Mapped[int] = mapped_column(Integer, nullable=False)
    successful_matches: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_matches: Mapped[int] = mapped_column(Integer, nullable=False)
    spoof_attempts_detected: Mapped[int] = mapped_column(Integer, nullable=False)
    live_rejected_as_spoof: Mapped[int] = mapped_column(Integer, nullable=False)
    calculation_details: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class ReviewTask(Base):
    __tablename__ = "review_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    kind: Mapped[str] = mapped_column(String(30), index=True)
    subject_ref: Mapped[str] = mapped_column(String(120), index=True)
    status: Mapped[str] = mapped_column(String(30), default="open", index=True)
    priority: Mapped[str] = mapped_column(String(20), default="medium")
    summary: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    resolution: Mapped[str | None] = mapped_column(String(40), nullable=True)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    verification_session_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)


class ActivityItem(Base):
    __tablename__ = "activity_items"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    type: Mapped[str] = mapped_column(String(30))
    title: Mapped[str] = mapped_column(String(180))
    description: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    status: Mapped[str] = mapped_column(String(30))


class RegisteredModel(Base):
    __tablename__ = "registered_models"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    model_name: Mapped[str] = mapped_column(String(120), nullable=False)
    version: Mapped[str] = mapped_column(String(60), nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String(30), nullable=False, default="draft", index=True)
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    change_summary: Mapped[str] = mapped_column(Text, nullable=False)
    parameters: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class DataDriftObservation(Base):
    __tablename__ = "data_drift_observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(30), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    is_test: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)
    metrics: Mapped[dict] = mapped_column(JSON, nullable=False)
    capture_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class DataDriftBaseline(Base):
    __tablename__ = "data_drift_baselines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    distributions: Mapped[dict] = mapped_column(JSON, nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DataDriftRun(Base):
    __tablename__ = "data_drift_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    calculation_date: Mapped[date] = mapped_column(Date, unique=True, nullable=False, index=True)
    window_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    window_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[str] = mapped_column(String(30), nullable=False)
    psi_values: Mapped[dict] = mapped_column(JSON, nullable=False)
    sample_counts: Mapped[dict] = mapped_column(JSON, nullable=False)
    needs_recalibration: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    manual_review_mode: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    alert_messages: Mapped[list] = mapped_column(JSON, nullable=False)
    recalibration: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)


class DataDriftControl(Base):
    __tablename__ = "data_drift_control"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    manual_review_mode: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    manual_review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)