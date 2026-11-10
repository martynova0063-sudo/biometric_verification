from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class HealthStatus(BaseModel):
    status: str
    service: str
    version: str


class PasswordLoginInput(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


class PasswordLoginToken(BaseModel):
    access_token: str
    token_type: Literal["bearer"]
    expires_in: int


class EmployeeResponse(BaseModel):
    guid: str
    personnel_number: str
    surname: str
    given_name: str
    patronymic: str
    organization: str
    is_test: bool
    has_photo: bool
    photo_url: str | None


class EmployeePhotoUploadUrlInput(BaseModel):
    size: int = Field(gt=0, le=5 * 1024 * 1024)
    content_type: Literal["image/jpeg"]


class EmployeePhotoUploadUrlResponse(BaseModel):
    upload_url: str
    object_path: str


class EmployeePhotoFinalizeInput(BaseModel):
    object_path: str = Field(min_length=1, max_length=500)


class DashboardSummary(BaseModel):
    total_events: int
    health_id_today: int
    verification_pass_rate: float
    open_review_tasks: int
    data_quality: float
    model_version: str
    categories: dict[str, int]


class ActivityItemResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    type: str
    title: str
    description: str
    created_at: datetime
    status: str


class HealthMeasurements(BaseModel):
    sleep_hours: float = Field(ge=0, le=24)
    stress_score: float = Field(ge=0, le=10)
    activity_minutes: float = Field(ge=0)
    resting_pulse: float = Field(ge=20, le=240)
    respiratory_rate: float = Field(ge=4, le=60)
    verification_quality: float = Field(ge=0, le=1)


class HealthContext(BaseModel):
    age_group: str | None = None
    has_chronic_conditions: bool | None = None
    work_shift: str | None = None


class HistoryItem(BaseModel):
    event_date: datetime
    health_id: float = Field(ge=0, le=1)


class HealthCalculationInput(BaseModel):
    employee_ref: str = Field(min_length=1)
    event_date: datetime
    measurements: HealthMeasurements
    context: HealthContext | None = None
    history: list[HistoryItem] = []


class EligibleHealthEmployeeResponse(BaseModel):
    employee_ref: str
    full_name: str
    organization: str
    is_test: bool
    verification_session_id: UUID
    verification_status: Literal["verified", "manual_review", "not_verified", "not_evaluable", "insufficient_quality"]
    eligibility_source: Literal["system", "manual"]
    verified_at: datetime


class PersonalBaselineMetricResponse(BaseModel):
    key: Literal[
        "health_id",
        "sleep_hours",
        "stress_score",
        "activity_minutes",
        "resting_pulse",
        "respiratory_rate",
    ]
    label: str
    unit: str
    sample_count: int
    ready: bool
    median: float | None
    iqr: float | None
    lower_bound: float | None
    upper_bound: float | None


class PersonalBaselineResponse(BaseModel):
    employee_ref: str
    window_start: datetime
    window_end: datetime
    window_days: int
    minimum_samples: int
    metrics: list[PersonalBaselineMetricResponse]


class HealthEvidence(BaseModel):
    key: str
    label: str
    value: str
    direction: Literal["positive", "neutral", "negative"]


class HealthResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    employee_ref: str
    event_date: datetime
    health_id: float
    category: Literal["stable", "attention", "review"]
    status: Literal["calculated", "review_required", "reviewed"]
    model_version: str
    components: dict[str, float]
    evidence: list[HealthEvidence]
    disclaimer: str
    created_at: datetime


class PersonalBaselineEvaluationResponse(BaseModel):
    status: Literal["outside_corridor", "within_corridor", "insufficient_data"]
    evaluated_metric_count: int
    total_metric_count: int
    outside_metrics: list[
        Literal[
            "health_id",
            "sleep_hours",
            "stress_score",
            "activity_minutes",
            "resting_pulse",
            "respiratory_rate",
        ]
    ]


class HealthResultListResponse(HealthResultResponse):
    personal_baseline: PersonalBaselineEvaluationResponse


class EmbeddingComparisonMean(BaseModel):
    reference: float | None
    video: float | None
    delta: float | None
    centroid_cosine_similarity: float | None


class EmbeddingComparisonVariance(BaseModel):
    reference: float | None
    video: float | None
    ratio: float | None


class EmbeddingComparisonOutliers(BaseModel):
    count: int | None
    total: int
    threshold: float | None


class EmbeddingComparisonDistribution(BaseModel):
    bin_centers: list[float]
    reference_counts: list[int]
    video_counts: list[int]


class EmbeddingComparisonResponse(BaseModel):
    status: Literal[
        "available",
        "insufficient_reference",
        "no_reference",
        "unavailable",
        "no_video_vectors",
    ]
    method: str
    reference_vectors: int
    video_vectors: int
    dimensions: int
    mean: EmbeddingComparisonMean
    variance: EmbeddingComparisonVariance
    outliers: EmbeddingComparisonOutliers
    distribution: EmbeddingComparisonDistribution
    message: str


class EmbeddingInterpretationRequest(BaseModel):
    refresh: bool = False


class EmbeddingInterpretationResponse(BaseModel):
    analysis_version: int
    verdict: Literal["consistent", "borderline", "discordant", "insufficient_data"]
    summary: str
    findings: list[str]
    recommendations: list[str]
    generated_at: datetime
    model: str
    disclaimer: str


class VerificationSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    employee_ref: str
    status: Literal["verified", "manual_review", "not_verified", "not_evaluable", "insufficient_quality"]
    match_score: float | None
    liveness_score: float | None
    quality_score: float | None
    decision: Literal["match", "no_match", "review"]
    algorithm_version: str
    created_at: datetime
    audit_ref: str
    attack_type: str | None
    is_match: bool | None
    ground_truth_match: bool | None
    is_live: bool | None
    ground_truth_liveness: bool | None
    cosine_distance: float | None
    embedding_comparison: EmbeddingComparisonResponse | None = None
    embedding_interpretation: EmbeddingInterpretationResponse | None = None
    arcface_video_enrollment: dict[str, Any] | None = None
    verification_elapsed_seconds: float | None = None


class VerificationSessionDetailResponse(VerificationSessionResponse):
    verification_id: UUID
    event_id: str
    worker_pseudonym: str
    model_version: str
    config_hash: str
    face_match: dict[str, Any]
    liveness: dict[str, Any]
    quality: dict[str, Any]
    route: str
    health_id: dict[str, Any]
    review_tasks: list[dict[str, Any]]
    disclaimer: str
    not_evaluable_reason: str | None = None
    recommended_action: str | None = None


class VerificationReviewSessionResponse(BaseModel):
    id: UUID
    employee_ref: str
    status: Literal["verified", "manual_review", "not_verified", "not_evaluable", "insufficient_quality"]
    match_score: float | None
    liveness_score: float | None
    quality_score: float | None
    decision: Literal["match", "no_match", "review"]


class VerificationReviewQueueItemResponse(BaseModel):
    verification_session: VerificationReviewSessionResponse


class VerificationSessionStatusResponse(BaseModel):
    session_id: UUID
    status: Literal["verified", "manual_review", "not_verified", "not_evaluable", "insufficient_quality"]
    decision: Literal["match", "no_match", "review"]
    is_match: bool | None
    is_live: bool | None
    ground_truth_match: bool | None
    ground_truth_liveness: bool | None
    review_status: Literal["open", "in_review", "resolved"] | None
    review_resolution: Literal["approve", "reject", "request_retry"] | None
    reviewer_note: str | None


class VideoReceiptCompleteResponse(BaseModel):
    session_id: UUID
    video_received: bool
    completed_at: datetime


class ReviewTaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    kind: Literal["verification", "health_id"]
    subject_ref: str
    status: Literal["open", "in_review", "resolved"]
    priority: Literal["low", "medium", "high"]
    summary: str
    created_at: datetime
    resolved_at: datetime | None
    resolution: str | None
    reviewer_note: str | None
    verification_session_id: UUID | None


class ReviewInput(BaseModel):
    resolution: Literal["approve", "reject", "request_retry"]
    reviewer_note: str = ""
    ground_truth_match: bool | None = None
    ground_truth_liveness: bool | None = None


class VerificationReviewDecisionInput(BaseModel):
    status: Literal["approve", "reject", "request_retry"]
    ground_truth_match: bool | None = None
    ground_truth_liveness: bool | None = None
    comment: str = ""


class BiometricMetricCounts(BaseModel):
    tp: int = Field(description="True Positive: модель приняла и оператор подтвердил совпадение.")
    fp: int = Field(description="False Positive: модель приняла, но оператор указал другого человека.")
    fn: int = Field(description="False Negative: модель отклонила, но оператор подтвердил совпадение.")
    tn: int = Field(description="True Negative: модель отклонила и оператор подтвердил другого человека.")
    genuine_cases: int
    impostor_cases: int
    labeled_cases: int


class LivenessMetricCounts(BaseModel):
    spoof_cases: int
    spoof_accepted_as_live: int
    spoof_rejected: int
    bona_fide_cases: int
    bona_fide_rejected_as_spoof: int
    bona_fide_accepted: int
    labeled_cases: int


class MetricCalculationDetails(BaseModel):
    biometric: BiometricMetricCounts
    liveness: LivenessMetricCounts


class BiometricMetricsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    calculation_date: date
    window_start: datetime
    window_end: datetime
    far: float
    frr: float
    cer: float
    apcer: float
    bpcer: float
    acer: float
    total_samples: int
    successful_matches: int
    failed_matches: int
    spoof_attempts_detected: int
    live_rejected_as_spoof: int
    calculation_details: MetricCalculationDetails | None = None
    created_at: datetime


class ModelComponent(BaseModel):
    key: str
    label: str
    weight: float
    source: str


class ModelCard(BaseModel):
    model_id: str
    model_name: str
    version: str
    status: str
    purpose: str
    change_summary: str
    disclaimer: str
    thresholds: dict[str, float]
    components: list[ModelComponent]
    limitations: list[str]
    parameters: dict[str, Any]
    parameter_fields: list[dict[str, Any]]


class ModelVersion(BaseModel):
    id: str | None = None
    model_name: str | None = None
    version: str
    released_at: datetime
    status: str
    change_summary: str


class ModelRegistryInput(BaseModel):
    model_name: str = Field(min_length=1, max_length=120)
    version: str = Field(min_length=1, max_length=60)
    purpose: str = Field(min_length=1, max_length=2000)
    change_summary: str = Field(min_length=1, max_length=2000)
    parameters: dict[str, Any]


class ModelRegistryResponse(BaseModel):
    id: str
    model_name: str
    version: str
    status: str
    purpose: str
    change_summary: str
    parameters: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class DataDriftMetricResponse(BaseModel):
    key: str
    label: str
    unit: str
    baseline_samples: int
    current_samples: int
    psi: float | None
    status: Literal["collecting", "insufficient_data", "stable", "warning", "critical"]
    baseline_available: bool


class DataDriftSummaryResponse(BaseModel):
    status: Literal["collecting", "insufficient_data", "stable", "warning", "critical"]
    baseline_status: Literal["collecting", "ready"]
    manual_review_mode: bool
    manual_review_reason: str | None = None
    needs_recalibration: bool
    collection_started_at: datetime | None = None
    baseline_window_start: datetime | None = None
    baseline_window_end: datetime | None = None
    baseline_created_at: datetime | None = None
    baseline_camera_models: list[dict[str, Any]]
    current_camera_models: list[dict[str, Any]]
    baseline_mean_brightness: float | None = None
    current_mean_brightness: float | None = None
    observations_total: int
    baseline_days_elapsed: int
    baseline_days_required: int
    analysis_window_start: datetime | None = None
    analysis_window_end: datetime | None = None
    minimum_samples: int
    target_far: float
    current_match_threshold: float
    recalibration_status: str
    proposed_match_threshold: float | None = None
    achieved_far: float | None = None
    achieved_frr: float | None = None
    labelled_genuine_samples: int
    labelled_impostor_samples: int
    last_run_at: datetime | None = None
    alert_messages: list[str]
    metrics: list[DataDriftMetricResponse]


class DataDriftRunResponse(BaseModel):
    id: int
    calculation_date: date
    window_start: datetime
    window_end: datetime
    status: Literal["collecting", "insufficient_data", "stable", "warning", "critical"]
    psi_values: dict[str, float | None]
    sample_counts: dict[str, int]
    needs_recalibration: bool
    manual_review_mode: bool
    alert_messages: list[str]
    recalibration: dict[str, Any]
    created_at: datetime


class DataDriftControlInput(BaseModel):
    enabled: bool
    reviewer_note: str = Field(min_length=1, max_length=500)