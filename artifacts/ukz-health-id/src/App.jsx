import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import './health-id-eligibility.css';
import './verification.css';
import './arcface-recognition.css';
import './metrics-review.css';
import DriftPage from './DataDriftPage';
import EmbeddingsIndexPage from './EmbeddingsIndexPage.jsx';
import ModelRegistryPanel from './ModelRegistryPanel';
import UserGuidePage from './UserGuidePage';
import { QueryClient, QueryClientProvider, useQuery, useQueryClient } from '@tanstack/react-query';
import { Link, Route, Router as WouterRouter, Switch, useLocation, useParams } from 'wouter';
import {
  Activity, AlertTriangle, ArrowDownRight, ArrowUpRight, Bell, BookOpen, Camera, Check, ChevronRight,
  CircleStop, ClipboardCheck, Clock3, Copy, Database, Download, FileCheck2, Fingerprint, Gauge, HeartPulse,
  History, LayoutDashboard, LogOut, Menu, Microscope, PanelLeftClose, Play, Plus, RefreshCw,
  Search, Settings2, ShieldCheck, SlidersHorizontal, Sparkles, Trash2, UserRound, X,
} from 'lucide-react';
import {
  getGetDashboardSummaryQueryKey, getGetHealthResultQueryKey, getGetPersonalHealthBaselineQueryKey, getGetVerificationSessionQueryKey, getHealthCheckQueryKey,
  getListActivityQueryKey,
  getListHealthResultsQueryKey, getListReviewTasksQueryKey, getListVerificationMetricsQueryKey, getListVerificationSessionsQueryKey,
  convertVerificationCameraVideo, useCalculateHealthId, useCalculateVerificationMetrics, useCreateVerificationSession, useDeleteVerificationSession,
  useCompleteVerificationSessionVideo,
  useGetDashboardSummary, useGetHealthResult, useGetModelCard, useGetPersonalHealthBaseline, useGetVerificationSession,
  useHealthCheck, useListActivity, useListHealthResults, useListModelVersions, useRequestVerificationReview,
  useListEligibleHealthEmployees, useListReviewTasks, useListVerificationMetrics, useListVerificationSessions, useResolveReviewTask,
} from '@workspace/api-client-react';
import { useAuth } from '@workspace/replit-auth-web';

const queryClient = new QueryClient();
const ReviewMediaContext = createContext({ reviewMedia: {}, retainReviewMedia: () => {}, releaseReviewMedia: () => {} });

const navItems = [
  { href: '/', label: 'Обзор', sub: 'Состояние системы', icon: LayoutDashboard },
  { href: '/health-results', label: 'HEALTH_ID', sub: 'Исследовательские результаты', icon: HeartPulse },
  { href: '/verification', label: 'Верификация', sub: 'Биометрические сессии', icon: Fingerprint },
  { href: '/embeddings', label: 'Эмбеддинги и индексация', sub: 'Улучшение качества распознавания', icon: Database },
  { href: '/review', label: 'Очередь проверки', sub: 'Внимание специалиста', icon: ClipboardCheck },
  { href: '/governance', label: 'Управление моделью', sub: 'Контроль модели', icon: ShieldCheck },
  { href: '/drift', label: 'Дрейф данных', sub: 'Стабильность сигналов', icon: Activity },
  { href: '/manual', label: 'Руководство', sub: 'Сценарии и API', icon: BookOpen },
];

const fmtDate = (value, withTime = true) => {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('ru-RU', withTime ? { day: '2-digit', month: 'short', hour: '2-digit', minute: '2-digit' } : { day: '2-digit', month: 'short', year: 'numeric' }).format(date);
};
const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const formatVerificationElapsed = (value) => {
  const seconds = Number(value);
  if (value == null || !Number.isFinite(seconds) || seconds < 0) return '—';
  if (seconds < 1) return `${Math.round(seconds * 1000)} мс`;
  if (seconds < 60) return `${seconds.toFixed(1)} с`;
  const totalSeconds = Math.round(seconds);
  return `${Math.floor(totalSeconds / 60)} мин ${totalSeconds % 60} с`;
};
const normalizedScore = (value) => {
  const score = Number(value);
  return Number.isFinite(score) ? Math.max(0, Math.min(1, score)) : 0;
};
const scoreBarColor = (value) => `hsl(${Math.round(normalizedScore(value) * 120)} 68% 42%)`;
const metricPercent = (value) => `${new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 }).format(Number(value || 0) * 100)}%`;
const compact = (value) => new Intl.NumberFormat('ru-RU', { notation: 'compact', maximumFractionDigits: 1 }).format(Number(value || 0));
const initials = (value = 'MR') => value.split(/[\s_-]+/).map((part) => part[0]).join('').slice(0, 2).toUpperCase();
const statusLabels = {
  calculated: 'рассчитан',
  review_required: 'требует проверки',
  reviewed: 'проверен',
  verified: 'подтверждён',
  manual_review: 'ручная проверка',
  not_verified: 'не подтверждён',
  not_evaluable: 'оценка невозможна',
  insufficient_quality: 'недостаточное качество',
  match: 'совпадение',
  no_match: 'нет совпадения',
  review: 'проверка',
  open: 'открыта',
  in_review: 'на проверке',
  resolved: 'закрыта',
  active: 'активна',
  current: 'текущая',
  draft: 'черновик',
  archived: 'архивная',
  research: 'исследовательская',
  approved: 'одобрено',
  pending: 'ожидает',
  high: 'высокий',
  medium: 'средний',
  low: 'низкий',
  approve: 'одобрить',
  reject: 'отклонить',
  request_retry: 'запросить повторную проверку',
  spoofing: 'подмена',
  replay: 'повторная запись',
  mask: 'маска',
  injection: 'подмена сигнала',
  stable: 'стабильная зона',
  success: 'успешно',
  attention: 'внимание',
  positive: 'положительный',
  neutral: 'нейтральный',
  negative: 'отрицательный',
};
const categoryLabels = { stable: 'Стабильно', attention: 'Внимание', review: 'Проверка', completeness: 'Полнота данных' };
const taskKindLabels = { verification: 'верификация', health_id: 'HEALTH_ID' };
const fieldLabels = {
  sleep_hours: 'Сон, часов',
  stress_score: 'Стресс, баллов',
  activity_minutes: 'Активность, минут',
  resting_pulse: 'Пульс в покое',
  respiratory_rate: 'Частота дыхания',
  verification_quality: 'Качество верификации',
  face_similarity: 'Сходство лица',
  blink_score: 'Сигнал моргания',
  head_motion_score: 'Движение головы',
  texture_score: 'Текстурный анализ',
  quality_score: 'Качество захвата',
  body: 'Физическое состояние',
  mental: 'Психоэмоциональное состояние',
  social: 'Активность и восстановление',
  completeness: 'Полнота данных',
  health_id_threshold: 'Порог HEALTH_ID',
  match_threshold: 'Порог совпадения',
  liveness_threshold: 'Порог liveness',
  review_threshold: 'Порог проверки',
  confidence_threshold: 'Порог уверенности',
  verification_threshold: 'Порог верификации',
};
const labelStatus = (value = '') => statusLabels[value.toLowerCase()] || value.replaceAll('_', ' ');
const labelCategory = (value = '') => categoryLabels[value] || statusLabels[value] || value.replaceAll('_', ' ');
const labelTaskKind = (value = '') => taskKindLabels[value] || value.replaceAll('_', ' ');
const labelField = (value = '') => fieldLabels[value] || value.replaceAll('_', ' ');
const personalBaselineMetricLabel = (key) => key === 'health_id' ? 'HEALTH_ID' : labelField(key);

function StatusPill({ children, tone = 'neutral' }) {
  return <span className={`status-pill status-${tone}`}><span className="status-dot" />{children}</span>;
}

function PersonalBaselineBadge({ evaluation, detailed = false }) {
  if (!evaluation) return null;
  const outside = evaluation.status === 'outside_corridor';
  const within = evaluation.status === 'within_corridor';
  const label = outside
    ? `Вне коридора · ${evaluation.outside_metrics.length}`
    : within
      ? `В коридоре · ${evaluation.evaluated_metric_count}/${evaluation.total_metric_count}`
      : 'Недостаточно данных';
  const className = `baseline-register-badge baseline-${evaluation.status}`;

  if (!detailed) return <span className={className}>{label}</span>;
  return <div className={`baseline-result-notice baseline-notice-${evaluation.status}`}>
    <strong>{label}</strong>
    {outside
      ? <small>За пределами: {evaluation.outside_metrics.map(personalBaselineMetricLabel).join(', ')}. Сравнение с 30 днями до даты сигнала.</small>
      : within
        ? <small>Сравнено показателей: {evaluation.evaluated_metric_count} из {evaluation.total_metric_count}. Окно: 30 дней до даты сигнала.</small>
        : <small>До даты сигнала недостаточно предыдущих наблюдений для сравнения.</small>}
  </div>;
}

function toneForStatus(value = '') {
  const normalized = value.toLowerCase();
  if (['verified', 'calculated', 'reviewed', 'resolved', 'active', 'match', 'approved', 'positive'].includes(normalized)) return 'good';
  if (['manual_review', 'not_evaluable', 'insufficient_quality', 'review_required', 'in_review', 'review', 'attention', 'pending'].includes(normalized)) return 'warn';
  if (['not_verified', 'rejected', 'negative'].includes(normalized)) return 'bad';
  return 'neutral';
}

function LoadingRows({ rows = 4 }) {
  return <div className="space-y-3" aria-label="Загрузка">
    {Array.from({ length: rows }).map((_, index) => <div key={index} className="skeleton-row" />)}
  </div>;
}

function EmptyState({ icon: Icon = Database, title, text, action }) {
  return <div className="empty-state">
    <div className="empty-icon"><Icon size={22} /></div>
    <h3>{title}</h3><p>{text}</p>{action}
  </div>;
}

function ErrorState({ onRetry }) {
  return <div className="empty-state error-state">
    <div className="empty-icon"><AlertTriangle size={22} /></div>
    <h3>Сигнал недоступен</h3><p>Сервис не вернул данные для этого раздела. Локальные выводы не делались.</p>
    {onRetry && <button className="button button-secondary" data-testid="button-retry" onClick={onRetry}><RefreshCw size={14} /> Повторить</button>}
  </div>;
}

function MetricCard({ label, value, detail, icon: Icon, accent = 'teal', trend }) {
  return <article className={`metric-card metric-${accent}`}>
    <div className="metric-top"><span className="eyebrow">{label}</span><span className="metric-icon"><Icon size={17} /></span></div>
    <strong className="display metric-value" data-testid={`metric-${label.toLowerCase().replaceAll(' ', '-')}`}>{value}</strong>
    <div className="metric-detail">{trend && <span className={trend > 0 ? 'trend-up' : 'trend-down'}>{trend > 0 ? <ArrowUpRight size={13} /> : <ArrowDownRight size={13} />}{Math.abs(trend)}%</span>}{detail}</div>
  </article>;
}

function Panel({ title, kicker, action, children, className = '' }) {
  return <section className={`panel soft-card ${className}`}>
    {(title || kicker || action) && <div className="panel-head"><div>{kicker && <div className="eyebrow">{kicker}</div>}{title && <h2 className="display panel-title">{title}</h2>}</div>{action}</div>}
    {children}
  </section>;
}

function ArcFaceEnrolledPhotoPreview({ employee }) {
  const [photoState, setPhotoState] = useState(employee.photo_url ? 'loading' : 'error');
  const employeeName = `${employee.surname} ${employee.given_name} ${employee.patronymic}`.trim();

  return <div className="arcface-saved-photo" role="status" data-testid="arcface-enrollment-reference-preview">
    {employee.photo_url && photoState !== 'error'
      ? <img
        src={employee.photo_url}
        alt={`Эталонное фото сотрудника ${employee.personnel_number}`}
        onLoad={() => setPhotoState('ready')}
        onError={() => setPhotoState('error')}
      />
      : <div className="arcface-saved-photo-placeholder" aria-hidden="true">Фото</div>}
    <span className="arcface-saved-photo-copy">
      <strong>Эталонное фото из системы</strong>
      <small>
        {photoState === 'loading'
          ? 'Загружаем фото…'
          : photoState === 'ready'
            ? `${employeeName} · ${employee.personnel_number}`
            : 'Не удалось загрузить сохранённое фото.'}
      </small>
    </span>
  </div>;
}

function ArcFaceRecognitionPanel({
  employeeByUuid,
  employeesLoading = false,
  employeesError = false,
  onSelectEmployee,
}) {
  const [file, setFile] = useState(null);
  const [employeeToEnrollUuid, setEmployeeToEnrollUuid] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [result, setResult] = useState(null);
  const [serviceStatus, setServiceStatus] = useState(null);
  const [isLoadingStatus, setIsLoadingStatus] = useState(true);
  const [isSearching, setIsSearching] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [isEnrolling, setIsEnrolling] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  useEffect(() => {
    let active = true;
    fetch('/api/arcface/status')
      .then(async (response) => {
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || 'Статус ArcFace недоступен.');
        if (active) setServiceStatus(body);
      })
      .catch((requestError) => {
        if (active) setError(requestError?.message || 'Статус ArcFace недоступен.');
      })
      .finally(() => {
        if (active) setIsLoadingStatus(false);
      });
    return () => { active = false; };
  }, []);

  const employeesWithPhoto = [...employeeByUuid.values()]
    .filter((employee) => employee.has_photo)
    .sort((a, b) => a.personnel_number.localeCompare(b.personnel_number, 'ru'));
  const employeeToEnroll = employeeByUuid.get(employeeToEnrollUuid);
  const employeeListLoading = employeesLoading && !employeesWithPhoto.length;
  const employeeListUnavailable = employeesError && !employeesWithPhoto.length;
  useEffect(() => {
    if (!file) {
      setPreviewUrl('');
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  const serviceReady = serviceStatus?.index_ready && serviceStatus?.model_ready;
  const updateServiceStatus = async () => {
    const response = await fetch('/api/arcface/status');
    const body = await response.json();
    if (!response.ok) throw new Error(body.detail || 'Статус ArcFace недоступен.');
    setServiceStatus(body);
  };

  const refreshIndex = async () => {
    setIsRefreshing(true);
    setError('');
    setNotice('');
    setResult(null);
    try {
      const response = await fetch('/api/refresh-index', { method: 'POST' });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Не удалось обновить индекс.');
      await updateServiceStatus();
      setNotice(`Индекс обновлён: ${body.vectors_count} векторов · ${body.employees_count} сотрудников · ${body.elapsed_seconds} с.`);
    } catch (requestError) {
      setError(requestError?.message || 'Не удалось обновить индекс.');
      await updateServiceStatus().catch(() => {});
    } finally {
      setIsRefreshing(false);
    }
  };

  const addEmployeeToIndex = async () => {
    if (!employeeToEnroll?.has_photo) return;
    setIsEnrolling(true);
    setError('');
    setNotice('');
    setResult(null);
    try {
      const response = await fetch(
        `/api/employees/${encodeURIComponent(employeeToEnroll.guid)}/arcface-index`,
        { method: 'POST' },
      );
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Не удалось добавить фото в индекс.');
      setNotice(
        body.already_indexed
          ? `Фото сотрудника ${employeeToEnroll.personnel_number} уже есть в индексе.`
          : body.updated
            ? `Эмбеддинг сотрудника ${employeeToEnroll.personnel_number} обновлён.`
            : `Эмбеддинг сотрудника ${employeeToEnroll.personnel_number} добавлен в индекс.`,
      );
      await updateServiceStatus().catch(() => {});
    } catch (requestError) {
      setError(requestError?.message || 'Не удалось добавить фото в индекс.');
    } finally {
      setIsEnrolling(false);
    }
  };

  const recognize = async (event) => {
    event.preventDefault();
    if (!file) return;
    setIsSearching(true);
    setError('');
    setNotice('');
    setResult(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch('/api/recognize?top_k=5&threshold=0.7', {
        method: 'POST',
        body: formData,
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail || 'Не удалось выполнить поиск.');
      setResult(body);
    } catch (requestError) {
      setError(requestError?.message || 'Не удалось выполнить поиск.');
    } finally {
      setIsSearching(false);
    }
  };

  return <Panel
    kicker="Дополнительный способ идентификации"
    title="Поиск кандидатов ArcFace"
    className="arcface-recognition-panel"
    action={<div className="arcface-header-actions">
      <button
        className="button button-primary arcface-enroll-button"
        type="button"
        onClick={addEmployeeToIndex}
        disabled={!employeeToEnroll?.has_photo || !serviceReady || isEnrolling || isRefreshing}
        data-testid="button-add-employee-to-arcface-index"
      >
        {isEnrolling ? 'Добавляем…' : 'Добавить в индекс'}
      </button>
      <button
        className="button button-secondary arcface-refresh-button"
        type="button"
        onClick={refreshIndex}
        disabled={isRefreshing || isEnrolling}
      >
        <RefreshCw size={14} className={isRefreshing ? 'arcface-spinning' : ''} />
        {isRefreshing ? 'Обновляем…' : 'Обновить индекс'}
      </button>
    </div>}
  >
    <div className="arcface-panel-body">
      <div className="arcface-index-status" role="status">
        <span className={`arcface-status-dot ${serviceReady ? 'arcface-status-ready' : ''}`} />
        {isLoadingStatus
          ? 'Проверяем модель и индекс…'
          : serviceReady
            ? `Готово · ${serviceStatus.vectors_count} векторов · ${serviceStatus.employees_count} сотрудников`
            : 'Модель или индекс пока недоступны'}
      </div>
      <label className="arcface-enroll-picker">
        <span>Сотрудник для добавления</span>
        <select
          value={employeeToEnrollUuid}
          onChange={(event) => {
            setEmployeeToEnrollUuid(event.target.value);
            setError('');
            setNotice('');
          }}
          disabled={!employeesWithPhoto.length || employeeListLoading || isEnrolling || isRefreshing}
          data-testid="select-arcface-enrollment-employee"
        >
          <option value="">
            {employeeListLoading
              ? 'Загружаем сотрудников…'
              : employeeListUnavailable
                ? 'Список сотрудников недоступен'
                : employeesWithPhoto.length
                  ? 'Выберите сотрудника с эталонным фото'
                  : 'Нет сотрудников с сохранённым фото'}
          </option>
          {employeesWithPhoto.map((employee) => (
            <option key={employee.guid} value={employee.guid}>
              {`${employee.surname} ${employee.given_name} ${employee.patronymic}`.trim()} · {employee.personnel_number}
            </option>
          ))}
        </select>
        <small>Используется фото, уже сохранённое в карточке сотрудника.</small>
      </label>
      {employeeToEnroll && <ArcFaceEnrolledPhotoPreview key={employeeToEnroll.guid} employee={employeeToEnroll} />}
      {!serviceReady && !isLoadingStatus && <p className="arcface-muted">
        {serviceStatus?.index_error || serviceStatus?.model_error || 'Проверьте файлы ArcFace в каталоге хранения и обновите индекс.'}
      </p>}
      <form className="arcface-search-form" onSubmit={recognize}>
        <label className="arcface-file-picker">
          <span>Одно фото для поиска</span>
          <input
            type="file"
            accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
            data-testid="input-arcface-photo"
            onChange={(event) => {
              const selectedFile = event.target.files?.[0] || null;
              setResult(null);
              setError('');
              setNotice('');
              if (selectedFile && selectedFile.size > 10 * 1024 * 1024) {
                setFile(null);
                setError('Размер изображения не должен превышать 10 МБ.');
                event.target.value = '';
                return;
              }
              if (selectedFile && !selectedFile.type.startsWith('image/')) {
                setFile(null);
                setError('Выберите файл изображения.');
                event.target.value = '';
                return;
              }
              setFile(selectedFile);
            }}
          />
          <small>JPG, PNG или WebP · до 10 МБ. Это отдельное фото для поиска; эталонное фото сотрудника не заменяется.</small>
        </label>
        {previewUrl && <img className="arcface-photo-preview" src={previewUrl} alt="Предпросмотр фото для поиска" />}
        <button className="button button-primary" type="submit" disabled={!file || !serviceReady || isSearching || isRefreshing || isEnrolling}>
          <Search size={14} />
          {isSearching ? 'Ищем кандидатов…' : 'Найти сотрудника'}
        </button>
      </form>
      {error && <div className="arcface-message arcface-message-error" role="alert"><AlertTriangle size={14} />{error}</div>}
      {notice && <div className="arcface-message arcface-message-success" role="status">{notice}</div>}
      {result && <div className="arcface-search-result" aria-live="polite">
        <div className={`arcface-match-status ${result.found ? 'arcface-match-found' : 'arcface-match-unknown'}`}>
          <Fingerprint size={17} />
          <span>
            <strong>{result.found ? 'Есть кандидат выше порога' : 'Кандидат выше порога не найден'}</strong>
            <small>Порог сходства: {(result.threshold * 100).toFixed(0)}%</small>
          </span>
        </div>
        {result.results?.length > 0
          ? <ol className="arcface-candidate-list">
            {result.results.map((candidate, index) => {
              const employee = employeeByUuid.get(candidate.uuid);
              const score = Number(candidate.cosine_similarity);
              return <li key={candidate.uuid}>
                <div className="arcface-candidate-identity">
                  <span className="arcface-candidate-rank">{index + 1}</span>
                  <span>
                    <strong>{employee ? `${employee.surname} ${employee.given_name} ${employee.patronymic}`.trim() : 'Сотрудник не найден в справочнике'}</strong>
                    <small className="arcface-uuid">{candidate.uuid}</small>
                  </span>
                </div>
                <div className="arcface-candidate-score">
                  <strong>{(score * 100).toFixed(1)}%</strong>
                  {employee && <button className="button button-secondary" type="button" onClick={() => onSelectEmployee(employee)}>
                    Выбрать для видео-проверки
                  </button>}
                </div>
              </li>;
            })}
          </ol>
          : <p className="arcface-muted">Индекс пуст — добавьте эмбеддинги и обновите его.</p>}
        <p className="arcface-disclaimer">{result.disclaimer}</p>
      </div>}
    </div>
  </Panel>;
}

function AppShell({ children, user, onLogout }) {
  const [location] = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);
  const health = useHealthCheck({ query: { queryKey: getHealthCheckQueryKey(), refetchInterval: 30000 } });
  const current = navItems.find((item) => item.href === location) || navItems.find((item) => location.startsWith(item.href) && item.href !== '/');
  return <div className="app-shell noise">
    <aside className={`sidebar ${mobileOpen ? 'sidebar-open' : ''}`}>
      <div className="brand-row">
        <div className="brand-mark"><span>H</span><i /></div>
        <div><div className="brand-name">ИС УКЗ</div><div className="brand-sub">исследовательская консоль</div></div>
        <button className="icon-button sidebar-close" data-testid="button-close-navigation" onClick={() => setMobileOpen(false)}><PanelLeftClose size={17} /></button>
      </div>
      <div className="research-badge"><span className="pulse-dot" /><span>ТОЛЬКО ИССЛЕДОВАНИЯ</span><span className="mono">v0.9</span></div>
      <nav className="main-nav" aria-label="Основная навигация">
        <div className="eyebrow nav-label">Рабочая область</div>
        {navItems.map(({ href, label, sub, icon: Icon }) => <Link href={href} key={href} className={`nav-item ${current?.href === href ? 'nav-active' : ''}`} data-testid={`link-${label.toLowerCase().replaceAll(' ', '-')}`} onClick={() => setMobileOpen(false)}>
          <span className="nav-icon"><Icon size={17} /></span><span><b>{label}</b><small>{sub}</small></span>{current?.href === href && <ChevronRight size={15} className="nav-chevron" />}
        </Link>)}
      </nav>
      <div className="sidebar-bottom">
        <div className="system-card">
          <div className="system-head"><span className={`system-light ${health.isError ? 'system-down' : ''}`} />Состояние системы <span className="mono">{health.isLoading ? '...' : health.isError ? 'сбой' : 'норма'}</span></div>
          <div className="system-line"><span>API</span><span>{health.isError ? 'внимание' : 'работает'}</span></div>
          <div className="system-line"><span>Хранение трасс</span><span>180 дней</span></div>
        </div>
        <div className="user-row"><div className="avatar">{initials(user?.firstName || user?.email || 'Пользователь')}</div><div className="user-details"><b>{[user?.firstName, user?.lastName].filter(Boolean).join(' ') || user?.email || 'Пользователь'}</b><small>{user?.email || 'Авторизованная сессия'}</small></div><button className="user-logout" type="button" aria-label="Выйти из системы" title="Выйти из системы" onClick={onLogout}><LogOut size={16} /></button></div>
      </div>
    </aside>
    {mobileOpen && <button className="mobile-scrim" aria-label="Закрыть меню" data-testid="button-close-scrim" onClick={() => setMobileOpen(false)} />}
    <main className="main-column">
      <header className="topbar">
        <button className="icon-button mobile-menu" data-testid="button-open-navigation" onClick={() => setMobileOpen(true)}><Menu size={19} /></button>
        <div className="crumbs"><span className="mono">ИС / УКЗ</span><ChevronRight size={13} /><strong>{current?.label || 'Консоль'}</strong></div>
        <div className="topbar-actions"><span className="review-mode"><span />Режим проверки</span><button className="icon-button" data-testid="button-notifications" aria-label="Уведомления"><Bell size={18} /></button><div className="top-avatar">{initials(user?.firstName || user?.email || 'Пользователь')}</div></div>
      </header>
      <div className="content">{children}</div>
    </main>
  </div>;
}

function AuthGate({ login, isLoading, error }) {
  const loginFailed = new URLSearchParams(window.location.search).has('login_error');
  return <main className="auth-gate">
    <section className="auth-card">
      <div className="brand-mark auth-brand-mark"><span>H</span><i /></div>
      <div className="eyebrow">ИС УКЗ · HEALTH_ID</div>
      <h1 className="display">{isLoading ? 'Проверяем вход' : 'Вход в систему'}</h1>
      <p>{isLoading ? 'Проверяем защищённую сессию…' : 'Для доступа к исследовательской консоли войдите через учётную запись.'}</p>
      {error && <div className="auth-message" role="alert">{error}</div>}
      {loginFailed && <div className="auth-message" role="alert">Не удалось завершить вход. Попробуйте ещё раз.</div>}
      {!isLoading && <button className="button button-primary auth-login-button" type="button" onClick={login}>Войти</button>}
      <small>Доступ к данным предоставляется только после входа.</small>
    </section>
  </main>;
}

function EmbeddingsPage() {
  const [, setLocation] = useLocation();
  const employees = useQuery({
    queryKey: ['employees'],
    queryFn: async () => {
      const response = await fetch('/api/employees');
      if (!response.ok) throw new Error('Не удалось загрузить список сотрудников.');
      return response.json();
    },
    staleTime: 5 * 60 * 1000,
  });
  const employeeByUuid = useMemo(
    () => new Map((employees.data || []).map((employee) => [employee.guid, employee])),
    [employees.data],
  );

  return <EmbeddingsIndexPage arcfacePanel={
    <ArcFaceRecognitionPanel
      employeeByUuid={employeeByUuid}
      employeesLoading={employees.isLoading}
      employeesError={employees.isError}
      onSelectEmployee={(employee) => {
        const query = new URLSearchParams({
          employee_ref: employee.personnel_number,
          new: '1',
        });
        setLocation(`/verification?${query.toString()}`);
      }}
    />
  } />;
}

function DashboardPage() {
  const [showCalc, setShowCalc] = useState(false);
  const summary = useGetDashboardSummary();
  const activity = useListActivity({ limit: 8 });
  const data = summary.data;
  const categories = data?.categories || {};
  const categoryEntries = Object.entries(categories);
  const categoryMax = Math.max(...categoryEntries.map(([, value]) => Number(value) || 0), 1);
  return <PageFrame
    eyebrow="Обзор / состояние системы"
    title="Добрый день, эксперт."
    intro="Трассируемый обзор сигналов, требующих внимания специалиста."
    action={<button className="button button-primary" data-testid="button-open-calculation" onClick={() => setShowCalc((value) => !value)}><Plus size={15} /> Рассчитать HEALTH_ID</button>}
  >
    {showCalc && <CalculationPanel onClose={() => setShowCalc(false)} />}
    {summary.isLoading ? <div className="metric-grid">{[1, 2, 3, 4].map((item) => <div className="metric-card skeleton-card" key={item} />)}</div> : summary.isError ? <ErrorState onRetry={summary.refetch} /> : <div className="metric-grid">
      <MetricCard label="Всего событий" value={compact(data?.total_events)} detail="всего записей" icon={Activity} />
      <MetricCard label="HEALTH_ID сегодня" value={compact(data?.health_id_today)} detail="новых расчётов" icon={HeartPulse} accent="coral" trend={4.8} />
      <MetricCard label="Доля подтверждений" value={pct(data?.verification_pass_rate)} detail="сессий верификации" icon={Fingerprint} accent="blue" trend={2.1} />
      <MetricCard label="Открытые задачи" value={compact(data?.open_review_tasks)} detail="требуют внимания" icon={ClipboardCheck} accent="amber" />
    </div>}
    <div className="dashboard-grid">
      <Panel kicker="Распределение HEALTH_ID" title="Зона сигнала" className="category-panel">
        <div className="category-intro"><div><span className="big-number display">{data?.data_quality ? pct(data.data_quality) : '—'}</span><span className="muted-text">качество данных</span></div><div className="model-chip"><span className="pulse-dot" />модель <b>{data?.model_version || '—'}</b></div></div>
        <div className="distribution">
          {categoryEntries.length ? categoryEntries.map(([key, value], index) => <div className="distribution-row" key={key}><div className="distribution-label"><span className={`category-mark mark-${index % 3}`} />{labelCategory(key)}<b>{value}</b></div><div className="distribution-track"><span className={`bar-fill fill-${index % 3}`} style={{ width: `${Math.max(6, (Number(value) / categoryMax) * 100)}%` }} /></div></div>) : <EmptyState icon={Gauge} title="Нет распределения" text="Распределение появится после поступления событий." />}
        </div>
        <div className="panel-footnote"><Sparkles size={14} /> Категории — описательные исследовательские сигналы, а не клинические диагнозы.</div>
      </Panel>
      <ActivityPanel activity={activity} />
    </div>
    <div className="dashboard-lower">
      <Panel kicker="Карта внимания" title="Работа движется через консоль" className="attention-panel">
        <div className="attention-map"><div className="map-ring ring-one" /><div className="map-ring ring-two" /><div className="map-center"><span className="pulse-dot" /><b>{data?.open_review_tasks ?? '—'}</b><small>открытых задач</small></div><div className="map-label label-top">сигнал обнаружен</div><div className="map-label label-right">решение эксперта</div><div className="map-label label-bottom">аудит сохранён</div></div>
        <div className="attention-copy"><p>Каждый результат содержит версию модели, контекст входных данных, цепочку свидетельств и исследовательское ограничение.</p><Link href="/governance" className="text-link" data-testid="link-read-governance">Управление моделью <ChevronRight size={14} /></Link></div>
      </Panel>
      <Panel kicker="Заметка оператора" title="Консоль показывает сигнал, но не выносит вердикт." className="note-panel">
        <div className="quote-mark">“</div><p className="operator-quote">Отмечайте неоднозначность. Сохраняйте контекст. Решение принимает квалифицированный специалист.</p><div className="note-meta"><span className="avatar small-avatar">УК</span><span>Принцип HEALTH_ID<br /><span className="muted-text">Управление моделью / 04</span></span></div>
      </Panel>
    </div>
  </PageFrame>;
}

function ActivityPanel({ activity }) {
  return <Panel kicker="Живая трасса" title="Последние действия" action={<Link href="/review" className="text-link" data-testid="link-view-review-activity">Открыть очередь <ChevronRight size={14} /></Link>}>
    {activity.isLoading ? <LoadingRows rows={4} /> : activity.isError ? <ErrorState onRetry={activity.refetch} /> : activity.data?.length ? <div className="activity-list">{activity.data.slice(0, 6).map((item) => <div className="activity-item" key={item.id} data-testid={`activity-item-${item.id}`}><div className={`activity-icon activity-${toneForStatus(item.status)}`}><Activity size={14} /></div><div className="activity-body"><div><b>{item.title}</b><span className="mono activity-time">{fmtDate(item.created_at)}</span></div><p>{item.description}</p></div><StatusPill tone={toneForStatus(item.status)}>{labelStatus(item.status)}</StatusPill></div>)}</div> : <EmptyState icon={History} title="Действий пока нет" text="Журнал аудита появится после поступления событий в систему." />}
  </Panel>;
}

function PageFrame({ eyebrow, title, intro, action, children }) {
  return <div className="page-in"><div className="page-header"><div><div className="eyebrow">{eyebrow}</div><h1 className="display page-title">{title}</h1><p className="page-intro">{intro}</p></div>{action && <div className="page-action">{action}</div>}</div>{children}</div>;
}

function CalculationPanel({ onClose, onCalculated }) {
  const calculate = useCalculateHealthId();
  const client = useQueryClient();
  const eligibleEmployees = useListEligibleHealthEmployees();
  const [form, setForm] = useState({ employee_ref: '', event_date: new Date().toISOString().slice(0, 10), sleep_hours: '7.2', stress_score: '3.1', activity_minutes: '42', resting_pulse: '68', respiratory_rate: '15', verification_quality: '0.94' });
  const selectedEmployee = (eligibleEmployees.data || []).find((employee) => employee.employee_ref === form.employee_ref);
  const update = (key, value) => setForm((current) => ({ ...current, [key]: value }));
  const submit = (event) => {
    event.preventDefault();
    calculate.mutate({ data: { employee_ref: form.employee_ref, event_date: `${form.event_date}T12:00:00`, measurements: { sleep_hours: Number(form.sleep_hours), stress_score: Number(form.stress_score), activity_minutes: Number(form.activity_minutes), resting_pulse: Number(form.resting_pulse), respiratory_rate: Number(form.respiratory_rate), verification_quality: Number(form.verification_quality) } } }, { onSuccess: (result) => { client.invalidateQueries({ queryKey: getListHealthResultsQueryKey({ limit: 100 }) }); client.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() }); client.invalidateQueries({ queryKey: getListActivityQueryKey({ limit: 8 }) }); onCalculated?.(result); } });
  };
  return <div className="inline-workbench page-in">
    <div className="workbench-head"><div><div className="eyebrow">Новый исследовательский расчёт</div><h2 className="display">Рассчитать прослеживаемый сигнал</h2></div><button className="icon-button" aria-label="Закрыть форму" data-testid="button-close-calculation" onClick={onClose}><X size={17} /></button></div>
    <form onSubmit={submit} className="calc-form">
      <label className="field"><span>Сотрудник · есть успешная верификация</span><select required data-testid="select-calculation-employee" value={form.employee_ref} onChange={(e) => { update('employee_ref', e.target.value); calculate.reset(); }} disabled={eligibleEmployees.isLoading || eligibleEmployees.isError || !eligibleEmployees.data?.length}><option value="">{eligibleEmployees.isLoading ? 'Загружаем допущенных сотрудников…' : 'Выберите сотрудника'}</option>{eligibleEmployees.data?.map((employee) => <option key={employee.employee_ref} value={employee.employee_ref}>{`${employee.full_name} · ${employee.employee_ref} · ${employee.eligibility_source === 'manual' ? 'подтверждён специалистом' : 'успешно проверен'}`}</option>)}</select>
        {eligibleEmployees.isError && <small className="calc-eligibility-message calc-eligibility-error" role="alert">{eligibleEmployees.error?.message || 'Не удалось загрузить список допущенных сотрудников.'}</small>}
        {!eligibleEmployees.isLoading && !eligibleEmployees.isError && !eligibleEmployees.data?.length && <small className="calc-eligibility-message">Нет сотрудников с успешной последней верификацией или решением «Подтвердить» в очереди проверки.</small>}
        {selectedEmployee && <small className="calc-eligibility-message">{selectedEmployee.eligibility_source === 'manual' ? 'Ручное подтверждение' : 'Успешная системная верификация'} · {fmtDate(selectedEmployee.verified_at)}</small>}
      </label>
      <label className="field"><span>Дата события</span><input type="date" required data-testid="input-calculation-date" value={form.event_date} onChange={(e) => update('event_date', e.target.value)} /></label>
      {[
        ['sleep_hours', 0, 24],
        ['stress_score', 0, 10],
        ['activity_minutes', 0],
        ['resting_pulse', 20, 240],
        ['respiratory_rate', 4, 60],
        ['verification_quality', 0, 1],
      ].map(([key, min, max]) => <label className="field" key={key}><span>{labelField(key)}</span><input type="number" min={min} max={max} step="any" required data-testid={`input-calculation-${key}`} value={form[key]} onChange={(e) => update(key, e.target.value)} /></label>)}
      <button className="button button-primary calc-submit" disabled={calculate.isPending || eligibleEmployees.isLoading || eligibleEmployees.isError || !eligibleEmployees.data?.length || !form.employee_ref} data-testid="button-submit-calculation">{calculate.isPending ? 'Выполняется расчёт…' : <><Play size={14} /> Рассчитать</>}</button>
    </form>
    {calculate.isError && <div className="form-alert form-alert-error"><AlertTriangle size={15} /> {calculate.error?.message || 'Не удалось выполнить расчёт. Проверьте введённые данные и повторите попытку.'}</div>}
    {calculate.data && <div className="success-result"><div><span className="eyebrow">Результат расчёта</span><strong className="display">{calculate.data.health_id?.toFixed(3) || '—'}</strong><span className="muted-text">HEALTH_ID · {labelCategory(calculate.data.category)}</span></div><div className="result-trace"><StatusPill tone={toneForStatus(calculate.data.status)}>{labelStatus(calculate.data.status)}</StatusPill><span className="mono">{calculate.data.id}</span><button className="icon-button" aria-label="Скопировать идентификатор расчёта" data-testid="button-copy-calculation-id" onClick={() => navigator.clipboard?.writeText(calculate.data.id)}><Copy size={14} /></button></div></div>}
  </div>;
}

function HealthResultsPage() {
  const params = useParams();
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');
  const [showCalculation, setShowCalculation] = useState(false);
  const [selectedId, setSelectedId] = useState(params.id || '');
  const listParams = useMemo(() => ({ limit: 100, ...(category !== 'all' ? { category } : {}) }), [category]);
  const results = useListHealthResults(listParams);
  const detail = useGetHealthResult(selectedId || 'none', { query: { enabled: Boolean(selectedId), queryKey: getGetHealthResultQueryKey(selectedId || 'none') } });
  const filtered = (results.data || []).filter((item) => `${item.employee_ref} ${item.id} ${item.category}`.toLowerCase().includes(search.toLowerCase()));
  const selectedListResult = results.data?.find((item) => item.id === selectedId);
  return <PageFrame eyebrow="Результаты исследований / HEALTH_ID" title="Реестр сигналов здоровья" intro="Ищите результаты, изучайте доказательства и учитывайте исследовательские ограничения." action={<div className="health-results-actions"><button className="button button-primary" data-testid="button-open-health-calculation" onClick={() => setShowCalculation((current) => !current)}><Plus size={15} /> Добавить и создать расчёт</button><button className="button button-secondary" data-testid="button-export-results"><Database size={15} /> Экспорт реестра</button></div>}>
    {showCalculation && <CalculationPanel onClose={() => setShowCalculation(false)} onCalculated={(result) => { setCategory('all'); setSearch(''); setSelectedId(result.id); }} />}
    <div className="filter-bar"><div className="search-field"><Search size={16} /><input type="search" data-testid="input-search-health-results" placeholder="Поиск по сотруднику или ID результата" value={search} onChange={(e) => setSearch(e.target.value)} /></div><div className="filter-tabs">{['all', 'stable', 'attention', 'review'].map((item) => <button key={item} data-testid={`button-filter-${item}`} className={category === item ? 'filter-active' : ''} onClick={() => setCategory(item)}>{item === 'all' ? 'Все сигналы' : labelCategory(item)}<span>{item === 'all' ? (results.data?.length || 0) : (results.data || []).filter((result) => result.category === item).length}</span></button>)}</div><button className="icon-button filter-more" aria-label="Дополнительные фильтры" data-testid="button-more-filters"><SlidersHorizontal size={16} /></button></div>
    <div className="results-layout"><Panel kicker={`${filtered.length} записей`} title="Реестр результатов" className="results-panel"><div className="table-wrap mobile-scroll"><table className="data-table"><thead><tr><th>Запись</th><th>HEALTH_ID</th><th>Категория</th><th>Статус</th><th>Личный коридор</th><th>Модель</th><th>Создано</th></tr></thead><tbody>{results.isLoading ? <tr><td colSpan="7"><LoadingRows rows={5} /></td></tr> : results.isError ? <tr><td colSpan="7"><ErrorState onRetry={results.refetch} /></td></tr> : filtered.length ? filtered.map((item) => { const healthIdOutside = item.personal_baseline?.outside_metrics?.includes('health_id'); return <tr key={item.id} className={selectedId === item.id ? 'row-selected' : ''} onClick={() => setSelectedId(item.id)} data-testid={`row-health-result-${item.id}`}><td><div className="record-cell"><span className="record-avatar">{initials(item.employee_ref)}</span><span><b>{item.employee_ref}</b><small className="mono">{item.id}</small></span></div></td><td><strong className={`score ${healthIdOutside ? 'score-outside-baseline' : ''}`} title={healthIdOutside ? 'HEALTH_ID вне персонального коридора' : undefined}>{Number(item.health_id).toFixed(3)}</strong></td><td><span className={`category-text cat-${item.category}`}>{labelCategory(item.category)}</span></td><td><StatusPill tone={toneForStatus(item.status)}>{labelStatus(item.status)}</StatusPill></td><td><PersonalBaselineBadge evaluation={item.personal_baseline} /></td><td className="mono muted-text">{item.model_version}</td><td className="mono muted-text">{fmtDate(item.created_at)}</td></tr>; }) : <tr><td colSpan="7"><EmptyState icon={HeartPulse} title="Нет подходящих сигналов" text="Попробуйте другой идентификатор сотрудника или категорию." /></td></tr>}</tbody></table></div></Panel><HealthDetail result={detail.data} baselineEvaluation={selectedListResult?.personal_baseline} loading={detail.isLoading && Boolean(selectedId)} error={detail.isError} /></div>
  </PageFrame>;
}

function PersonalBaselinePanel({ employeeRef }) {
  const [requested, setRequested] = useState(false);
  const baseline = useGetPersonalHealthBaseline(employeeRef, {
    query: {
      enabled: requested,
      queryKey: getGetPersonalHealthBaselineQueryKey(employeeRef),
    },
  });

  const formatValue = (value) => value == null
    ? '—'
    : new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 }).format(value);

  return <section className="personal-baseline" aria-label="Персональные baseline-коридоры">
    <div className="personal-baseline-heading">
      <div>
        <div className="eyebrow">Персональный baseline</div>
        <h3>Коридоры показателей · 30 дней</h3>
        <p>Медиана ± 1,5 IQR. Для коридора требуется минимум 4 наблюдения.</p>
      </div>
      <button
        className="button button-secondary"
        data-testid="button-calculate-personal-baseline"
        disabled={baseline.isFetching}
        onClick={() => setRequested(true)}
      >
        {baseline.isFetching ? 'Расчёт…' : requested ? 'Обновить baseline' : 'Рассчитать baseline'}
      </button>
    </div>
    {baseline.isError && <div className="personal-baseline-error" role="alert">
      <span>{baseline.error?.message || 'Не удалось рассчитать персональные коридоры.'}</span>
      <button className="text-link" onClick={() => baseline.refetch()}>Повторить</button>
    </div>}
    {baseline.data && <div className="personal-baseline-list">
      {baseline.data.metrics.map((metric) => <div className="personal-baseline-row" key={metric.key} data-testid={`personal-baseline-${metric.key}`}>
        <div className="personal-baseline-label">
          <b>{metric.label}</b>
          <small>{metric.sample_count} из {baseline.data.minimum_samples} наблюдений минимум</small>
        </div>
        {metric.ready ? <div className="personal-baseline-values">
          <strong>{formatValue(metric.lower_bound)} — {formatValue(metric.upper_bound)} <small>{metric.unit}</small></strong>
          <span>Медиана {formatValue(metric.median)} · IQR {formatValue(metric.iqr)}</span>
        </div> : <span className="personal-baseline-insufficient">Недостаточно данных{metric.sample_count ? ` · медиана ${formatValue(metric.median)}` : ''}</span>}
      </div>)}
      <p className="personal-baseline-note">Исходные показатели сохраняются только для новых расчётов; это исследовательские коридоры, не медицинские нормы.</p>
    </div>}
  </section>;
}

function HealthDetail({ result, baselineEvaluation, loading, error }) {
  if (!result && !loading) return <Panel className="detail-panel detail-empty"><EmptyState icon={Microscope} title="Выберите результат" text="Откройте запись, чтобы изучить компоненты сигнала и цепочку свидетельств." /></Panel>;
  return <Panel kicker="Подробности трассы" title={result ? result.employee_ref : 'Загрузка записи'} className="detail-panel">{loading ? <LoadingRows rows={6} /> : error ? <ErrorState /> : <div className="detail-content"><div className="detail-score"><span className="eyebrow">Оценка HEALTH_ID</span><strong className={`display ${baselineEvaluation?.outside_metrics?.includes('health_id') ? 'score-outside-baseline' : ''}`}>{Number(result.health_id).toFixed(3)}</strong><StatusPill tone={toneForStatus(result.category)}>{labelCategory(result.category)}</StatusPill></div><PersonalBaselineBadge evaluation={baselineEvaluation} detailed /><div className="detail-meta"><span><small>ID результата</small><b className="mono">{result.id}</b></span><span><small>Дата события</small><b>{fmtDate(result.event_date, false)}</b></span><span><small>Модель</small><b className="mono">{result.model_version}</b></span></div><div className="subsection"><div className="eyebrow">Компоненты</div>{Object.entries(result.components || {}).map(([key, value]) => <div className="component-row" key={key}><span>{labelField(key)}</span><div className="component-track"><span style={{ width: `${Math.min(100, Number(value) * 100)}%` }} /></div><b className="mono">{Number(value).toFixed(2)}</b></div>)}</div><div className="subsection"><div className="eyebrow">Цепочка свидетельств</div>{(result.evidence || []).map((item) => <div className="evidence-row" key={item.key}><span className={`evidence-icon evidence-${item.direction}`}><Check size={12} /></span><span><b>{item.label}</b><small>{item.value}</small></span><span className="mono evidence-direction">{labelStatus(item.direction)}</span></div>)}</div><div className="disclaimer"><AlertTriangle size={15} /><span>{result.disclaimer || 'Результат предназначен только для исследований. Это не медицинский диагноз и не рекомендация по лечению.'}</span></div><PersonalBaselinePanel employeeRef={result.employee_ref} /></div>}</Panel>;
}

function SelectedFilePreview({ file, mediaKind, title }) {
  const [previewUrl, setPreviewUrl] = useState('');
  useEffect(() => {
    if (!file) {
      setPreviewUrl('');
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);
  if (!file || !previewUrl) return null;
  const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
  return <div className="upload-preview">
    {mediaKind === 'photo' ? <img src={previewUrl} alt={title || `Предпросмотр файла ${file.name}`} /> : <video src={previewUrl} controls preload="metadata" aria-label={title || `Предпросмотр файла ${file.name}`} />}
    <div><b title={file.name}>{title || file.name}</b><small>{file.name} · {sizeMb} МБ</small></div>
  </div>;
}

function VerificationPage() {
  const params = useParams();
  const [, setLocation] = useLocation();
  const initialSearch = new URLSearchParams(window.location.search);
  const initialEmployeeRef = initialSearch.get('employee_ref') || '';
  const { retainReviewMedia } = useContext(ReviewMediaContext);
  const [selectedId, setSelectedId] = useState(params.id || '');
  const [showForm, setShowForm] = useState(initialSearch.get('new') === '1');
  const [enrollVideoEmbeddings, setEnrollVideoEmbeddings] = useState(false);
  const [videoFile, setVideoFile] = useState(null);
  const [videoSource, setVideoSource] = useState('upload');
  const [cameraStream, setCameraStream] = useState(null);
  const [cameraCaptureInfo, setCameraCaptureInfo] = useState({});
  const [cameraError, setCameraError] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isConvertingRecording, setIsConvertingRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const cameraPreviewRef = useRef(null);
  const cameraStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const recordStartedAtRef = useRef(0);
  const recordingUrlRef = useRef('');
  const [recordingUrl, setRecordingUrl] = useState('');
  const [form, setForm] = useState({ employee_ref: initialEmployeeRef });
  const [referencePhotoDraft, setReferencePhotoDraft] = useState(null);
  const [referenceCameraStream, setReferenceCameraStream] = useState(null);
  const [referenceCameraError, setReferenceCameraError] = useState('');
  const [referencePhotoSaveError, setReferencePhotoSaveError] = useState('');
  const [referencePhotoSaveMessage, setReferencePhotoSaveMessage] = useState('');
  const [isSavingReferencePhoto, setIsSavingReferencePhoto] = useState(false);
  const [showReferencePhotoCapture, setShowReferencePhotoCapture] = useState(false);
  const referenceCameraPreviewRef = useRef(null);
  const referenceCameraStreamRef = useRef(null);
  const employees = useQuery({
    queryKey: ['employees'],
    queryFn: async () => {
      const response = await fetch('/api/employees');
      if (!response.ok) throw new Error('Не удалось загрузить список сотрудников.');
      return response.json();
    },
    staleTime: 5 * 60 * 1000,
  });
  const sortedEmployees = useMemo(() => [...(employees.data || [])].sort((a, b) => {
    const nameA = `${a.surname} ${a.given_name} ${a.patronymic}`.trim();
    const nameB = `${b.surname} ${b.given_name} ${b.patronymic}`.trim();
    return nameA.localeCompare(nameB, 'ru');
  }), [employees.data]);
  const testEmployees = sortedEmployees.filter((employee) => employee.is_test);
  const companyEmployees = sortedEmployees.filter((employee) => !employee.is_test);
  const companyName = companyEmployees[0]?.organization || 'Сотрудники компании';
  const selectedEmployee = sortedEmployees.find((employee) => employee.personnel_number === form.employee_ref);
  const referencePhoto = useQuery({
    queryKey: ['employee-reference-photo', selectedEmployee?.guid],
    enabled: Boolean(selectedEmployee?.has_photo),
    queryFn: async () => {
      const response = await fetch(selectedEmployee.photo_url);
      if (!response.ok) throw new Error('Эталонное фото сотрудника недоступно.');
      const blob = await response.blob();
      if (!blob.type.startsWith('image/')) throw new Error('Сервер вернул файл, который не является изображением.');
      return new File([blob], `reference-${selectedEmployee.personnel_number}.jpg`, { type: blob.type });
    },
    staleTime: Infinity,
  });
  const [filters, setFilters] = useState({
    query: '',
    decision: 'all',
    status: 'all',
    liveness: 'all',
    period: 'all',
    minMatchScore: '',
    minLivenessScore: '',
  });
  const list = useListVerificationSessions({ limit: 100 });
  const create = useCreateVerificationSession();
  const completeVideo = useCompleteVerificationSessionVideo();
  const deleteSession = useDeleteVerificationSession();
  const requestReview = useRequestVerificationReview();
  const [videoCompletionSessionId, setVideoCompletionSessionId] = useState('');
  const [videoCompletionStatus, setVideoCompletionStatus] = useState('idle');
  const [videoCompletionError, setVideoCompletionError] = useState('');
  const detail = useGetVerificationSession(selectedId || 'none', { query: { enabled: Boolean(selectedId), queryKey: getGetVerificationSessionQueryKey(selectedId || 'none') } });
  useEffect(() => {
    if (cameraPreviewRef.current && cameraStream) cameraPreviewRef.current.srcObject = cameraStream;
  }, [cameraStream]);
  useEffect(() => {
    if (referenceCameraPreviewRef.current && referenceCameraStream) {
      referenceCameraPreviewRef.current.srcObject = referenceCameraStream;
    }
  }, [referenceCameraStream]);
  useEffect(() => () => {
    const recorder = mediaRecorderRef.current;
    if (recorder?.state === 'recording') {
      recorder.ondataavailable = null;
      recorder.onstop = null;
      recorder.stop();
    }
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    referenceCameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current);
  }, []);
  useEffect(() => {
    if (!isRecording) return undefined;
    const timer = window.setInterval(() => {
      setRecordingSeconds(Math.floor((Date.now() - recordStartedAtRef.current) / 1000));
    }, 250);
    return () => window.clearInterval(timer);
  }, [isRecording]);
  const turnOffCamera = () => {
    const recorder = mediaRecorderRef.current;
    if (recorder?.state === 'recording') {
      recorder.ondataavailable = null;
      recorder.onstop = null;
      recorder.stop();
    }
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    cameraStreamRef.current = null;
    setCameraStream(null);
    setIsRecording(false);
  };
  const stopReferencePhotoCamera = () => {
    referenceCameraStreamRef.current?.getTracks().forEach((track) => track.stop());
    referenceCameraStreamRef.current = null;
    setReferenceCameraStream(null);
  };
  const connectReferencePhotoCamera = async () => {
    setReferenceCameraError('');
    setReferencePhotoSaveError('');
    setReferencePhotoSaveMessage('');
    if (isRecording || isConvertingRecording) {
      setReferenceCameraError('Сначала завершите запись видео.');
      return;
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      setReferenceCameraError('Этот браузер не поддерживает подключение к камере.');
      return;
    }
    try {
      turnOffCamera();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      referenceCameraStreamRef.current = stream;
      setReferenceCameraStream(stream);
    } catch (error) {
      setReferenceCameraError(error?.name === 'NotAllowedError'
        ? 'Нет доступа к камере. Разрешите доступ в настройках браузера и повторите попытку.'
        : 'Не удалось подключить камеру. Проверьте, что она подключена и не используется другим приложением.');
    }
  };
  const captureReferencePhoto = () => {
    const video = referenceCameraPreviewRef.current;
    if (!video?.videoWidth || !video?.videoHeight) {
      setReferenceCameraError('Камера ещё не готова. Подождите немного и повторите снимок.');
      return;
    }
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d')?.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob((blob) => {
      if (!blob) {
        setReferenceCameraError('Не удалось создать снимок. Повторите попытку.');
        return;
      }
      if (blob.size > 5 * 1024 * 1024) {
        setReferenceCameraError('Снимок слишком большой. Подойдите ближе к камере и повторите.');
        return;
      }
      setReferencePhotoDraft(new File([blob], 'employee-reference.jpg', {
        type: 'image/jpeg',
        lastModified: Date.now(),
      }));
      setReferenceCameraError('');
      setReferencePhotoSaveError('');
      setReferencePhotoSaveMessage('');
      stopReferencePhotoCamera();
    }, 'image/jpeg', 0.92);
  };
  const saveReferencePhoto = async () => {
    if (!selectedEmployee || !referencePhotoDraft) return;
    const employee = selectedEmployee;
    const isReplacement = employee.has_photo;
    const photo = referencePhotoDraft;
    setIsSavingReferencePhoto(true);
    setReferencePhotoSaveError('');
    setReferencePhotoSaveMessage('');
    try {
      const urlResponse = await fetch(`/api/employees/${employee.guid}/photo/upload-url`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ size: photo.size, content_type: photo.type }),
      });
      const uploadInfo = await urlResponse.json().catch(() => ({}));
      if (!urlResponse.ok) {
        throw new Error(uploadInfo.detail || 'Не удалось подготовить защищённую загрузку фото.');
      }

      const uploadResponse = await fetch(uploadInfo.upload_url, {
        method: 'PUT',
        headers: { 'Content-Type': photo.type },
        body: photo,
      });
      if (!uploadResponse.ok) {
        throw new Error('Не удалось загрузить снимок в защищённое хранилище. Повторите попытку.');
      }

      const saveResponse = await fetch(`/api/employees/${employee.guid}/photo`, {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ object_path: uploadInfo.object_path }),
      });
      const savedEmployee = await saveResponse.json().catch(() => ({}));
      if (!saveResponse.ok) {
        throw new Error(savedEmployee.detail || 'Не удалось сохранить эталонное фото сотрудника.');
      }

      queryClient.setQueryData(['employees'], (current) => (
        Array.isArray(current)
          ? current.map((item) => item.guid === savedEmployee.guid ? savedEmployee : item)
          : current
      ));
      await queryClient.invalidateQueries({ queryKey: ['employees'] });
      await queryClient.invalidateQueries({ queryKey: ['employee-reference-photo', employee.guid] });
      setReferencePhotoDraft(null);
      setShowReferencePhotoCapture(false);
      setReferencePhotoSaveMessage(isReplacement
        ? 'Новое фото сохранено. Предыдущее фото добавлено в dataset.'
        : 'Фото сохранено для выбранного сотрудника.');
    } catch (error) {
      setReferencePhotoSaveError(error?.message || 'Не удалось сохранить снимок.');
    } finally {
      setIsSavingReferencePhoto(false);
    }
  };
  const changeSelectedEmployee = (employeeRef) => {
    stopReferencePhotoCamera();
    setReferencePhotoDraft(null);
    setShowReferencePhotoCapture(false);
    setReferenceCameraError('');
    setReferencePhotoSaveError('');
    setReferencePhotoSaveMessage('');
    setEnrollVideoEmbeddings(false);
    setForm((current) => ({ ...current, employee_ref: employeeRef }));
  };
  const openReferencePhotoCapture = () => {
    setShowReferencePhotoCapture(true);
    if (!referenceCameraStream && !referencePhotoDraft) connectReferencePhotoCamera();
  };
  const closeReferencePhotoCapture = () => {
    stopReferencePhotoCamera();
    setReferencePhotoDraft(null);
    setShowReferencePhotoCapture(false);
    setReferenceCameraError('');
    setReferencePhotoSaveError('');
  };
  const startCamera = async () => {
    setCameraError('');
    if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
      setCameraError('Этот браузер не поддерживает запись с камеры. Выберите видеофайл с компьютера.');
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'user',
          width: { min: 640, ideal: 640 },
          height: { min: 480, ideal: 480 },
          aspectRatio: { ideal: 4 / 3 },
          frameRate: { min: 15, ideal: 30 },
        },
        audio: false,
      });
      const track = stream.getVideoTracks()[0];
      const settings = track?.getSettings?.() || {};
      const actualWidth = Number.isFinite(settings.width) ? settings.width : undefined;
      const actualHeight = Number.isFinite(settings.height) ? settings.height : undefined;
      const actualFps = Number.isFinite(settings.frameRate) ? settings.frameRate : undefined;
      if ((actualWidth != null && actualWidth < 640)
        || (actualHeight != null && actualHeight < 480)
        || (actualFps != null && actualFps < 15)) {
        stream.getTracks().forEach((mediaTrack) => mediaTrack.stop());
        setCameraError('Камера не обеспечивает минимум 640 × 480 и 15 кадров/с. Выберите другую камеру или видеофайл.');
        return;
      }
      cameraStreamRef.current = stream;
      setCameraStream(stream);
      setCameraCaptureInfo({
        camera_model: track?.label?.trim()?.slice(0, 120) || undefined,
        camera_width: actualWidth,
        camera_height: actualHeight,
        camera_fps: actualFps,
        camera_facing_mode: ['user', 'environment'].includes(settings.facingMode) ? settings.facingMode : 'user',
      });
    } catch (error) {
      setCameraError(error?.name === 'NotAllowedError'
        ? 'Нет доступа к камере. Разрешите доступ в настройках браузера и повторите попытку.'
        : error?.name === 'OverconstrainedError'
          ? 'Камера не поддерживает минимум 640 × 480 и 15 кадров/с. Выберите другую камеру или видеофайл.'
          : 'Не удалось подключить камеру. Проверьте, что она подключена и не используется другим приложением.');
    }
  };
  const startRecording = () => {
    const stream = cameraStreamRef.current;
    if (!stream || typeof MediaRecorder === 'undefined') return;
    const mimeType = ['video/mp4;codecs=avc1.42E01E', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
      .find((type) => MediaRecorder.isTypeSupported(type));
    try {
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recordedChunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordedChunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        const recordedType = (recorder.mimeType || mimeType || 'video/webm').split(';', 1)[0].toLowerCase();
        const recordedBlob = new Blob(recordedChunksRef.current, { type: recordedType });
        if (!recordedBlob.size) {
          setCameraError('Запись не получилась. Проверьте камеру и попробуйте ещё раз.');
          return;
        }
        const saveRecording = async () => {
          setIsConvertingRecording(true);
          setCameraError('');
          try {
            if (!['video/mp4', 'video/webm'].includes(recordedType)) {
              throw new Error('Браузер записал видео в неподдерживаемом формате. Попробуйте другой браузер.');
            }
            const mp4Blob = await convertVerificationCameraVideo({
              video_file: new File(
                [recordedBlob],
                `camera-recording.${recordedType === 'video/mp4' ? 'mp4' : 'webm'}`,
                { type: recordedType },
              ),
            });
            if (!mp4Blob.size || mp4Blob.type.split(';', 1)[0].toLowerCase() !== 'video/mp4') {
              throw new Error('Сервер вернул некорректный MP4-файл. Повторите запись.');
            }

            const file = new File(
              [mp4Blob],
              `camera-${new Date().toISOString().replace(/[:.]/g, '-')}.mp4`,
              { type: 'video/mp4' },
            );
            setVideoFile(file);
            setRecordingUrl((previousUrl) => {
              if (previousUrl) URL.revokeObjectURL(previousUrl);
              const nextUrl = URL.createObjectURL(file);
              recordingUrlRef.current = nextUrl;
              return nextUrl;
            });
            const downloadUrl = URL.createObjectURL(file);
            const downloadLink = document.createElement('a');
            downloadLink.href = downloadUrl;
            downloadLink.download = file.name;
            document.body.appendChild(downloadLink);
            downloadLink.click();
            downloadLink.remove();
            window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 30_000);
          } catch (error) {
            setCameraError(error?.message || 'Не удалось сохранить запись в формате MP4.');
          } finally {
            setIsConvertingRecording(false);
          }
        };
        void saveRecording();
      };
      mediaRecorderRef.current = recorder;
      recordStartedAtRef.current = Date.now();
      setRecordingSeconds(0);
      recorder.start(1000);
      setIsRecording(true);
    } catch {
      setCameraError('Не удалось начать запись с этой камерой. Попробуйте выбрать видеофайл с компьютера.');
    }
  };
  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') mediaRecorderRef.current.stop();
    setIsRecording(false);
  };
  const clearRecording = () => {
    setVideoFile(null);
    setRecordingUrl((previousUrl) => {
      if (previousUrl) URL.revokeObjectURL(previousUrl);
      recordingUrlRef.current = '';
      return '';
    });
  };
  const switchVideoSource = (source) => {
    if (source === 'upload') turnOffCamera();
    setVideoSource(source);
    setCameraError('');
    if (source === 'upload') setCameraCaptureInfo({});
    clearRecording();
  };
  const closeVerificationForm = () => {
    turnOffCamera();
    stopReferencePhotoCamera();
    setReferencePhotoDraft(null);
    setShowReferencePhotoCapture(false);
    setReferenceCameraError('');
    setReferencePhotoSaveError('');
    setReferencePhotoSaveMessage('');
    setShowForm(false);
    setVideoSource('upload');
    setCameraCaptureInfo({});
    setEnrollVideoEmbeddings(false);
    clearRecording();
  };
  const selectedSummary = list.data?.find((item) => item.id === selectedId) || detail.data;
  const filteredSessions = useMemo(() => {
    const query = filters.query.trim().toLocaleLowerCase('ru');
    const minMatchScore = filters.minMatchScore === '' ? null : Number(filters.minMatchScore) / 100;
    const minLivenessScore = filters.minLivenessScore === '' ? null : Number(filters.minLivenessScore) / 100;
    const periodMs = { day: 24 * 60 * 60 * 1000, week: 7 * 24 * 60 * 60 * 1000, month: 30 * 24 * 60 * 60 * 1000, quarter: 90 * 24 * 60 * 60 * 1000 }[filters.period];
    const cutoff = periodMs ? Date.now() - periodMs : null;

    return (list.data || []).filter((item) => {
      const searchable = `${item.employee_ref || ''} ${item.id || ''} ${item.audit_ref || ''}`.toLocaleLowerCase('ru');
      const createdAt = item.created_at ? Date.parse(item.created_at) : Number.NaN;
      if (query && !searchable.includes(query)) return false;
      if (filters.decision !== 'all' && item.decision !== filters.decision) return false;
      if (filters.status !== 'all' && item.status !== filters.status) return false;
      if (filters.liveness === 'live' && item.is_live !== true) return false;
      if (filters.liveness === 'not_live' && item.is_live !== false) return false;
      if (filters.liveness === 'unknown' && item.is_live != null) return false;
      if (cutoff && (!Number.isFinite(createdAt) || createdAt < cutoff)) return false;
      if (minMatchScore != null && Number(item.match_score || 0) < minMatchScore) return false;
      if (minLivenessScore != null && Number(item.liveness_score || 0) < minLivenessScore) return false;
      return true;
    });
  }, [filters, list.data]);
  const hasActiveFilters = Object.entries(filters).some(([key, value]) => value !== 'all' && value !== '');
  const submit = (event) => {
    event.preventDefault();
    if (!selectedEmployee || !referencePhoto.data || !videoFile) return;
    const isCameraRecording = videoSource === 'camera';
    const submittedPhoto = referencePhoto.data;
    const submittedVideo = videoFile;
    create.mutate({ data: {
      employee_ref: form.employee_ref,
      photo_file: submittedPhoto,
      video_file: submittedVideo,
      video_source: isCameraRecording ? 'camera' : 'upload',
      enroll_video_embeddings: enrollVideoEmbeddings,
      ...(isCameraRecording ? cameraCaptureInfo : {}),
    } }, { onSuccess: async (session) => {
      retainReviewMedia(session.id, { photo: submittedPhoto, video: submittedVideo });
      if (isCameraRecording) {
        setVideoCompletionSessionId(session.id);
        setVideoCompletionStatus('pending');
        setVideoCompletionError('');
        try {
          await completeVideo.mutateAsync({ sessionId: session.id });
          setVideoCompletionStatus('complete');
          await queryClient.invalidateQueries({ queryKey: getGetVerificationSessionQueryKey(session.id) });
        } catch (error) {
          setVideoCompletionStatus('error');
          setVideoCompletionError(error?.message || 'Не удалось подтвердить завершение приёма видео.');
        }
      } else {
        setVideoCompletionSessionId('');
        setVideoCompletionStatus('idle');
        setVideoCompletionError('');
      }
      setSelectedId(session.id);
      closeVerificationForm();
      queryClient.invalidateQueries({ queryKey: getListVerificationSessionsQueryKey({ limit: 100 }) });
      queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
      queryClient.invalidateQueries({ queryKey: getListActivityQueryKey({ limit: 8 }) });
    } });
  };
  const retryVideoCompletion = () => {
    if (!videoCompletionSessionId || completeVideo.isPending) return;
    setVideoCompletionStatus('pending');
    setVideoCompletionError('');
    completeVideo.mutate({ sessionId: videoCompletionSessionId }, {
      onSuccess: () => {
        setVideoCompletionStatus('complete');
        queryClient.invalidateQueries({ queryKey: getGetVerificationSessionQueryKey(videoCompletionSessionId) });
      },
      onError: (error) => {
        setVideoCompletionStatus('error');
        setVideoCompletionError(error?.message || 'Не удалось подтвердить завершение приёма видео.');
      },
    });
  };
  const deleteSelectedSession = () => {
    const session = selectedSummary;
    if (!session || deleteSession.isPending) return;
    const confirmed = window.confirm(`Удалить сессию ${session.employee_ref} (${session.id})?\n\nСвязанные задачи ручной проверки также будут удалены. Запись об удалении останется в журнале. Отменить действие будет нельзя.`);
    if (!confirmed) return;
    deleteSession.mutate({ sessionId: session.id }, { onSuccess: () => {
      queryClient.removeQueries({ queryKey: getGetVerificationSessionQueryKey(session.id) });
      queryClient.invalidateQueries({ queryKey: getListVerificationSessionsQueryKey({ limit: 100 }) });
      queryClient.invalidateQueries({ queryKey: getListReviewTasksQueryKey() });
      queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
      queryClient.invalidateQueries({ queryKey: getListActivityQueryKey({ limit: 8 }) });
      if (create.data?.id === session.id) create.reset();
      if (videoCompletionSessionId === session.id) {
        setVideoCompletionSessionId('');
        setVideoCompletionStatus('idle');
        setVideoCompletionError('');
      }
      setSelectedId('');
      if (params.id === session.id) setLocation('/verification');
    } });
  };
  const sendToReviewQueue = (sessionId) => {
    requestReview.mutate({ sessionId }, { onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: getListReviewTasksQueryKey() });
      queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
      queryClient.invalidateQueries({ queryKey: getListActivityQueryKey({ limit: 8 }) });
      queryClient.invalidateQueries({ queryKey: getGetVerificationSessionQueryKey(sessionId) });
    } });
  };
  const reviewRequestForSelected = requestReview.variables?.sessionId === selectedId;
  return <PageFrame eyebrow="Подтверждение личности / верификация" title="Сессии верификации" intro="Выберите сотрудника и загрузите готовое видео или запишите его с веб-камеры. Эталонное фото будет подставлено из системы автоматически." action={<button className="button button-primary" data-testid="button-open-verification" onClick={() => showForm ? closeVerificationForm() : setShowForm(true)}><Plus size={15} /> Новая сессия</button>}>
    {showForm && <div className="inline-workbench"><div className="workbench-head"><div><div className="eyebrow">Новая сессия верификации</div><h2 className="display">Выберите сотрудника и видео</h2></div><button className="icon-button" data-testid="button-close-verification" aria-label="Закрыть форму" onClick={closeVerificationForm}><X size={17} /></button></div><form className="verification-form" onSubmit={submit}>
      <div className="field employee-picker-field"><span>Сотрудник · табельный номер</span>
        <select required data-testid="select-verification-employee" value={form.employee_ref} onChange={(event) => changeSelectedEmployee(event.target.value)} disabled={employees.isLoading || employees.isError || isSavingReferencePhoto}>
          <option value="">{employees.isLoading ? 'Загрузка сотрудников…' : 'Выберите сотрудника'}</option>
          {testEmployees.length > 0 && <optgroup label="Тестовые">{testEmployees.map((employee) => <option key={employee.guid} value={employee.personnel_number}>{`${employee.surname} ${employee.given_name} ${employee.patronymic} · ${employee.personnel_number}`}</option>)}</optgroup>}
          {companyEmployees.length > 0 && <optgroup label={companyName}>{companyEmployees.map((employee) => <option key={employee.guid} value={employee.personnel_number}>{`${employee.surname} ${employee.given_name} ${employee.patronymic} · ${employee.personnel_number}`}</option>)}</optgroup>}
        </select>
        {!selectedEmployee && <small>Список отсортирован по ФИО. Фото будет загружено автоматически после выбора сотрудника.</small>}
        {employees.isError && <small className="reference-photo-error" role="alert">{employees.error?.message || 'Список сотрудников недоступен.'}</small>}
      </div>
      <div className="field reference-photo-field"><span>Эталонное фото</span>
        {selectedEmployee?.has_photo && !(showReferencePhotoCapture && (referencePhotoDraft || referenceCameraStream)) && referencePhoto.isLoading && <small>Загружаем эталонное фото…</small>}
        {selectedEmployee?.has_photo && !(showReferencePhotoCapture && (referencePhotoDraft || referenceCameraStream)) && referencePhoto.data && <SelectedFilePreview file={referencePhoto.data} mediaKind="photo" title={`Эталонное фото · ${selectedEmployee.personnel_number}`} />}
        {selectedEmployee?.has_photo && !(showReferencePhotoCapture && (referencePhotoDraft || referenceCameraStream)) && referencePhoto.isError && <small className="reference-photo-error" role="alert">{referencePhoto.error?.message || 'Не удалось загрузить эталонное фото.'}</small>}
        {selectedEmployee && showReferencePhotoCapture && referencePhotoDraft && <SelectedFilePreview file={referencePhotoDraft} mediaKind="photo" title={`Снимок · ${selectedEmployee.personnel_number}`} />}
        {selectedEmployee && showReferencePhotoCapture && !referencePhotoDraft && referenceCameraStream && <div className="camera-preview-frame employee-reference-camera"><video ref={referenceCameraPreviewRef} autoPlay playsInline muted aria-label="Предпросмотр снимка сотрудника" /></div>}
        {selectedEmployee && !selectedEmployee.has_photo && !referencePhotoDraft && !referenceCameraStream && <div className="employee-photo-empty">Эталонное фото отсутствует</div>}
      </div>
      <div className="field employee-photo-action-field"><span>Фото</span>
        {!selectedEmployee && <small>Сначала выберите сотрудника.</small>}
        {selectedEmployee && !showReferencePhotoCapture && <button className="button button-secondary" type="button" data-testid="button-connect-employee-camera" disabled={isSavingReferencePhoto} onClick={openReferencePhotoCapture}><Camera size={14} /> {selectedEmployee.has_photo ? 'Заменить фото' : 'Сделать фото'}</button>}
        {selectedEmployee && showReferencePhotoCapture && referenceCameraStream && !referencePhotoDraft && <>
          <button className="button button-primary" type="button" data-testid="button-capture-employee-photo" onClick={captureReferencePhoto}><Camera size={14} /> Сделать снимок</button>
          <button className="button button-secondary" type="button" onClick={stopReferencePhotoCamera}>Отключить камеру</button>
        </>}
        {selectedEmployee && showReferencePhotoCapture && referencePhotoDraft && <>
          <button className="button button-secondary" type="button" disabled={isSavingReferencePhoto} onClick={() => { setReferencePhotoDraft(null); setReferencePhotoSaveError(''); setReferencePhotoSaveMessage(''); openReferencePhotoCapture(); }}>Переснять</button>
          <button className="button button-primary" type="button" disabled={isSavingReferencePhoto} data-testid="button-save-employee-photo" onClick={saveReferencePhoto}>{isSavingReferencePhoto ? 'Сохраняем…' : selectedEmployee.has_photo ? 'Сохранить замену' : 'Сохранить фото'}</button>
        </>}
        {selectedEmployee && showReferencePhotoCapture && !referenceCameraStream && !referencePhotoDraft && <button className="button button-secondary" type="button" data-testid="button-retry-employee-camera" disabled={isSavingReferencePhoto} onClick={connectReferencePhotoCamera}><Camera size={14} /> Подключить камеру</button>}
        {selectedEmployee?.has_photo && <small>При замене прежнее фото сохраняется в закрытой коллекции dataset.</small>}
        {selectedEmployee && showReferencePhotoCapture && <button className="employee-photo-cancel" type="button" disabled={isSavingReferencePhoto} onClick={closeReferencePhotoCapture}>Отмена</button>}
        {referenceCameraError && <small className="reference-photo-error" role="alert">{referenceCameraError}</small>}
        {referencePhotoSaveError && <small className="reference-photo-error" role="alert">{referencePhotoSaveError}</small>}
        {referencePhotoSaveMessage && <small className="employee-photo-saved" role="status">{referencePhotoSaveMessage}</small>}
      </div>
      <div className="video-source-picker" role="group" aria-label="Источник видео">
        <button type="button" className={`video-source-option ${videoSource === 'upload' ? 'video-source-active' : ''}`} aria-pressed={videoSource === 'upload'} disabled={isRecording} onClick={() => switchVideoSource('upload')}><Play size={14} /> Загрузить файл</button>
        <button type="button" className={`video-source-option ${videoSource === 'camera' ? 'video-source-active' : ''}`} aria-pressed={videoSource === 'camera'} disabled={isRecording} onClick={() => switchVideoSource('camera')}><Camera size={14} /> Записать с камеры</button>
      </div>
      {videoSource === 'upload' ? <>
        <label className="field upload-field"><span>Видео для проверки</span><input type="file" accept="video/mp4,video/webm,video/quicktime,.mp4,.m4v,.mov,.webm,.mkv" data-testid="input-video-file" onChange={(e) => { clearRecording(); setVideoFile(e.target.files?.[0] || null); }} /><small>MP4, MOV, WebM или MKV · до 100 МБ</small></label>
        <SelectedFilePreview file={videoFile} mediaKind="video" />
      </> : <div className="camera-capture">
        <div className="camera-toolbar">
          <div><span className={`camera-status-dot ${cameraStream ? 'camera-status-on' : ''}`} />{isRecording ? `Идёт запись · ${String(Math.floor(recordingSeconds / 60)).padStart(2, '0')}:${String(recordingSeconds % 60).padStart(2, '0')}` : isConvertingRecording ? 'Сохраняем в MP4…' : cameraStream ? 'Камера подключена' : 'Камера не подключена'}</div>
          {!cameraStream
            ? <button className="button button-secondary" type="button" disabled={isConvertingRecording} onClick={startCamera}><Camera size={14} /> Подключить камеру</button>
            : <div className="camera-actions">
              {!isRecording
                ? <button className="button button-primary" type="button" disabled={isConvertingRecording} onClick={startRecording}><Play size={14} /> Записать видео</button>
                : <button className="button button-danger" type="button" onClick={stopRecording}><CircleStop size={14} /> Остановить запись</button>}
              <button className="button button-secondary camera-off-button" type="button" disabled={isRecording || isConvertingRecording} onClick={turnOffCamera}>Выключить камеру</button>
            </div>}
        </div>
        <div className="camera-preview-frame">
          {cameraStream
            ? <video ref={cameraPreviewRef} autoPlay playsInline muted aria-label="Предпросмотр веб-камеры" />
            : <div className="camera-placeholder"><Camera size={26} /><span>Разрешите доступ к камере, чтобы увидеть видеопоток</span></div>}
          {isRecording && <span className="camera-recording-badge"><i /> REC</span>}
        </div>
        {cameraError && <div className="camera-error" role="alert">{cameraError}</div>}
        {videoFile && <div className="camera-recording-result">
          <SelectedFilePreview file={videoFile} mediaKind="video" title="Записанное видео" />
          {recordingUrl && <a className="button button-secondary camera-download" href={recordingUrl} download={videoFile.name}><Download size={14} /> Сохранить видео на компьютер</a>}
        </div>}
         <small className="camera-help">{isConvertingRecording ? 'Преобразуем видеозапись в MP4 640 × 480. Это может занять немного времени.' : 'Камера должна поддерживать минимум 640 × 480 и 15 кадров/с. После остановки запись сохраняется в MP4 640 × 480.'}</small>
      </div>}
      <label className="video-index-opt-in">
        <input
          type="checkbox"
          data-testid="input-enroll-video-embeddings"
          checked={enrollVideoEmbeddings}
          onChange={(event) => setEnrollVideoEmbeddings(event.target.checked)}
        />
        <span>
          <strong>Пополнить индекс ResNet50 + ArcFace</strong>
          <small>
            Добавить признаки выбранных кадров только после успешной верификации.
            Видео и эмбеддинги сессии не сохраняются.
          </small>
        </span>
      </label>
      <button className="button button-primary" disabled={create.isPending || isConvertingRecording || !selectedEmployee?.has_photo || referencePhoto.isLoading || referencePhoto.isError || !videoFile} data-testid="button-submit-verification">{isConvertingRecording ? 'Сохраняем MP4…' : create.isPending ? 'Анализ видео…' : <><Fingerprint size={15} /> Запустить проверку</>}</button>
    </form><p className="upload-privacy-note">Сохранённый снимок используется как защищённое эталонное фото выбранного сотрудника. Фото и видео самой сессии не сохраняются после обработки.</p>
      {create.isError && <div className="form-alert form-alert-error" role="alert"><AlertTriangle size={15} />{create.error?.message || 'Не удалось создать сессию. Проверьте файлы и повторите попытку.'}</div>}</div>}
    {!showForm && create.data && <div className="success-result"><div><span className="eyebrow">Результат верификации</span><strong className="display">{labelStatus(create.data.decision)}</strong><span className="muted-text">Совпадение {create.data.match_score == null ? 'не оценено' : pct(create.data.match_score)} · проверка живости {create.data.liveness_score == null ? 'не оценена' : pct(create.data.liveness_score)}</span></div><StatusPill tone={toneForStatus(create.data.status)}>{labelStatus(create.data.status)}</StatusPill></div>}
    {!showForm && videoCompletionSessionId && <div className={`video-completion-notice ${videoCompletionStatus === 'error' ? 'video-completion-error' : ''}`} role={videoCompletionStatus === 'error' ? 'alert' : 'status'}>
      <span>{videoCompletionStatus === 'pending' ? 'Завершаем приём видео…' : videoCompletionStatus === 'complete' ? 'Приём видео подтверждён.' : `Сессия создана, но завершение приёма видео не подтверждено: ${videoCompletionError}`}</span>
      {videoCompletionStatus === 'error' && <button className="button button-secondary" type="button" disabled={completeVideo.isPending} onClick={retryVideoCompletion}>{completeVideo.isPending ? 'Повтор…' : 'Повторить'}</button>}
    </div>}
    <div className="results-layout">
      <Panel kicker={`${filteredSessions.length} из ${list.data?.length || 0} сессий`} title="Реестр сессий" className="results-panel">
        <div className="verification-filters" role="search" aria-label="Фильтры реестра сессий">
          <div className="verification-filter-grid">
            <label className="verification-filter verification-filter-search">
              <span>Поиск по сотруднику, ID или аудиту</span>
              <span className="verification-search-control"><Search size={14} aria-hidden="true" /><input data-testid="input-verification-search" value={filters.query} onChange={(event) => setFilters({ ...filters, query: event.target.value })} placeholder="Введите идентификатор" /></span>
            </label>
            <label className="verification-filter">
              <span>Решение</span>
              <select data-testid="select-verification-decision" value={filters.decision} onChange={(event) => setFilters({ ...filters, decision: event.target.value })}>
                <option value="all">Все решения</option><option value="match">Совпадение</option><option value="review">Ручная проверка</option><option value="no_match">Нет совпадения</option>
              </select>
            </label>
            <label className="verification-filter">
              <span>Статус сессии</span>
              <select data-testid="select-verification-status" value={filters.status} onChange={(event) => setFilters({ ...filters, status: event.target.value })}>
                <option value="all">Все статусы</option><option value="verified">Подтверждена</option><option value="manual_review">Ручная проверка</option><option value="not_verified">Не подтверждена</option><option value="not_evaluable">Оценка невозможна</option><option value="insufficient_quality">Недостаточное качество</option>
              </select>
            </label>
            <label className="verification-filter">
              <span>Живость</span>
              <select data-testid="select-verification-liveness" value={filters.liveness} onChange={(event) => setFilters({ ...filters, liveness: event.target.value })}>
                <option value="all">Любой результат</option><option value="live">Подтверждена</option><option value="not_live">Не подтверждена</option><option value="unknown">Не оценена</option>
              </select>
            </label>
            <label className="verification-filter">
              <span>Период</span>
              <select data-testid="select-verification-period" value={filters.period} onChange={(event) => setFilters({ ...filters, period: event.target.value })}>
                <option value="all">За всё время</option><option value="day">Последние 24 часа</option><option value="week">Последние 7 дней</option><option value="month">Последние 30 дней</option><option value="quarter">Последние 90 дней</option>
              </select>
            </label>
          </div>
          <div className="verification-filter-footer">
            <span className="muted-text" data-testid="text-verification-filter-count">Показано {filteredSessions.length} из {list.data?.length || 0} загруженных сессий</span>
            <button className="button button-secondary verification-reset-button" type="button" data-testid="button-reset-verification-filters" disabled={!hasActiveFilters} onClick={() => setFilters({ query: '', decision: 'all', status: 'all', liveness: 'all', period: 'all', minMatchScore: '', minLivenessScore: '' })}><X size={13} /> Сбросить</button>
          </div>
        </div>
        <div className="table-wrap mobile-scroll">
          <table className="data-table">
            <thead><tr><th>Сессия</th><th>Статус</th><th>Совпадение</th><th>Проверка живости</th><th>Качество захвата</th><th>Ссылка аудита</th></tr></thead>
            <tbody>
              {list.isLoading ? <tr><td colSpan="6"><LoadingRows rows={5} /></td></tr>
                : list.isError ? <tr><td colSpan="6"><ErrorState onRetry={list.refetch} /></td></tr>
                  : filteredSessions.length ? filteredSessions.map((item) => <tr key={item.id} className={selectedId === item.id ? 'row-selected' : ''} onClick={() => { deleteSession.reset(); setSelectedId(item.id); }} data-testid={`row-verification-${item.id}`}><td><div className="record-cell"><span className="record-avatar fingerprint-avatar"><Fingerprint size={14} /></span><span><b>{item.employee_ref}</b><small className="mono">{item.id}</small></span></div></td><td><StatusPill tone={toneForStatus(item.status)}>{labelStatus(item.status)}</StatusPill></td><td><strong className="score">{item.match_score == null ? '—' : pct(item.match_score)}</strong></td><td><strong className="score">{item.liveness_score == null ? '—' : pct(item.liveness_score)}</strong></td><td><strong className="score">{item.quality_score == null ? '—' : pct(item.quality_score)}</strong></td><td className="mono muted-text">{item.audit_ref}</td></tr>)
                    : <tr><td colSpan="6"><EmptyState icon={Fingerprint} title={list.data?.length ? 'По фильтрам ничего не найдено' : 'Сессий пока нет'} text={list.data?.length ? 'Измените параметры фильтрации или сбросьте их.' : 'Создайте сессию, чтобы сохранить проверяемый результат.'} /></td></tr>}
            </tbody>
          </table>
        </div>
      </Panel>
      <VerificationDetail
        session={detail.data}
        sessionSummary={selectedSummary}
        loading={detail.isLoading && Boolean(selectedId)}
        error={detail.isError}
        onRetry={detail.refetch}
        onDelete={deleteSelectedSession}
        deletePending={deleteSession.isPending}
        deleteError={deleteSession.isError && deleteSession.variables?.sessionId === selectedId}
        onRequestReview={sendToReviewQueue}
        requestReviewPending={reviewRequestForSelected && requestReview.isPending}
        requestReviewSuccess={reviewRequestForSelected && requestReview.isSuccess}
        requestReviewError={reviewRequestForSelected && requestReview.isError}
      />
    </div>
  </PageFrame>;
}

function VerificationDetail({
  session,
  sessionSummary,
  loading,
  error,
  onRetry,
  onDelete,
  deletePending,
  deleteError,
  onRequestReview,
  requestReviewPending,
  requestReviewSuccess,
  requestReviewError,
}) {
  if (!session && !loading) return <Panel className="detail-panel detail-empty"><EmptyState icon={Fingerprint} title="Выберите сессию" text="Откройте сессию, чтобы изучить оценки и аудиторскую цепочку." /></Panel>;
  const isAutomaticMatch = session?.status === 'verified' && session?.decision === 'match';
  const hasOpenReviewTask = (session?.review_tasks || []).some((task) => ['open', 'in_review'].includes(task.status));
  const reviewAlreadyRequested = hasOpenReviewTask || requestReviewSuccess;
  return <Panel
    kicker="Трасса сессии"
    title={sessionSummary ? sessionSummary.employee_ref : 'Загрузка сессии'}
    className="detail-panel"
    action={(sessionSummary || session) && <button className="button button-danger" type="button" disabled={deletePending} data-testid="button-delete-verification" onClick={onDelete}><Trash2 size={14} />{deletePending ? 'Удаление…' : 'Удалить'}</button>}
  >
    {loading ? <LoadingRows rows={5} /> : error ? <ErrorState onRetry={onRetry} /> : <div className="detail-content">
      {deleteError && <div className="form-alert form-alert-error" role="alert"><AlertTriangle size={15} /> Не удалось удалить сессию. Повторите попытку.</div>}
      <div className="verification-decision"><div className={`decision-orb orb-${toneForStatus(session.status)}`}><Fingerprint size={23} /></div><div><span className="eyebrow">Итоговое решение</span><strong className="display">{labelStatus(session.decision)}</strong><StatusPill tone={toneForStatus(session.status)}>{labelStatus(session.status)}</StatusPill></div></div>
      {session.status === 'not_evaluable' && <div className="not-evaluable-notice" role="status"><strong>Система не смогла выполнить достоверную оценку.</strong><p>{session.not_evaluable_reason || 'Проверьте эталон, качество видео и доступность компонентов обработки.'}</p><span>Запросите повторную запись или продолжите ручную проверку.</span><Link href="/review" className="text-link">Открыть очередь проверки <ChevronRight size={14} /></Link></div>}
      {session.status === 'insufficient_quality' && <div className="insufficient-quality-notice" role="status"><strong>Входные данные прочитаны, но качество не прошло порог.</strong><p>{session.quality?.blocking_issues?.join(' ') || 'Проверьте эталонное фото и качество видеозаписи.'}</p><span>{session.match_score == null && session.liveness_score == null ? 'Для этой сессии числовые оценки не сохранены. Чтобы получить новый расчёт, создайте сессию повторно с исходными файлами.' : 'Доступные оценки показаны ниже; запросите повторную запись или продолжите ручную проверку.'}</span><Link href="/review" className="text-link">Открыть очередь проверки <ChevronRight size={14} /></Link></div>}
      {session.arcface_video_enrollment && <div
        className={`video-index-result ${session.arcface_video_enrollment.status === 'updated' ? 'video-index-result-success' : ''}`}
        role="status"
        data-testid="status-video-index-enrollment"
      >
        <strong>Индекс ResNet50 + ArcFace</strong>
        <span>{session.arcface_video_enrollment.message}</span>
        {session.arcface_video_enrollment.added_vectors > 0 && <small>
          Добавлено: {session.arcface_video_enrollment.added_vectors} · всего видео-векторов: {session.arcface_video_enrollment.video_vectors_count}
        </small>}
      </div>}
      {isAutomaticMatch && <div className="verification-review-request">
        <div className="verification-review-copy">
          <strong>Нужна дополнительная проверка?</strong>
          <small>Ручной запрос добавит задачу специалисту, не изменяя решение системы «Совпадение».</small>
        </div>
        <div className="verification-review-action">
          <button
            className={`button ${reviewAlreadyRequested ? 'button-secondary' : 'button-primary'}`}
            type="button"
            disabled={reviewAlreadyRequested || requestReviewPending}
            data-testid={`button-request-verification-review-${session.id}`}
            onClick={() => onRequestReview(session.id)}
          >
            <ClipboardCheck size={14} />
            {requestReviewPending ? 'Отправляем…' : reviewAlreadyRequested ? 'Уже в очереди проверки' : 'Отправить в очередь проверки'}
          </button>
          {requestReviewError && <small className="verification-review-error" role="alert">Не удалось отправить сессию в очередь. Повторите попытку.</small>}
        </div>
      </div>}
      <div className="score-grid">{[['Совпадение', session.match_score], ['Проверка живости', session.liveness_score]].map(([label, value]) => {
        const score = value == null ? 0 : normalizedScore(value);
        return <div className="score-tile" key={label}><span className="eyebrow">{label}</span><b className="display">{value == null ? '—' : pct(value)}</b><div className="score-line"><span style={{ width: `${score * 100}%`, backgroundColor: scoreBarColor(value) }} /></div></div>;
      })}<div className="score-tile duration-tile"><span className="eyebrow">Время проверки</span><b className="display">{formatVerificationElapsed(session.verification_elapsed_seconds)}</b><small className="muted-text">обработка на сервере</small></div></div><div className="detail-meta"><span><small>Алгоритм</small><b className="mono">{session.algorithm_version}</b></span><span><small>Создано</small><b>{fmtDate(session.created_at)}</b></span><span><small>Ссылка аудита</small><b className="mono">{session.audit_ref}</b></span></div>{session.attack_type && <div className="attack-flag"><AlertTriangle size={14} /> Сигнал атаки: <b>{labelStatus(session.attack_type)}</b></div>}<VerificationAnalysis session={session} /><div className="disclaimer"><ShieldCheck size={15} /><span>{session.disclaimer || 'Биометрическая оценка является исследовательской, не подтверждает личность юридически и не является медицинским заключением. Неоднозначные результаты требуют ручной проверки.'}</span></div>
    </div>}
  </Panel>;
}

function VerificationAnalysis({ session }) {
  const face = session.face_match;
  const hasFaceDetails = face && Object.keys(face).length > 0;
  const signals = session.liveness?.liveness_signals;
  const hasSignals = signals && Object.keys(signals).length > 0;
  const multiFaceCount = Number(face?.multi_candidate_frames ?? 0);
  const multiFaceRatio = Number(face?.multi_candidate_ratio ?? 0);
  const multiFaceBlockThreshold = face?.multi_face_block_threshold;
  const multiFaceBlocked = face?.multi_face_blocked === true;
  const selectedFrameIds = Array.isArray(face?.selected_frame_ids) ? face.selected_frame_ids : [];
  const frameQuality = Array.isArray(face?.frame_quality) ? face.frame_quality : [];
  const rejectedFrames = frameQuality.filter((item) => item?.eligible === false);
  if (!hasFaceDetails && !hasSignals) return null;

  const signalNames = {
    blink: 'Моргания',
    head_motion: 'Движение головы',
    texture: 'Текстура',
    depth: 'Глубина',
    rppg: 'rPPG',
  };
  const multiFaceStrategyNames = {
    select_closest_to_reference: 'выбор по сходству с эталоном',
    max: 'выбор самого крупного лица',
    skip_frame: 'пропуск кадра с несколькими кандидатами',
  };
  const frameReasonNames = {
    blur: 'размытие',
    underexposure: 'недостаточная яркость',
    overexposure: 'избыточная яркость',
    low_contrast: 'низкий контраст',
    uneven_illumination: 'неравномерное освещение или тень',
    face_too_small: 'слишком маленькое лицо',
    pose_out_of_range: 'сильный поворот или наклон',
    empty_face_crop: 'пустая область лица',
  };
  const multiFaceStrategyName = multiFaceStrategyNames[face.multi_face_strategy] || face.multi_face_strategy;
  return <details className="analysis-details" open={multiFaceCount > 0}>
    <summary>Подробные расчёты и сигналы</summary>
    {hasFaceDetails && <section className="analysis-section">
      <h4>Сходство лица</h4>
      <p>Решение: {labelStatus(face.decision)} · порог: {pct(face.threshold)} · обнаружено лиц: {face.detected_frames} из {face.total_frames} кадров видео</p>
      <p>Детектор: {face.detector_backend} · признаки: {face.embedding_method} · лучший кадр: {face.best_frame_id ?? '—'}</p>
      {face.frame_selection_strategy && <p>Выборка: равномерно по всему ролику · медиана по {selectedFrameIds.length} кадрам (цель: {face.requested_frame_count}) · пригодных: {face.usable_frame_count ?? 0} · исключено: {face.excluded_frame_count ?? 0}.</p>}
      {selectedFrameIds.length > 0 && <p>Кадры, вошедшие в медиану: {selectedFrameIds.join(', ')}.</p>}
      {rejectedFrames.length > 0 && <details className="frame-scores">
        <summary>Исключённые кадры по качеству ({rejectedFrames.length})</summary>
        <p>{rejectedFrames.map((item) => `${item.frame_id}: ${(item.reasons || []).map((reason) => frameReasonNames[reason] || reason).join(', ')}`).join(' · ')}</p>
      </details>}
      {multiFaceCount > 0 && <p className="analysis-warning analysis-warning-multiface" role="alert"><AlertTriangle size={13} aria-hidden="true" /> Обнаружено несколько лиц на {multiFaceCount === 1 ? '1 кадре' : `${multiFaceCount} кадрах`} выборки ({(multiFaceRatio * 100).toFixed(1)}%). {multiFaceBlocked ? `Доля превышает порог блокировки ${multiFaceBlockThreshold == null ? '—' : pct(multiFaceBlockThreshold)}; сессия направлена на проверку.` : face.multi_face_blocked === false ? `Доля не превышает порог блокировки ${multiFaceBlockThreshold == null ? '—' : pct(multiFaceBlockThreshold)}; отказа только по этому признаку нет.` : ''}</p>}
      {face.multi_face_strategy && <p>Стратегия выбора: {multiFaceStrategyName}. Пропущено кадров: {face.skipped_multi_candidate_frames ?? 0}.</p>}
      {face.per_frame_scores?.length > 0 && <details className="frame-scores">
        <summary>Покадровые оценки ({face.per_frame_scores.length})</summary>
        <p>{face.per_frame_scores.map((score) => pct(score)).join(' · ')}</p>
      </details>}
    </section>}
    {hasSignals && <section className="analysis-section">
      <h4>Сигналы живости</h4>
      {Object.entries(signals).map(([key, signal]) => <div className="analysis-row" key={key}>
        <span>{signalNames[key] || key}</span>
        <b>{pct(signal.score)}</b>
        <small>{signal.passed ? 'пройдено' : 'не пройдено / недостаточно данных'}</small>
        {signal.details?.reason && <small>{signal.details.reason}</small>}
      </div>)}
    </section>}
    {session.health_id && <p className="analysis-note">{session.health_id.error || 'HEALTH_ID не рассчитан.'}</p>}
    <small className="analysis-note">ID конфигурации: <span className="mono">{session.config_hash}</span></small>
  </details>;
}

function ReviewPage() {
  const [status, setStatus] = useState('all');
  const [activeTask, setActiveTask] = useState(null);
  const [note, setNote] = useState('');
  const [groundTruth, setGroundTruth] = useState({ match: null, liveness: null });
  const { reviewMedia, releaseReviewMedia } = useContext(ReviewMediaContext);
  const params = useMemo(() => ({ limit: 100, ...(status !== 'all' ? { status } : {}) }), [status]);
  const tasks = useListReviewTasks(params);
  const resolve = useResolveReviewTask();
  const verificationId = activeTask?.verification_session_id || '';
  const verification = useGetVerificationSession(verificationId || 'none', {
    query: {
      enabled: activeTask?.kind === 'verification' && Boolean(verificationId),
      queryKey: getGetVerificationSessionQueryKey(verificationId || 'none'),
    },
  });
  const media = verificationId ? reviewMedia[verificationId] : null;
  useEffect(() => {
    setGroundTruth({
      match: verification.data?.ground_truth_match ?? null,
      liveness: verification.data?.ground_truth_liveness ?? null,
    });
  }, [activeTask?.id, verification.data?.id]);
  const decide = (resolution) => {
    if (!activeTask) return;
    const submitGroundTruth = activeTask.kind === 'verification' && resolution !== 'request_retry';
    if (submitGroundTruth && (groundTruth.match === null || groundTruth.liveness === null)) return;
    const data = { resolution, reviewer_note: note };
    if (submitGroundTruth) {
      data.ground_truth_match = groundTruth.match;
      data.ground_truth_liveness = groundTruth.liveness;
    }
    resolve.mutate({ taskId: activeTask.id, data }, { onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: getListReviewTasksQueryKey(params) });
      queryClient.invalidateQueries({ queryKey: getGetDashboardSummaryQueryKey() });
      queryClient.invalidateQueries({ queryKey: getListActivityQueryKey({ limit: 8 }) });
      if (verificationId) queryClient.invalidateQueries({ queryKey: getGetVerificationSessionQueryKey(verificationId) });
      if (verificationId && resolution !== 'request_retry') releaseReviewMedia(verificationId);
      setActiveTask(null);
      setNote('');
      setGroundTruth({ match: null, liveness: null });
    } });
  };
  return <PageFrame eyebrow="Контроль оператора / очередь проверки" title="Очередь проверки" intro="Разбирайте неоднозначные случаи с учётом контекста и передавайте окончательное решение специалисту." action={<div className="queue-count"><span className="pulse-dot" />{tasks.data?.filter((task) => task.status !== 'resolved').length || 0} требуют внимания</div>}>
    <div className="review-toolbar"><div className="filter-tabs">{['all', 'open', 'in_review', 'resolved'].map((item) => <button key={item} className={status === item ? 'filter-active' : ''} data-testid={`button-review-filter-${item}`} onClick={() => setStatus(item)}>{item === 'all' ? 'Все' : labelStatus(item)}<span>{item === 'all' ? (tasks.data?.length || 0) : (tasks.data || []).filter((task) => task.status === item).length}</span></button>)}</div><div className="queue-legend"><span><i className="legend-dot legend-high" />Высокий приоритет</span><span><i className="legend-dot legend-medium" />Обычный приоритет</span></div></div>
    <div className="review-layout"><Panel kicker="Очередь решений" title="Случаи для проверки специалистом" className="tasks-panel">{tasks.isLoading ? <LoadingRows rows={5} /> : tasks.isError ? <ErrorState onRetry={tasks.refetch} /> : tasks.data?.length ? <div className="task-list">{tasks.data.map((task) => <div className={`task-card ${activeTask?.id === task.id ? 'task-active' : ''}`} key={task.id}>
      <button type="button" className="task-card-select" aria-pressed={activeTask?.id === task.id} data-testid={`button-task-${task.id}`} onClick={() => { setActiveTask(task); setNote(''); }}><div className={`priority-bar priority-${task.priority}`} /><div className="task-main"><div className="task-top"><StatusPill tone={toneForStatus(task.status)}>{labelStatus(task.status)}</StatusPill><span className="task-kind"><FileCheck2 size={13} />{labelTaskKind(task.kind)}</span><span className="mono task-time">{fmtDate(task.created_at)}</span></div><h3>{task.summary}</h3><div className="task-subject"><span className="record-avatar tiny-avatar">{initials(task.subject_ref)}</span><span className="mono">{task.subject_ref}</span></div></div><ChevronRight size={17} className="task-arrow" /></button>
      {task.kind === 'verification' && task.verification_session_id && <Link href={`/verification/${task.verification_session_id}`} className="task-trace-link" title="Перейти к трассе сессии и посмотреть расшифровку" data-testid={`link-session-transcript-${task.verification_session_id}`}><History size={13} /><span>Трасса и расшифровка</span></Link>}
    </div>)}</div> : <EmptyState icon={ClipboardCheck} title="Очередь пуста" text="Подходящих задач нет. Система ожидает новые события." />}</Panel><ReviewDecision task={activeTask} note={note} setNote={setNote} onDecide={decide} pending={resolve.isPending} error={resolve.isError} verificationSession={verification.data} sessionLoading={verification.isLoading} media={media} groundTruth={groundTruth} setGroundTruth={setGroundTruth} /></div>
  </PageFrame>;
}

function GroundTruthChoice({ title, value, onChange, testId }) {
  return <div className="ground-truth-choice" role="group" aria-label={title}>
    <span>{title}</span>
    <div>
      <button type="button" className={value === true ? 'ground-truth-selected' : ''} aria-pressed={value === true} data-testid={`${testId}-yes`} onClick={() => onChange(true)}>Да</button>
      <button type="button" className={value === false ? 'ground-truth-selected' : ''} aria-pressed={value === false} data-testid={`${testId}-no`} onClick={() => onChange(false)}>Нет</button>
    </div>
  </div>;
}

function ReviewDecision({ task, note, setNote, onDecide, pending, error, verificationSession, sessionLoading, media, groundTruth, setGroundTruth }) {
  if (!task) return <Panel className="decision-panel decision-empty"><EmptyState icon={UserRound} title="Выберите задачу" text="Здесь появятся контекст случая и действия для принятия решения." /></Panel>;
  const biometricTask = task.kind === 'verification';
  const labelsChosen = groundTruth.match !== null && groundTruth.liveness !== null && (!biometricTask || Boolean(task.verification_session_id));
  return <Panel kicker="Рабочее место специалиста" title="Решение по случаю" className="decision-panel"><div className="decision-subject"><div className="subject-avatar">{initials(task.subject_ref)}</div><div><span className="eyebrow">{labelTaskKind(task.kind)} · случай</span><h3>{task.subject_ref}</h3><StatusPill tone={toneForStatus(task.priority === 'high' ? 'attention' : task.status)}>{labelStatus(task.priority)} приоритет</StatusPill></div></div><div className="review-summary"><span className="eyebrow">Сводка системы</span><p>{task.summary}</p><div className="review-context"><span><Clock3 size={14} /> Создано {fmtDate(task.created_at)}</span><span><Database size={14} /> Идентификатор случая сохранён</span></div></div>{biometricTask && <section className="review-biometric-block" aria-label="Данные биометрической проверки">
    <div className="eyebrow">Результат модели · исходные значения не изменяются</div>
    {sessionLoading ? <LoadingRows rows={2} /> : verificationSession ? <div className="review-model-values">
      <div><small>Сырой cosine score</small><b className="mono">{verificationSession.cosine_distance == null ? '—' : Number(verificationSession.cosine_distance).toFixed(6)}</b></div>
      <div><small>Решение по совпадению</small><b>{verificationSession.is_match == null ? '—' : verificationSession.is_match ? 'Да' : 'Нет'}</b></div>
      <div><small>Решение по живости</small><b>{verificationSession.is_live == null ? '—' : verificationSession.is_live ? 'Да' : 'Нет'}</b></div>
      <div><small>Оценка совпадения</small><b>{pct(verificationSession.match_score)}</b></div>
      <div><small>Оценка живости</small><b>{pct(verificationSession.liveness_score)}</b></div>
    </div> : <div className="media-unavailable">{task.verification_session_id ? 'Не удалось загрузить сессию верификации.' : 'Для этой старой задачи не связана сессия верификации.'}</div>}
    {media ? <div className="review-media-grid">
      {media.photo && <SelectedFilePreview file={media.photo} mediaKind="photo" title="Эталонное фото" />}
      {media.video && <SelectedFilePreview file={media.video} mediaKind="video" title="Видео с кадрами для просмотра" />}
    </div> : <div className="media-unavailable">Файлы не хранятся на сервере. Фото и видео доступны только в активной вкладке браузера, где была создана эта сессия.</div>}
    <div className="ground-truth-fields">
      <div className="eyebrow">Разметка оператора · Ground Truth</div>
      <GroundTruthChoice title="Это тот же человек?" value={groundTruth.match} testId="ground-truth-match" onChange={(match) => setGroundTruth((current) => ({ ...current, match }))} />
      <GroundTruthChoice title="На видео живой человек, не фото/маска?" value={groundTruth.liveness} testId="ground-truth-liveness" onChange={(liveness) => setGroundTruth((current) => ({ ...current, liveness }))} />
    </div>
  </section>}<label className="field"><span>Примечание специалиста <em>необязательно</em></span><textarea data-testid="input-reviewer-note" value={note} onChange={(e) => setNote(e.target.value)} placeholder="Укажите контекст принятого решения…" rows="4" /></label>{error && <div className="form-alert form-alert-error"><AlertTriangle size={14} /> Не удалось сохранить решение. Повторите попытку.</div>}<div className="decision-actions"><button className="button button-secondary" disabled={pending} data-testid="button-request-retry" onClick={() => onDecide('request_retry')}><RefreshCw size={14} /> Запросить повторную проверку</button><button className="button button-danger" disabled={pending || (biometricTask && !labelsChosen)} data-testid="button-reject-review" onClick={() => onDecide('reject')}><X size={14} /> Отклонить</button><button className="button button-primary" disabled={pending || (biometricTask && !labelsChosen)} data-testid="button-approve-review" onClick={() => onDecide('approve')}><Check size={14} /> Подтвердить</button></div><div className="disclaimer"><ShieldCheck size={15} /><span>Разметка оператора сохраняется отдельно от исходного решения модели и используется для расчёта качества.</span></div></Panel>;
}

function VerificationMetricsPanel() {
  const [windowDays, setWindowDays] = useState(7);
  const history = useListVerificationMetrics({ limit: 20 });
  const calculate = useCalculateVerificationMetrics();
  const calculateNow = () => calculate.mutate({ params: { window_days: windowDays } }, {
    onSuccess: () => queryClient.invalidateQueries({ queryKey: getListVerificationMetricsQueryKey({ limit: 20 }) }),
  });
  const latest = history.data?.[0];
  const calculationDetails = latest?.calculation_details;
  const biometric = calculationDetails?.biometric;
  const liveness = calculationDetails?.liveness;
  const hasBiometricCases = Boolean(biometric?.genuine_cases && biometric?.impostor_cases);
  const hasLivenessCases = Boolean(liveness?.spoof_cases && liveness?.bona_fide_cases);
  const showRate = (key, hasCases) => calculationDetails && !hasCases ? '—' : metricPercent(latest[key]);
  const historyRate = (item, key, hasCases) => item.calculation_details && !hasCases ? '—' : metricPercent(item[key]);
  const hasAnyLabeledCases = calculationDetails
    ? Boolean(biometric?.labeled_cases || liveness?.labeled_cases)
    : Boolean(latest?.total_samples);
  const rateItems = latest ? [
    ['FAR', showRate('far', Boolean(biometric?.impostor_cases)), 'Ложный допуск чужого человека'],
    ['FRR', showRate('frr', Boolean(biometric?.genuine_cases)), 'Ложный отказ своему человеку'],
    ['CER', showRate('cer', hasBiometricCases), 'Среднее FAR и FRR'],
    ['APCER', showRate('apcer', Boolean(liveness?.spoof_cases)), 'Атака принята за живого человека'],
    ['BPCER', showRate('bpcer', Boolean(liveness?.bona_fide_cases)), 'Живой человек отклонён как атака'],
    ['ACER', showRate('acer', hasLivenessCases), 'Среднее APCER и BPCER'],
  ] : [];
  const breakdownItems = biometric && liveness ? [
    {
      label: 'FAR',
      formula: 'FP / (FP + TN)',
      substitution: `${biometric.fp} / (${biometric.fp} + ${biometric.tn})`,
      numerator: biometric.fp,
      denominator: biometric.impostor_cases,
      value: latest.far,
      population: 'Все размеченные пары с другим человеком.',
    },
    {
      label: 'FRR',
      formula: 'FN / (FN + TP)',
      substitution: `${biometric.fn} / (${biometric.fn} + ${biometric.tp})`,
      numerator: biometric.fn,
      denominator: biometric.genuine_cases,
      value: latest.frr,
      population: 'Все размеченные пары с тем же человеком.',
    },
    {
      label: 'APCER',
      formula: 'Атаки, принятые за живого / все атаки',
      substitution: `${liveness.spoof_accepted_as_live} / ${liveness.spoof_cases}`,
      numerator: liveness.spoof_accepted_as_live,
      denominator: liveness.spoof_cases,
      value: latest.apcer,
      population: 'Фото, маски и другие размеченные атаки.',
    },
    {
      label: 'BPCER',
      formula: 'Живые, отклонённые как атака / все живые',
      substitution: `${liveness.bona_fide_rejected_as_spoof} / ${liveness.bona_fide_cases}`,
      numerator: liveness.bona_fide_rejected_as_spoof,
      denominator: liveness.bona_fide_cases,
      value: latest.bpcer,
      population: 'Все размеченные предъявления живого человека.',
    },
  ] : [];
  return <Panel kicker="Биометрическая аналитика" title="Качество автоматических решений" className="metrics-log-panel" action={<div className="metrics-controls"><label htmlFor="metrics-window-days">Окно</label><select id="metrics-window-days" value={windowDays} onChange={(event) => setWindowDays(Number(event.target.value))}><option value={7}>7 дней</option><option value={30}>30 дней</option><option value={90}>90 дней</option></select><button className="button button-primary" disabled={calculate.isPending} data-testid="button-calculate-verification-metrics" onClick={calculateNow}>{calculate.isPending ? 'Расчёт…' : 'Рассчитать и сохранить'}</button></div>}>
    <p className="metrics-explainer">FAR и FRR считают только записи с is_match и ground_truth_match; APCER и BPCER — с is_live и ground_truth_liveness. Записи без нужной пары меток исключаются. Исходные решения модели не изменяются.</p>
    {calculate.isError && <div className="form-alert form-alert-error"><AlertTriangle size={14} /> Не удалось рассчитать метрики. Проверьте соединение с API и повторите попытку.</div>}
    {history.isLoading ? <div className="metrics-loading"><LoadingRows rows={3} /></div> : history.isError ? <ErrorState onRetry={history.refetch} /> : latest ? <>
      <div className="metrics-latest-meta"><span>Последний расчёт: <b>{fmtDate(latest.created_at)}</b></span><span>Окно: <b>{fmtDate(latest.window_start)} — {fmtDate(latest.window_end)}</b></span><span>Биометрическая выборка: <b>{biometric?.labeled_cases ?? latest.total_samples}</b></span>{liveness && <span>Проверки живости: <b>{liveness.labeled_cases}</b></span>}</div>
      {!hasAnyLabeledCases && <div className="metrics-empty-note">Пока нет размеченных сессий с данными для расчёта. После ручной проверки запустите расчёт снова.</div>}
      <div className="metrics-rate-grid">{rateItems.map(([label, value, description]) => <div className="metric-card" key={label}><div className="metric-top"><span className="eyebrow">{label}</span><span className="metric-icon"><Gauge size={15} /></span></div><strong className="display metric-value">{value}</strong><div className="metric-detail">{description}</div></div>)}</div>
      {calculationDetails ? <section className="metrics-breakdown" aria-label="Детализация расчёта метрик">
        <div className="metrics-breakdown-heading"><div><span className="eyebrow">Числитель / знаменатель</span><h3>Детализация последнего расчёта</h3></div><span>CER = (FAR + FRR) / 2 · ACER = (APCER + BPCER) / 2</span></div>
        <div className="metrics-breakdown-grid">{breakdownItems.map((item) => <article className="metrics-breakdown-card" key={item.label}>
          <div className="metrics-breakdown-title"><b>{item.label}</b><span>{item.population}</span></div>
          <code>{item.formula}</code>
          <div className="metrics-breakdown-equation"><span>{item.denominator ? `${item.substitution} = ${metricPercent(item.value)}` : '— · нет размеченных случаев'}</span><small>{item.numerator} ошибок из {item.denominator} наблюдений</small></div>
        </article>)}</div>
        <div className="metrics-counts"><span><b>TP {biometric.tp}</b> — правильные допуски</span><span><b>FP {biometric.fp}</b> — ложные допуски</span><span><b>FN {biometric.fn}</b> — ложные отказы</span><span><b>TN {biometric.tn}</b> — правильные отказы</span><span><b>Атак отклонено {liveness.spoof_rejected}</b> из {liveness.spoof_cases}</span><span><b>Живых отклонено {liveness.bona_fide_rejected_as_spoof}</b> из {liveness.bona_fide_cases}</span></div>
      </section> : <div className="metrics-empty-note">Для этого сохранённого расчёта подробные счётчики не записывались. Они появятся у новых расчётов.</div>}
      <div className="table-wrap mobile-scroll metrics-table-wrap"><table className="data-table"><thead><tr><th>Расчёт</th><th>Окно, дни</th><th>Био. выборка</th><th>Живость</th><th>FAR</th><th>FRR</th><th>APCER</th><th>BPCER</th><th>CER</th><th>ACER</th></tr></thead><tbody>{history.data.map((item) => <tr key={item.id}><td>{fmtDate(item.created_at)}</td><td>{Math.round((new Date(item.window_end) - new Date(item.window_start)) / 86400000)}</td><td>{item.calculation_details?.biometric.labeled_cases ?? item.total_samples}</td><td>{item.calculation_details?.liveness.labeled_cases ?? '—'}</td><td>{historyRate(item, 'far', Boolean(item.calculation_details?.biometric.impostor_cases))}</td><td>{historyRate(item, 'frr', Boolean(item.calculation_details?.biometric.genuine_cases))}</td><td>{historyRate(item, 'apcer', Boolean(item.calculation_details?.liveness.spoof_cases))}</td><td>{historyRate(item, 'bpcer', Boolean(item.calculation_details?.liveness.bona_fide_cases))}</td><td>{historyRate(item, 'cer', Boolean(item.calculation_details?.biometric.genuine_cases && item.calculation_details?.biometric.impostor_cases))}</td><td>{historyRate(item, 'acer', Boolean(item.calculation_details?.liveness.spoof_cases && item.calculation_details?.liveness.bona_fide_cases))}</td></tr>)}</tbody></table></div>
    </> : <EmptyState icon={Gauge} title="Метрики ещё не рассчитывались" text="Выберите период и сохраните первый снимок метрик." />}
  </Panel>;
}

function GovernancePage() {
  const card = useGetModelCard();
  const versions = useListModelVersions();
  const model = card.data;
  const [managerOpen, setManagerOpen] = useState(false);
  return <PageFrame eyebrow="Управление / контроль модели" title="Управление моделью" intro="Назначение показателей, пороги, происхождение данных и известные ограничения модели." action={<button className="button button-secondary" aria-expanded={managerOpen} aria-controls="model-registry-disclosure" data-testid="button-toggle-model-registry" onClick={() => setManagerOpen((open) => !open)}><BookOpen size={15} />{managerOpen ? 'Скрыть карточку модели' : 'Карточка модели'}</button>}>
    <div id="model-registry-disclosure">{managerOpen && <ModelRegistryPanel onClose={() => setManagerOpen(false)} />}</div>
    <VerificationMetricsPanel />
    {card.isLoading ? <div className="governance-skeleton"><div /><div /><div /></div> : card.isError ? <ErrorState onRetry={card.refetch} /> : <><div className="governance-hero soft-card"><div className="hero-symbol"><ShieldCheck size={26} /></div><div className="governance-title"><div className="eyebrow">Активная модель / {labelStatus(model?.status || '') || '—'}</div><h2 className="display">{model?.model_name || 'HEALTH_ID'} <span>{model?.version}</span></h2><p>{model?.purpose}</p></div><div className="governance-stamp"><span className="stamp-line" /><span>ИССЛЕДОВАНИЕ<br />ПОД КОНТРОЛЕМ</span></div></div><div className="governance-grid">
        <Panel kicker="Известные ограничения" title="Ограничения" className="limitations-panel"><ul className="limitation-list">{(model?.limitations || []).map((item) => <li key={item}><span><AlertTriangle size={14} /></span>{item}</li>)}</ul><div className="full-disclaimer">{model?.disclaimer || 'Система предназначена только для исследований. Результаты не являются медицинским диагнозом и не должны быть единственным основанием для клинических, трудовых решений или решений о доступе.'}</div></Panel>
        <Panel kicker="Происхождение" title="История версий" className="versions-panel"><div className="version-list">{versions.isLoading ? <LoadingRows rows={3} /> : versions.isError ? <ErrorState onRetry={versions.refetch} /> : versions.data?.map((version) => <div className="version-row" key={version.id || version.version}><div className="version-marker"><span /></div><div><div className="version-head"><b className="mono">{version.version}</b><StatusPill tone={toneForStatus(version.status)}>{labelStatus(version.status)}</StatusPill></div><p>{version.change_summary}</p><small className="mono">{fmtDate(version.released_at, false)}</small></div></div>)}</div></Panel></div></>}
  </PageFrame>;
}

function NotFoundPage() {
  return <PageFrame eyebrow="404 / раздел не найден" title="Такой страницы нет." intro="В консоли не удалось найти данные для этого адреса." action={<Link href="/" className="button button-primary" data-testid="link-return-overview"><LayoutDashboard size={15} /> На главную</Link>}><div className="not-found-card soft-card signal-grid"><div className="not-found-code display">404</div><div><p>Данные для этого адреса не найдены.</p><Link href="/health-results" className="text-link" data-testid="link-open-results">Открыть реестр результатов <ChevronRight size={14} /></Link></div></div></PageFrame>;
}

function Router({ user, onLogout }) {
  return <AppShell user={user} onLogout={onLogout}><Switch><Route path="/" component={DashboardPage} /><Route path="/health-results" component={HealthResultsPage} /><Route path="/health-results/:id" component={HealthResultsPage} /><Route path="/verification" component={VerificationPage} /><Route path="/verification/:id" component={VerificationPage} /><Route path="/embeddings" component={EmbeddingsPage} /><Route path="/review" component={ReviewPage} /><Route path="/governance" component={GovernancePage} /><Route path="/drift" component={DriftPage} /><Route path="/manual" component={UserGuidePage} /><Route component={NotFoundPage} /></Switch></AppShell>;
}

function App() {
  const auth = useAuth();
  const [reviewMedia, setReviewMedia] = useState({});
  const retainReviewMedia = useCallback((sessionId, files) => {
    setReviewMedia((current) => ({ ...current, [sessionId]: files }));
  }, []);
  const releaseReviewMedia = useCallback((sessionId) => {
    setReviewMedia((current) => {
      const next = { ...current };
      delete next[sessionId];
      return next;
    });
  }, []);
  const contextValue = useMemo(() => ({ reviewMedia, retainReviewMedia, releaseReviewMedia }), [reviewMedia, retainReviewMedia, releaseReviewMedia]);
  if (auth.isLoading || !auth.isAuthenticated) {
    return <AuthGate login={auth.login} isLoading={auth.isLoading} error={auth.error} />;
  }
  return <ReviewMediaContext.Provider value={contextValue}><QueryClientProvider client={queryClient}><WouterRouter base={import.meta.env.BASE_URL.replace(/\/$/, '')}><Router user={auth.user} onLogout={auth.logout} /></WouterRouter></QueryClientProvider></ReviewMediaContext.Provider>;
}

export default App;