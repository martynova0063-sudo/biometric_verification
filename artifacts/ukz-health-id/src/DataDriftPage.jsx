import { useEffect, useMemo, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import {
  Activity, AlertTriangle, Camera, Check, ChevronDown, Clock3, Download,
  Moon, Printer, RefreshCw, ShieldAlert, ShieldCheck, Sun,
} from 'lucide-react';
import {
  getGetDataDriftSummaryQueryKey,
  getListDataDriftRunsQueryKey,
  useGetDataDriftSummary,
  useListDataDriftRuns,
  useRunDataDriftAnalysis,
  useUpdateDataDriftControl,
} from '@workspace/api-client-react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './data-drift.css';

const HISTORY_PARAMS = { limit: 30 };
const AUTO_REFRESH_OPTIONS = [
  { value: 0, label: 'Выкл.' },
  { value: 5 * 60 * 1000, label: '5 минут' },
  { value: 15 * 60 * 1000, label: '15 минут' },
  { value: 60 * 60 * 1000, label: '1 час' },
  { value: 24 * 60 * 60 * 1000, label: '24 часа' },
];

const STATUS_COPY = {
  collecting: { label: 'Сбор данных', className: 'drift-status-collecting' },
  insufficient_data: { label: 'Недостаточно данных', className: 'drift-status-insufficient' },
  stable: { label: 'Стабильно', className: 'drift-status-stable' },
  warning: { label: 'Предупреждение', className: 'drift-status-warning' },
  critical: { label: 'Критический дрейф', className: 'drift-status-critical' },
};

const RECALIBRATION_COPY = {
  not_run: 'Ожидает запуска',
  initial_collection: 'Начальный сбор данных — порог фиксирован',
  insufficient_labels: 'Недостаточно проверенных случаев',
  target_unachievable: 'Цель FAR не достигнута',
  model_unavailable: 'Активная модель не найдена',
  unchanged: 'Порог соответствует разметке',
  applied: 'Порог обновлён по разметке',
};

function formatDate(value, withTime = true) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '—';
  return new Intl.DateTimeFormat('ru-RU', withTime
    ? { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' }
    : { day: '2-digit', month: 'short', year: 'numeric' }).format(date);
}

function formatNumber(value, digits = 2) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  return new Intl.NumberFormat('ru-RU', { maximumFractionDigits: digits }).format(Number(value));
}

function formatPercent(value, digits = 2) {
  if (value == null || !Number.isFinite(Number(value))) return '—';
  return `${formatNumber(Number(value) * 100, digits)}%`;
}

function statusFor(value) {
  return STATUS_COPY[value] || STATUS_COPY.insufficient_data;
}

function downloadCsv(rows, filename) {
  if (!rows.length) return;
  const headers = Object.keys(rows[0]);
  const csvCell = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const content = [headers, ...rows.map((row) => headers.map((header) => row[header]))]
    .map((row) => row.map(csvCell).join(','))
    .join('\r\n');
  const blob = new Blob([`\uFEFF${content}`], { type: 'text/csv;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function DriftTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return <div className="drift-tooltip">
    <b>{label}</b>
    <span>PSI: {formatNumber(payload[0].value, 3)}</span>
  </div>;
}

function DriftPage() {
  const queryClient = useQueryClient();
  const [metricKey, setMetricKey] = useState('match_score');
  const [autoRefreshMs, setAutoRefreshMs] = useState(0);
  const [darkMode, setDarkMode] = useState(() => (
    typeof document !== 'undefined' && document.documentElement.classList.contains('dark')
  ));
  const [reviewerNote, setReviewerNote] = useState('');
  const [controlError, setControlError] = useState('');

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  const summaryQuery = useGetDataDriftSummary({
    query: {
      queryKey: getGetDataDriftSummaryQueryKey(),
      refetchInterval: autoRefreshMs || false,
      refetchOnWindowFocus: false,
    },
  });
  const historyQuery = useListDataDriftRuns(HISTORY_PARAMS, {
    query: {
      queryKey: getListDataDriftRunsQueryKey(HISTORY_PARAMS),
      refetchInterval: autoRefreshMs || false,
      refetchOnWindowFocus: false,
    },
  });
  const runAnalysis = useRunDataDriftAnalysis();
  const updateControl = useUpdateDataDriftControl();
  const summary = summaryQuery.data;
  const history = historyQuery.data || [];
  const selectedMetric = summary?.metrics?.find((metric) => metric.key === metricKey) || summary?.metrics?.[0];

  const chartData = useMemo(() => [...history]
    .sort((left, right) => left.calculation_date.localeCompare(right.calculation_date))
    .map((run) => ({
      date: run.calculation_date,
      psi: run.psi_values?.[metricKey],
    }))
    .filter((item) => typeof item.psi === 'number' && Number.isFinite(item.psi)), [history, metricKey]);

  const invalidateDriftQueries = () => {
    queryClient.invalidateQueries({ queryKey: getGetDataDriftSummaryQueryKey() });
    queryClient.invalidateQueries({ queryKey: getListDataDriftRunsQueryKey(HISTORY_PARAMS) });
  };

  const startAnalysis = () => {
    runAnalysis.mutate(undefined, { onSuccess: invalidateDriftQueries });
  };

  const changeManualMode = (enabled) => {
    const note = reviewerNote.trim();
    if (!note) {
      setControlError('Укажите причину изменения режима.');
      return;
    }
    setControlError('');
    updateControl.mutate(
      { data: { enabled, reviewer_note: note } },
      {
        onSuccess: () => {
          setReviewerNote('');
          invalidateDriftQueries();
        },
        onError: (error) => {
          setControlError(error?.message || 'Не удалось изменить режим ручной проверки.');
        },
      },
    );
  };

  const lastRefresh = summary?.last_run_at ? formatDate(summary.last_run_at) : 'Проверка ещё не запускалась';
  const progress = summary?.baseline_days_required
    ? Math.min(100, (summary.baseline_days_elapsed / summary.baseline_days_required) * 100)
    : 0;
  const pageStatus = statusFor(summary?.status || 'collecting');
  const currentMeanBrightness = summary?.current_mean_brightness;
  const baselineMeanBrightness = summary?.baseline_mean_brightness;
  const criticalModeLocked = summary?.status === 'critical';

  return <div className={`page-in drift-page ${darkMode ? 'drift-dark' : ''}`}>
    <div className="page-header drift-header">
      <div>
        <div className="eyebrow">Контроль / стабильность данных</div>
        <h1 className="display page-title">Дрейф данных</h1>
        <p className="page-intro">Отслеживание изменений биометрических и HEALTH_ID сигналов относительно проверенной базовой выборки.</p>
        <div className="drift-updated"><Clock3 size={13} /> Последний анализ: {lastRefresh}</div>
      </div>
      <div className="drift-actions print-hidden">
        <div className="drift-refresh-group" aria-label="Управление обновлением данных">
          <button
            className="drift-refresh-button"
            data-testid="button-run-drift-analysis"
            onClick={startAnalysis}
            disabled={runAnalysis.isPending}
          >
            <RefreshCw size={14} className={runAnalysis.isPending ? 'drift-spin' : ''} />
            {runAnalysis.isPending ? 'Анализ…' : 'Проверить сейчас'}
          </button>
          <span className="drift-refresh-divider" />
          <label className="drift-refresh-select-wrap">
            <span className="sr-only">Автообновление</span>
            <select
              data-testid="select-drift-auto-refresh"
              value={autoRefreshMs}
              onChange={(event) => setAutoRefreshMs(Number(event.target.value))}
              aria-label="Интервал автообновления"
            >
              {AUTO_REFRESH_OPTIONS.map((option) => <option key={option.value} value={option.value}>
                {option.value ? `Авто: ${option.label}` : 'Авто: выкл.'}
              </option>)}
            </select>
            <ChevronDown size={13} aria-hidden="true" />
          </label>
        </div>
        <button className="drift-icon-button" data-testid="button-print-drift" title="Печать или сохранить в PDF" onClick={() => window.print()}><Printer size={15} /></button>
        <button className="drift-icon-button" data-testid="button-toggle-drift-dark-mode" title={darkMode ? 'Светлая тема' : 'Тёмная тема'} onClick={() => setDarkMode((value) => !value)}>{darkMode ? <Sun size={15} /> : <Moon size={15} />}</button>
      </div>
    </div>

    {summaryQuery.isLoading ? <div className="drift-loading-grid" aria-label="Загрузка состояния дрейфа">
      {[1, 2, 3, 4].map((item) => <div className="drift-skeleton" key={item} />)}
      <div className="drift-skeleton drift-skeleton-wide" />
    </div> : summaryQuery.isError ? <div className="drift-error" role="alert">
      <AlertTriangle size={18} /><span>Не удалось загрузить мониторинг дрейфа.</span>
      <button className="button button-secondary" data-testid="button-retry-drift-summary" onClick={() => summaryQuery.refetch()}>Повторить</button>
    </div> : <>
      <section className={`drift-state-banner ${pageStatus.className}`} data-testid="status-drift-overall">
        <div className="drift-state-icon">{summary?.status === 'critical' ? <ShieldAlert size={20} /> : summary?.status === 'stable' ? <ShieldCheck size={20} /> : <Activity size={20} />}</div>
        <div className="drift-state-copy">
          <b>{pageStatus.label}</b>
          <span>{summary?.manual_review_mode
            ? `Все новые проверки направляются специалисту. ${summary.manual_review_reason || ''}`
            : summary?.baseline_status === 'collecting'
              ? !summary.collection_started_at
                ? 'Ожидаются первые новые наблюдения. После их поступления первые 14 дней будут использоваться фиксированные пороги и ручная проверка пограничных результатов.'
                : summary.baseline_days_elapsed < 14
                  ? 'Идёт начальный 14-дневный сбор: пороги остаются фиксированными, пограничные результаты направляются на ручную проверку.'
                  : 'Сбор продолжается до 30 дней и минимального объёма базовых наблюдений.'
              : summary?.needs_recalibration
                ? 'Обнаружен сигнал для повторной калибровки. Автоматические изменения возможны только при достаточной разметке.'
                : 'Автоматический режим не выявил превышения порогов по доступным метрикам.'}</span>
        </div>
        {summary?.needs_recalibration && <span className="drift-flag">Требуется калибровка</span>}
      </section>

      {summary?.alert_messages?.length > 0 && <section className="drift-alert-list" aria-label="Предупреждения">
        {summary.alert_messages.map((alert, index) => <div className="drift-alert" key={`${index}-${alert}`} data-testid={`alert-drift-${index}`}>
          <AlertTriangle size={14} /><span>{alert}</span>
        </div>)}
      </section>}

      <div className="drift-kpi-grid">
        <article className="drift-kpi soft-card" data-testid="card-drift-baseline">
          <div className="drift-kpi-top"><span>Базовая выборка</span><Activity size={16} /></div>
          <strong>{summary?.baseline_status === 'ready' ? 'Готова' : `${summary?.baseline_days_elapsed || 0} / ${summary?.baseline_days_required || 30} дн.`}</strong>
          <div className="drift-progress" aria-label={`Сбор базовой выборки: ${Math.round(progress)}%`}>
            <span style={{ width: `${progress}%` }} />
          </div>
          <small>{summary?.baseline_status === 'ready'
            ? `Создана ${formatDate(summary.baseline_created_at, false)}`
            : summary?.collection_started_at
              ? `Сбор с ${formatDate(summary.collection_started_at, false)}`
              : 'Сбор начнётся после первых новых наблюдений.'}</small>
        </article>
        <article className="drift-kpi soft-card" data-testid="card-drift-observations">
          <div className="drift-kpi-top"><span>Наблюдения</span><Camera size={16} /></div>
          <strong>{formatNumber(summary?.observations_total, 0)}</strong>
          <small>Учитываются только записи зарегистрированных нетестовых сотрудников.</small>
        </article>
        <article className="drift-kpi soft-card" data-testid="card-drift-match-threshold">
          <div className="drift-kpi-top"><span>Порог совпадения</span><ShieldCheck size={16} /></div>
          <strong>{formatNumber(summary?.current_match_threshold, 4)}</strong>
          <small>Целевой эмпирический FAR: {formatPercent(summary?.target_far, 3)}</small>
        </article>
        <article className="drift-kpi soft-card" data-testid="card-drift-recalibration">
          <div className="drift-kpi-top"><span>Калибровка порога</span><RefreshCw size={16} /></div>
          <strong className="drift-kpi-text">{RECALIBRATION_COPY[summary?.recalibration_status] || summary?.recalibration_status || 'Ожидает запуска'}</strong>
          <small>{summary?.labelled_impostor_samples || 0} impostor · {summary?.labelled_genuine_samples || 0} genuine случаев</small>
        </article>
      </div>

      <div className="drift-main-grid">
        <section className="panel soft-card drift-chart-panel">
          <div className="panel-head drift-panel-head">
            <div><div className="eyebrow">Скользящее окно / 7 дней</div><h2 className="display panel-title">История PSI</h2></div>
            <div className="drift-chart-tools print-hidden">
              <label className="drift-metric-select">
                <span className="sr-only">Метрика на графике</span>
                <select data-testid="select-drift-chart-metric" value={selectedMetric?.key || metricKey} onChange={(event) => setMetricKey(event.target.value)}>
                  {(summary?.metrics || []).map((metric) => <option key={metric.key} value={metric.key}>{metric.label}</option>)}
                </select>
              </label>
              <button
                className="drift-icon-button"
                data-testid="button-export-drift-chart"
                title="Скачать данные графика CSV"
                disabled={!chartData.length}
                onClick={() => downloadCsv(chartData, `psi-${metricKey}.csv`)}
              ><Download size={15} /></button>
            </div>
          </div>
          <div className="drift-chart-meta">
            <span>{selectedMetric?.label || 'Метрика'}</span>
            <span>Порог предупреждения 0,10</span>
            <span>Критический порог 0,25</span>
          </div>
          <div className="drift-chart">
            {historyQuery.isLoading ? <div className="drift-chart-skeleton" /> : chartData.length ? <ResponsiveContainer width="100%" height={280} debounce={0}>
              <LineChart data={chartData} margin={{ top: 12, right: 15, bottom: 4, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={darkMode ? 'rgba(255,255,255,.1)' : '#e3e0d8'} />
                <XAxis dataKey="date" tick={{ fontSize: 11, fill: darkMode ? '#c2cbc9' : '#64706e' }} stroke={darkMode ? '#667673' : '#b9c0ba'} tickFormatter={(value) => formatDate(value, false)} />
                <YAxis domain={[0, 'auto']} tick={{ fontSize: 11, fill: darkMode ? '#c2cbc9' : '#64706e' }} stroke={darkMode ? '#667673' : '#b9c0ba'} />
                <Tooltip content={<DriftTooltip />} isAnimationActive={false} cursor={{ stroke: darkMode ? '#9aa9a5' : '#687a75', strokeDasharray: '3 3' }} />
                <ReferenceLine y={0.1} stroke="#c98a31" strokeDasharray="5 4" />
                <ReferenceLine y={0.25} stroke="#c94d49" strokeDasharray="5 4" />
                <Line type="monotone" dataKey="psi" name="PSI" stroke="#168b7c" strokeWidth={2.5} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer> : <div className="drift-chart-empty">
              <Activity size={21} />
              <b>{summary?.baseline_status === 'ready' ? 'Пока нет расчётов PSI' : 'График появится после базовой выборки'}</b>
              <span>{summary?.baseline_status === 'ready'
                ? 'Для каждой точки нужно не менее 100 значений выбранной метрики в текущем окне.'
                : 'Сначала система соберёт 30 дней базовых наблюдений и минимальный объём данных.'}</span>
            </div>}
          </div>
          <div className="drift-chart-foot">Сравниваются гистограммы шириной 0,05; метрики с недостаточным числом наблюдений не включаются в оценку.</div>
        </section>

        <section className="panel soft-card drift-context-panel">
          <div className="panel-head drift-panel-head"><div><div className="eyebrow">Условия съёмки</div><h2 className="display panel-title">Камера и освещение</h2></div><Camera size={17} /></div>
          <div className="drift-context-stats">
            <div><span>Средняя яркость · база</span><b>{formatNumber(baselineMeanBrightness, 1)} <small>/ 255</small></b></div>
            <div><span>Средняя яркость · 7 дней</span><b>{formatNumber(currentMeanBrightness, 1)} <small>/ 255</small></b></div>
          </div>
          <div className="drift-camera-columns">
            <div>
              <h3>Базовые камеры</h3>
              {summary?.baseline_camera_models?.length ? summary.baseline_camera_models.map((item) => <div className="drift-camera-row" key={`base-${item.camera_model}`}><span title={item.camera_model}>{item.camera_model}</span><b>{item.samples}</b></div>) : <p className="drift-muted">Модели камеры появятся после новых видеосессий.</p>}
            </div>
            <div>
              <h3>Последние 7 дней</h3>
              {summary?.current_camera_models?.length ? summary.current_camera_models.map((item) => <div className="drift-camera-row" key={`current-${item.camera_model}`}><span title={item.camera_model}>{item.camera_model}</span><b>{item.samples}</b></div>) : <p className="drift-muted">Нет сведений о камерах в текущем окне.</p>}
            </div>
          </div>
          <div className="drift-chart-foot">Учитываются только новые проверки. Названия моделей камер поступают из браузера; уникальный ID устройства не сохраняется.</div>
        </section>
      </div>

      <section className="panel soft-card drift-metrics-panel">
        <div className="panel-head drift-panel-head">
          <div><div className="eyebrow">Порог PSI / 0,10 и 0,25</div><h2 className="display panel-title">Состояние метрик</h2></div>
          <span className="drift-minimum"><Check size={13} /> минимум {summary?.minimum_samples || 100} наблюдений на метрику</span>
        </div>
        <div className="drift-table-scroll">
          <table className="drift-table">
            <thead><tr><th>Метрика</th><th>PSI</th><th>База</th><th>7 дней</th><th>Состояние</th></tr></thead>
            <tbody>
              {(summary?.metrics || []).map((metric) => {
                const state = statusFor(metric.status);
                return <tr key={metric.key} data-testid={`row-drift-metric-${metric.key}`}>
                  <td><b>{metric.label}</b><small>{metric.unit}</small></td>
                  <td className="mono">{metric.psi == null ? '—' : formatNumber(metric.psi, 3)}</td>
                  <td className="mono">{formatNumber(metric.baseline_samples, 0)}</td>
                  <td className="mono">{formatNumber(metric.current_samples, 0)}</td>
                  <td><span className={`drift-pill ${state.className}`}>{state.label}</span></td>
                </tr>;
              })}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel soft-card drift-control-panel">
        <div className="panel-head drift-panel-head">
          <div><div className="eyebrow">Операторский контроль</div><h2 className="display panel-title">Ручной режим и порог</h2></div>
          <span className={`drift-pill ${summary?.manual_review_mode ? 'drift-status-critical' : 'drift-status-stable'}`}>{summary?.manual_review_mode ? 'Ручная проверка включена' : 'Автоматический маршрут доступен'}</span>
        </div>
        <div className="drift-control-content">
          <div className="drift-control-copy">
            <p>{summary?.manual_review_mode
              ? 'Новые сессии направляются на ручную проверку. Выход из режима требует комментария специалиста и невозможен при критическом PSI.'
              : 'При PSI ≥ 0,25 сервер автоматически включает ручную проверку. Порог совпадения меняется только по размеченным случаям.'}</p>
            <div className="drift-calibration-details">
              <span>Предложенный порог: <b>{formatNumber(summary?.proposed_match_threshold, 4)}</b></span>
              <span>FAR по разметке: <b>{formatPercent(summary?.achieved_far, 3)}</b></span>
              <span>FRR по разметке: <b>{formatPercent(summary?.achieved_frr, 2)}</b></span>
            </div>
          </div>
          <div className="drift-control-form">
            <label htmlFor="drift-reviewer-note">Комментарий оператора</label>
            <textarea
              id="drift-reviewer-note"
              data-testid="input-drift-reviewer-note"
              maxLength={500}
              rows={2}
              value={reviewerNote}
              onChange={(event) => setReviewerNote(event.target.value)}
              placeholder="Укажите причину изменения режима"
            />
            {controlError && <div className="drift-inline-error" role="alert">{controlError}</div>}
            <button
              className={summary?.manual_review_mode ? 'button button-secondary' : 'button button-primary'}
              data-testid="button-toggle-drift-manual-mode"
              disabled={updateControl.isPending || (summary?.manual_review_mode && criticalModeLocked)}
              onClick={() => changeManualMode(!summary?.manual_review_mode)}
            >
              {summary?.manual_review_mode ? 'Возобновить автоматический маршрут' : 'Включить ручную проверку'}
            </button>
            {summary?.manual_review_mode && criticalModeLocked && <small>Сначала PSI должен выйти из критической зоны.</small>}
          </div>
        </div>
      </section>

      <section className="panel soft-card drift-history-panel">
        <div className="panel-head drift-panel-head"><div><div className="eyebrow">Ежедневный мониторинг</div><h2 className="display panel-title">История проверок</h2></div><span className="drift-run-count">{history.length} записей</span></div>
        {historyQuery.isLoading ? <div className="drift-history-skeleton" /> : historyQuery.isError ? <div className="drift-table-empty" role="alert">Не удалось загрузить историю PSI. <button className="text-link" data-testid="button-retry-drift-history" onClick={() => historyQuery.refetch()}>Повторить</button></div> : history.length ? <div className="drift-table-scroll">
          <table className="drift-table drift-run-table">
            <thead><tr><th>Дата анализа</th><th>Статус</th><th>Метрик с PSI</th><th>Перекалибровка</th><th>Ручной режим</th></tr></thead>
            <tbody>{history.map((run) => {
              const state = statusFor(run.status);
              const values = Object.values(run.psi_values || {}).filter((value) => typeof value === 'number');
              return <tr key={run.id} data-testid={`row-drift-run-${run.id}`}>
                <td><b>{formatDate(run.calculation_date, false)}</b><small>Окно: {formatDate(run.window_start, false)} — {formatDate(run.window_end, false)}</small></td>
                <td><span className={`drift-pill ${state.className}`}>{state.label}</span></td>
                <td className="mono">{values.length}</td>
                <td>{RECALIBRATION_COPY[run.recalibration?.status] || run.recalibration?.status || '—'}</td>
                <td>{run.manual_review_mode ? 'Включён' : 'Выключен'}</td>
              </tr>;
            })}</tbody>
          </table>
        </div> : <div className="drift-table-empty"><Clock3 size={18} /><span>История появится после первого ежедневного или ручного запуска анализа.</span></div>}
      </section>
    </>}
  </div>;
}

export default DriftPage;
