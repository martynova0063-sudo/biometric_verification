import { useMemo, useState } from 'react';
import { AlertCircle, ArrowDownRight, ArrowUpRight, Clock3, Database, Fingerprint, RefreshCw } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { Link } from 'wouter';
import {
  getListVerificationSessionsQueryKey,
  useCreateEmbeddingInterpretation,
  useListVerificationSessions,
} from '@workspace/api-client-react';
import './embeddings-index.css';

const integerFormat = new Intl.NumberFormat('ru-RU');

function formatValue(value, digits = 4) {
  return value === null || value === undefined || !Number.isFinite(Number(value))
    ? '—'
    : new Intl.NumberFormat('ru-RU', { maximumFractionDigits: digits }).format(Number(value));
}

function formatDate(value) {
  if (!value) return 'Время не указано';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat('ru-RU', {
    day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  }).format(date);
}

const statusCopy = {
  insufficient_reference: {
    title: 'Недостаточно эталонных данных',
    body: 'Для этой сессии не набрано достаточно эталонных векторов. Сравнение ограничено и не может трактоваться как совпадение или несовпадение.',
  },
  no_reference: {
    title: 'Эталонные данные отсутствуют',
    body: 'В сессии нет эталонных векторов. Без них сравнение распределений не позволяет оценить сходство.',
  },
  unavailable: {
    title: 'Сравнение недоступно',
    body: 'Метрики сравнения для этой сессии не рассчитаны. Это не является результатом проверки личности.',
  },
  no_video_vectors: {
    title: 'Видеовекторы не получены',
    body: 'Для видеопотока не сформированы векторы. Данных недостаточно для сравнения с эталоном.',
  },
};

const interpretationVerdicts = {
  consistent: { title: 'Распределения в целом согласуются', tone: 'consistent' },
  borderline: { title: 'Пограничный технический результат', tone: 'borderline' },
  discordant: { title: 'Обнаружено выраженное расхождение', tone: 'discordant' },
  insufficient_data: { title: 'Недостаточно данных', tone: 'insufficient' },
};

function DistributionChart({ comparison }) {
  const { bin_centers: centers = [], reference_counts: reference = [], video_counts: video = [] } = comparison?.distribution || {};
  const length = Math.min(centers.length, reference.length, video.length);
  const chart = useMemo(() => {
    if (!length) return null;
    const points = Array.from({ length }, (_, index) => ({
      center: Number(centers[index]),
      reference: Math.max(0, Number(reference[index]) || 0),
      video: Math.max(0, Number(video[index]) || 0),
    })).filter((point) => Number.isFinite(point.center));
    if (!points.length) return null;
    const maxCount = Math.max(1, ...points.flatMap((point) => [point.reference, point.video]));
    const width = 760;
    const height = 260;
    const pad = { top: 18, right: 18, bottom: 40, left: 42 };
    const innerW = width - pad.left - pad.right;
    const innerH = height - pad.top - pad.bottom;
    const x = (index) => pad.left + (points.length === 1 ? innerW / 2 : (index / (points.length - 1)) * innerW);
    const y = (count) => pad.top + innerH - (count / maxCount) * innerH;
    const pathFor = (key) => points.map((point, index) => `${index ? 'L' : 'M'} ${x(index)} ${y(point[key])}`).join(' ');
    return { points, width, height, pad, innerW, innerH, maxCount, x, y, referencePath: pathFor('reference'), videoPath: pathFor('video') };
  }, [centers, reference, video, length]);

  if (!chart) {
    return <div className="emb-chart-empty">Для этой сессии нет агрегированных данных распределения.</div>;
  }

  const gridValues = [0, Math.round(chart.maxCount / 2), chart.maxCount];
  return (
    <div className="emb-chart-wrap" data-testid="visualization-distribution">
      <svg className="emb-chart" viewBox={`0 0 ${chart.width} ${chart.height}`} role="img" aria-label="Наложенные распределения эталонных и видеовекторов">
        {gridValues.map((tick, index) => {
          const y = chart.y(tick);
          return (
            <g key={`${tick}-${index}`}>
              <line x1={chart.pad.left} x2={chart.width - chart.pad.right} y1={y} y2={y} className="emb-chart-grid" />
              <text x={chart.pad.left - 10} y={y + 4} textAnchor="end" className="emb-chart-tick">{integerFormat.format(tick)}</text>
            </g>
          );
        })}
        {chart.points.map((point, index) => {
          const barW = Math.max(3, Math.min(18, chart.innerW / Math.max(1, chart.points.length) * 0.56));
          const x = chart.x(index);
          const refY = chart.y(point.reference);
          const vidY = chart.y(point.video);
          return (
            <g key={`${point.center}-${index}`}>
              <rect x={x - barW * 0.55} y={refY} width={barW} height={chart.pad.top + chart.innerH - refY} rx="2" className="emb-bar-reference" />
              <rect x={x - barW * 0.45} y={vidY} width={barW} height={chart.pad.top + chart.innerH - vidY} rx="2" className="emb-bar-video" />
            </g>
          );
        })}
        <path d={chart.referencePath} className="emb-chart-line emb-chart-line-reference" />
        <path d={chart.videoPath} className="emb-chart-line emb-chart-line-video" />
        <line x1={chart.pad.left} x2={chart.width - chart.pad.right} y1={chart.pad.top + chart.innerH} y2={chart.pad.top + chart.innerH} className="emb-chart-axis" />
        <text x={chart.pad.left} y={chart.height - 10} className="emb-chart-tick">{formatValue(chart.points[0].center, 3)}</text>
        <text x={chart.width - chart.pad.right} y={chart.height - 10} textAnchor="end" className="emb-chart-tick">{formatValue(chart.points[chart.points.length - 1].center, 3)}</text>
      </svg>
    </div>
  );
}

function Metric({ label, value, note, accent }) {
  return (
    <div className={`emb-metric ${accent || ''}`}>
      <span className="emb-metric-label">{label}</span>
      <strong className="emb-metric-value">{value}</strong>
      <span className="emb-metric-note">{note}</span>
    </div>
  );
}

export default function EmbeddingsIndexPage({ arcfacePanel }) {
  const queryClient = useQueryClient();
  const { data: sessions, isLoading, isError, refetch, isFetching } = useListVerificationSessions({ limit: 100 });
  const interpretationMutation = useCreateEmbeddingInterpretation({
    mutation: {
      onSuccess: (interpretation, variables) => {
        queryClient.setQueryData(
          getListVerificationSessionsQueryKey({ limit: 100 }),
          (current) => (Array.isArray(current)
            ? current.map((session) => (
              session.id === variables.sessionId
                ? { ...session, embedding_interpretation: interpretation }
                : session
            ))
            : current),
        );
      },
    },
  });
  const availableSessions = useMemo(
    () => (Array.isArray(sessions) ? sessions : [])
      .filter((session) => session?.embedding_comparison)
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [sessions],
  );
  const [selectedId, setSelectedId] = useState('');
  const activeId = availableSessions.some((session) => session.id === selectedId)
    ? selectedId
    : availableSessions[0]?.id || '';
  const selectedSession = availableSessions.find((session) => session.id === activeId);
  const comparison = selectedSession?.embedding_comparison;
  const stateCopy = comparison?.status && comparison.status !== 'available' ? statusCopy[comparison.status] : null;
  const delta = comparison?.mean?.delta;
  const ratio = comparison?.variance?.ratio;
  const outliers = comparison?.outliers;
  const interpretation = selectedSession?.embedding_interpretation;
  const interpretationVerdict = interpretationVerdicts[interpretation?.verdict];
  const canInterpret = Boolean(activeId && comparison?.status === 'available');

  return (
    <div className="emb-page">
      <header className="emb-page-header">
        <div className="emb-heading-mark"><Fingerprint size={20} strokeWidth={1.8} /></div>
        <div className="emb-heading-copy">
          <div className="emb-eyebrow">HEALTH_ID <span> / </span> ИССЛЕДОВАТЕЛЬСКАЯ КОНСОЛЬ</div>
          <h1>Эмбеддинги и индексация</h1>
          <p>Улучшение качества распознавания</p>
        </div>
        <div className="emb-header-note"><span className="emb-live-dot" /> Исследовательский режим</div>
      </header>

      <section className="emb-arcface-section" aria-label="Поиск ArcFace">
        {arcfacePanel || (
          <div className="emb-unavailable" role="status" data-testid="status-arcface-unavailable">
            <div className="emb-unavailable-icon"><Database size={17} /></div>
            <div><strong>Панель ArcFace недоступна</strong><span>Поиск кандидатов сейчас не подключён к этой странице.</span></div>
          </div>
        )}
      </section>

      <section className="emb-comparison" aria-labelledby="emb-comparison-title">
        <div className="emb-section-heading">
          <div>
            <div className="emb-eyebrow">АНАЛИЗ СЕССИЙ <span> / </span> АГРЕГИРОВАННЫЕ ДАННЫЕ</div>
            <h2 id="emb-comparison-title">Сравнение эмбеддингов</h2>
            <p>Распределения и сводные измерения для эталонных и видеовекторов.</p>
          </div>
          <button className="emb-refresh" type="button" onClick={() => refetch()} disabled={isFetching} data-testid="button-refresh-sessions" aria-label="Обновить список сессий">
            <RefreshCw size={15} className={isFetching ? 'emb-rotating' : ''} />
            <span>{isFetching ? 'Обновление' : 'Обновить'}</span>
          </button>
        </div>

        {isLoading ? (
          <div className="emb-loading" aria-label="Загрузка сессий">
            <div className="emb-skeleton emb-skeleton-select" />
            <div className="emb-skeleton emb-skeleton-chart" />
            <div className="emb-skeleton-row"><i /><i /><i /></div>
          </div>
        ) : isError ? (
          <div className="emb-state-card emb-state-error" role="alert" data-testid="status-session-error">
            <AlertCircle size={19} />
            <div><strong>Не удалось загрузить сессии</strong><p>Проверьте соединение и повторите запрос.</p></div>
            <button type="button" onClick={() => refetch()} className="emb-retry" data-testid="button-retry-sessions">Повторить</button>
          </div>
        ) : availableSessions.length === 0 ? (
          <div className="emb-state-card emb-state-empty" data-testid="status-no-comparisons">
            <div className="emb-empty-symbol"><Database size={20} /></div>
            <div><strong>Сессий для сравнения пока нет</strong><p>Сравнение рассчитывается при обработке новых видео. Старые записи нельзя пересчитать, потому что видео не сохраняются.</p></div>
            <Link href="/verification" className="emb-retry" data-testid="link-create-verification">Создать видеосессию</Link>
          </div>
        ) : (
          <>
            <div className="emb-session-toolbar">
              <label htmlFor="emb-session-select">Сессия для анализа</label>
              <select
                id="emb-session-select"
                value={activeId}
                onChange={(event) => {
                  interpretationMutation.reset();
                  setSelectedId(event.target.value);
                }}
                data-testid="select-embedding-session"
              >
                {availableSessions.map((session) => (
                  <option key={session.id} value={session.id}>
                    {session.employee_ref || 'Сотрудник не указан'} · {formatDate(session.created_at)}
                  </option>
                ))}
              </select>
              <span className="emb-session-count">{availableSessions.length} {availableSessions.length === 1 ? 'сессия' : 'сессий'} с данными</span>
            </div>

            {comparison && (
              <>
                <div className={`emb-status-banner ${stateCopy ? 'emb-status-limited' : 'emb-status-ready'}`} role="status" data-testid="status-comparison">
                  <span className="emb-status-indicator" />
                  <div>
                    <strong>{stateCopy?.title || 'Агрегированные измерения доступны'}</strong>
                    <p>{stateCopy?.body || comparison.message || `Метод: ${comparison.method || 'не указан'}. Сопоставляются агрегаты в ${formatValue(comparison.dimensions, 0)} измерениях; это исследовательские данные, а не самостоятельное решение о личности.`}</p>
                  </div>
                  <div className="emb-vector-counts">
                    <span><b>{integerFormat.format(comparison.reference_vectors ?? 0)}</b> эталонных</span>
                    <span><b>{integerFormat.format(comparison.video_vectors ?? 0)}</b> видео</span>
                  </div>
                </div>

                <div className="emb-analysis-grid">
                  <section className="emb-chart-panel">
                    <div className="emb-panel-heading">
                      <div><span className="emb-panel-index">01</span><h3>Распределение векторов</h3></div>
                      <span className="emb-panel-unit">агрегированные корзины</span>
                    </div>
                    <div className="emb-chart-legend">
                      <span><i className="emb-legend-reference" /> Эталон</span>
                      <span><i className="emb-legend-video" /> Видео</span>
                      <small>Частота значений компонентов</small>
                    </div>
                    <DistributionChart comparison={comparison} />
                    <p className="emb-chart-caption">Визуализация построена по значениям компонентов эмбеддингов; показаны только агрегированные корзины.</p>
                  </section>

                  <section className="emb-metrics-panel">
                    <div className="emb-panel-heading">
                      <div><span className="emb-panel-index">02</span><h3>Сводные измерения</h3></div>
                      <span className="emb-panel-unit">без порогового вывода</span>
                    </div>
                    <div className="emb-metrics-grid">
                      <Metric
                        label="Сдвиг среднего"
                        value={formatValue(delta)}
                        note={delta === null || delta === undefined ? 'значение не рассчитано' : `видео − эталон · среднее видео ${formatValue(comparison.mean?.video)}`}
                        accent={delta > 0 ? 'emb-metric-warm' : delta < 0 ? 'emb-metric-cool' : ''}
                      />
                      <Metric
                        label="Изменение дисперсии"
                        value={ratio === null || ratio === undefined ? '—' : `${formatValue(ratio)}×`}
                        note={ratio === null || ratio === undefined ? 'отношение не рассчитано' : `видео / эталон · видео ${formatValue(comparison.variance?.video)}`}
                      />
                      <Metric
                        label="Выбросы"
                        value={outliers?.count === null || outliers?.count === undefined ? '—' : integerFormat.format(outliers.count)}
                        note={outliers?.total !== undefined ? `из ${integerFormat.format(outliers.total)} видео-векторов${outliers.threshold != null ? ` · порог ${formatValue(outliers.threshold)}` : ''}` : 'счётчик не рассчитан'}
                      />
                      <Metric
                        label="Сходство центроидов"
                        value={formatValue(comparison.mean?.centroid_cosine_similarity)}
                        note="косинусная мера · справочно"
                      />
                    </div>
                    <div className="emb-measure-foot">
                      {delta !== null && delta !== undefined && (delta < 0
                        ? <ArrowDownRight size={14} aria-hidden="true" />
                        : <ArrowUpRight size={14} aria-hidden="true" />)}
                      <span>Среднее эталона: {formatValue(comparison.mean?.reference)} <b>·</b> Размерность: {formatValue(comparison.dimensions, 0)}</span>
                    </div>
                  </section>
                </div>
                <section className="emb-interpretation-panel" aria-labelledby="emb-interpretation-title" data-testid="panel-embedding-interpretation">
                  <div className="emb-panel-heading emb-interpretation-heading">
                    <div><span className="emb-panel-index">03</span><h3 id="emb-interpretation-title">Расшифровка и вывод LLM</h3></div>
                    <button
                      className="emb-refresh emb-interpretation-button"
                      type="button"
                      onClick={() => interpretationMutation.mutate({
                        sessionId: activeId,
                        data: { refresh: Boolean(interpretation) },
                      })}
                      disabled={!canInterpret || interpretationMutation.isPending}
                      data-testid="button-generate-embedding-interpretation"
                    >
                      <RefreshCw size={14} className={interpretationMutation.isPending ? 'emb-rotating' : ''} />
                      <span>
                        {interpretationMutation.isPending
                          ? 'Формируем вывод'
                          : interpretation ? 'Обновить вывод' : 'Сформировать вывод'}
                      </span>
                    </button>
                  </div>

                  {interpretationMutation.isError && (
                    <div className="emb-interpretation-error" role="alert" data-testid="status-embedding-interpretation-error">
                      Не удалось получить вывод от OpenAI. Проверьте OPENAI_API_KEY, доступность API и попробуйте ещё раз.
                    </div>
                  )}

                  {interpretation ? (
                    <>
                      <div className={`emb-interpretation-verdict emb-interpretation-${interpretationVerdict?.tone || 'insufficient'}`}>
                        <span className="emb-interpretation-kicker">Технический вердикт</span>
                        <strong>{interpretationVerdict?.title || 'Интерпретация готова'}</strong>
                        <p>{interpretation.summary}</p>
                      </div>
                      <div className="emb-interpretation-columns">
                        <div>
                          <h4>Что показывают метрики</h4>
                          <ul>{interpretation.findings.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>
                        </div>
                        <div>
                          <h4>Что проверить</h4>
                          <ul>{interpretation.recommendations.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>
                        </div>
                      </div>
                      <p className="emb-interpretation-disclaimer">
                        {interpretation.disclaimer}
                      </p>
                      <div className="emb-interpretation-meta">
                        Модель: {interpretation.model} · сформировано {formatDate(interpretation.generated_at)}
                      </div>
                    </>
                  ) : (
                    <div className="emb-interpretation-empty">
                      <p>
                        LLM сопоставит сдвиг среднего, дисперсию, косинусное сходство, выбросы и
                        распределения. Исходные векторы, видео и данные сотрудника не передаются —
                        используются только агрегированные метрики. Запрос выполняется по кнопке;
                        повторная генерация отправит новый платный запрос в OpenAI.
                      </p>
                      {!canInterpret && (
                        <small>Для вывода нужны рассчитанные агрегированные метрики сравнения.</small>
                      )}
                    </div>
                  )}
                </section>
                <div className="emb-method-note"><Clock3 size={13} /> Сессия {selectedSession?.id} <span>·</span> {formatDate(selectedSession?.created_at)} <span>·</span> Метод: {comparison.method || 'не указан'}</div>
              </>
            )}
          </>
        )}
      </section>
      <footer className="emb-page-footer"><span>HEALTH_ID</span><span>Эмбеддинги — исследовательские измерения, не идентификационное решение</span></footer>
    </div>
  );
}
