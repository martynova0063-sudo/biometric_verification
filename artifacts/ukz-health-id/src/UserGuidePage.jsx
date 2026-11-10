import {
  BookOpen,
  Camera,
  ChevronRight,
  Code2,
  Database,
  ExternalLink,
  Fingerprint,
  HeartPulse,
  ShieldCheck,
} from 'lucide-react';
import './user-guide.css';

const guideSections = [
  ['start', 'Начало работы'],
  ['verification', 'Верификация'],
  ['review', 'Очередь проверки'],
  ['health-id', 'HEALTH_ID'],
  ['monitoring', 'Управление и дрейф'],
  ['api', 'API и контракты'],
];

function GuideSection({ id, number, title, children }) {
  return (
    <section className="guide-section panel soft-card" id={id}>
      <div className="panel-head">
        <div>
          <div className="eyebrow">Раздел {number}</div>
          <h2 className="display panel-title">{title}</h2>
        </div>
      </div>
      <div className="guide-section-body">{children}</div>
    </section>
  );
}

function ApiOperation({ method, path, description }) {
  return (
    <tr>
      <td><span className={`guide-method guide-method-${method.toLowerCase()}`}>{method}</span></td>
      <td><code>{path}</code></td>
      <td>{description}</td>
    </tr>
  );
}

export default function UserGuidePage() {
  return (
    <div className="page-in user-guide">
      <header className="page-header">
        <div>
          <div className="eyebrow">Справка / ИС УКЗ</div>
          <h1 className="display page-title">Руководство пользователя</h1>
          <p className="page-intro">
            Порядок работы с верификацией, очередью решений, HEALTH_ID и API.
            Система предназначена для исследований, а не для постановки диагноза
            или единоличного принятия кадровых решений.
          </p>
        </div>
        <div className="page-action guide-header-actions">
          <a className="button button-secondary" href="/api/docs" target="_blank" rel="noreferrer">
            <Code2 size={15} /> Открыть API
            <ExternalLink size={13} />
          </a>
          <a className="guide-spec-link" href="/api/openapi.json" target="_blank" rel="noreferrer">
            OpenAPI JSON <ExternalLink size={12} />
          </a>
        </div>
      </header>

      <div className="guide-layout">
        <aside className="guide-index panel soft-card">
          <div className="panel-head">
            <div>
              <div className="eyebrow">Навигация</div>
              <h2 className="display panel-title">Содержание</h2>
            </div>
            <BookOpen size={17} />
          </div>
          <nav aria-label="Содержание руководства">
            {guideSections.map(([id, title], index) => (
              <a href={`#${id}`} key={id}>
                <span>{String(index + 1).padStart(2, '0')}</span>
                {title}
                <ChevronRight size={13} />
              </a>
            ))}
          </nav>
        </aside>

        <div className="guide-content">
          <GuideSection id="start" number="01" title="Начало работы">
            <p>
              Войдите в систему под своей учётной записью. Основные разделы находятся
              в левом меню; на небольшом экране меню открывается кнопкой в верхней
              панели. Для завершения работы используйте кнопку выхода рядом с именем
              пользователя.
            </p>
            <div className="guide-callout">
              <ShieldCheck size={17} />
              <span>
                Результаты носят исследовательский характер. Они не являются
                медицинским диагнозом и не должны быть единственным основанием
                для решения о допуске к работе или доступе.
              </span>
            </div>
            <div className="guide-feature-grid">
              <div><Fingerprint size={17} /><b>Верификация</b><span>Сессии, фотографии и результаты проверки личности.</span></div>
              <div><Database size={17} /><b>Эмбеддинги и индексация</b><span>ArcFace-поиск кандидатов и распределения векторов.</span></div>
              <div><HeartPulse size={17} /><b>HEALTH_ID</b><span>Исследовательский расчёт и его история.</span></div>
              <div><Camera size={17} /><b>Очередь проверки</b><span>Случаи, ожидающие решения специалиста.</span></div>
            </div>
          </GuideSection>

          <GuideSection id="verification" number="02" title="Верификация сотрудника">
            <ol className="guide-steps">
              <li><b>Откройте «Верификация» и создайте новую сессию.</b> Выберите сотрудника из списка. Эталонное фото загрузится автоматически.</li>
              <li><b>Проверьте эталонное фото.</b> Если его нет, подключите камеру и сохраните снимок. Если фото уже есть, кнопка «Заменить фото» позволяет сделать новый снимок. Старый снимок сохраняется в закрытом архиве dataset.</li>
              <li><b>Добавьте видео.</b> Загрузите файл или запишите его с камеры. Запись с камеры сохраняется в MP4 с размером кадра 640 × 480; на форме указаны поддерживаемые форматы и ограничение размера.</li>
              <li><b>Запустите проверку и изучите результат.</b> Качество захвата отображается в реестре сессий, а в трассе — совпадение, живость и время серверной обработки.</li>
            </ol>
            <div className="guide-callout guide-callout-muted">
              Фото и видео, приложенные к сессии верификации, не сохраняются после обработки. Эталонное фото сотрудника хранится отдельно.
            </div>
            <div className="guide-status-list">
              <div><code>verified</code><span>Система подтвердила совпадение.</span></div>
              <div><code>manual_review</code><span>Результат ожидает ручной проверки.</span></div>
              <div><code>not_verified</code><span>Система не подтвердила совпадение.</span></div>
              <div><code>insufficient_quality</code><span>Видео читается, но данных недостаточно для надёжной оценки.</span></div>
              <div><code>not_evaluable</code><span>Техническая ошибка помешала оценке, например файл повреждён или компонент обработки недоступен.</span></div>
            </div>
          </GuideSection>

          <GuideSection id="review" number="03" title="Очередь проверки и Ground Truth">
            <p>
              В очередь попадают задачи со статусом <code>open</code> или
              <code> in_review</code>. Откройте задачу, изучите системную сводку
              и доступный контекст, затем укажите независимые оценки специалиста:
              совпадает ли человек и является ли он живым.
            </p>
            <ul className="guide-bullets">
              <li><b>Подтвердить</b> или <b>отклонить</b> можно после заполнения обеих оценок Ground Truth.</li>
              <li><b>Запросить повторную проверку</b> можно без заполнения этих оценок.</li>
              <li>Комментарий специалиста необязателен и сохраняется вместе с решением.</li>
              <li>Ground Truth хранится отдельно: действия специалиста не меняют системные статус, решение, совпадение и живость.</li>
              <li>Метки Ground Truth используются в расчётах показателей качества и живости.</li>
            </ul>
          </GuideSection>

          <GuideSection id="health-id" number="04" title="HEALTH_ID и результаты">
            <p>
              В форме расчёта выберите сотрудника из списка допущенных, затем
              укажите дату события и измерения. В список попадают сотрудники,
              у которых последняя верификация успешно пройдена системой или
              вручную подтверждена специалистом. Более поздний отказ или запрос
              повторной проверки блокирует расчёт. После расчёта запись появляется в реестре;
              откройте её, чтобы посмотреть значение, категорию и объясняющие
              факторы.
            </p>
            <p>
              Реестр можно просматривать и фильтровать. Идентификатор результата
              пригодится для запросов к API и ссылки на детальную запись.
              Интерпретируйте показатели с учётом ограничений модели в разделе
              «Управление моделью».
            </p>
          </GuideSection>

          <GuideSection id="monitoring" number="05" title="Управление моделью и дрейф данных">
            <p>
              «Управление моделью» показывает активную версию, назначение,
              ограничения, историю версий и сохранённые биометрические метрики.
              При расчёте метрик выберите период наблюдений.
            </p>
            <p>
              В разделе «Дрейф данных» смотрите состояние мониторинга,
              ежедневную историю и сигналы изменения входных данных. Переключение
              режима ручной проверки влияет на обработку новых сессий; основание
              изменения фиксируйте в комментарии.
            </p>
          </GuideSection>

          <GuideSection id="api" number="06" title="API: доступ и контракты">
            <p>
              API использует базовый путь <code>/api</code>. Маршруты требуют
              действующую авторизованную сессию. Откройте встроенную документацию,
              чтобы увидеть полный OpenAPI-контракт, типы запросов, ответы и коды
              ошибок.
            </p>
            <div className="guide-link-row">
              <a className="button button-secondary" href="/api/docs" target="_blank" rel="noreferrer">
                <Code2 size={15} /> Swagger UI <ExternalLink size={12} />
              </a>
              <a className="text-link" href="/api/openapi.json" target="_blank" rel="noreferrer">
                Скачать OpenAPI JSON <ChevronRight size={14} />
              </a>
            </div>

            <h3 className="guide-subheading">Сотрудники для расчёта HEALTH_ID</h3>
            <div className="guide-table-wrap">
              <table className="guide-api-table">
                <thead><tr><th>Метод</th><th>Путь</th><th>Назначение</th></tr></thead>
                <tbody>
                  <ApiOperation method="GET" path="/api/health-results/eligible-employees" description="Список сотрудников, чья последняя верификация успешна или вручную подтверждена." />
                  <ApiOperation method="POST" path="/api/health-results" description="Расчёт HEALTH_ID; возвращает 409, если у сотрудника нет подходящей последней верификации." />
                  <ApiOperation method="GET" path="/api/health-results/employees/{employee_ref}/personal-baseline" description="Коридоры HEALTH_ID и измерений сотрудника за последние 30 дней." />
                </tbody>
              </table>
            </div>
            <pre className="guide-code"><code>{`GET /api/health-results/eligible-employees

[
  {
    "employee_ref": "EMP-2048",
    "full_name": "Иванов Иван Иванович",
    "organization": "Организация",
    "is_test": false,
    "verification_session_id": "d3c0ccf7-fdb5-4fd2-8615-72ec68a54472",
    "verification_status": "manual_review",
    "eligibility_source": "manual",
    "verified_at": "2026-10-25T11:00:00"
  }
]`}</code></pre>
            <p>
              Персональные коридоры рассчитываются как медиана ± 1,5 IQR. Для
              каждого показателя нужно минимум четыре наблюдения; исходные
              измерения сохраняются для новых расчётов HEALTH_ID.
            </p>
            <p>
              <code>eligibility_source</code> равен <code>system</code> для
              успешной автоматической проверки и <code>manual</code> для
              подтверждённой задачи ручной проверки. Для расчёта также подходит
              системный статус <code>verified</code>; после ручного решения
              «Подтвердить» метки Ground Truth не ограничивают допуск.
            </p>

            <h3 className="guide-subheading">Поиск кандидатов ArcFace по фото</h3>
            <p>
              Откройте вкладку «Эмбеддинги и индексация». Поиск возвращает до пяти разных UUID сотрудников и cosine similarity.
              По умолчанию порог кандидата — 0,7; его можно изменить параметром
              запроса и калибровать на проверочной выборке. Фото обрабатывается в
              памяти и не сохраняется. Результат не подтверждает личность и не
              должен быть единственным основанием для допуска.
            </p>
            <p>
              Там же доступны сводные распределения ArcFace-векторов из кадров
              видео последней сессии и эталонных векторов индекса: сдвиг среднего,
              отношение дисперсий и оценка выбросов. Сырые векторы видео не
              сохраняются вместе с сессией; при малом числе эталонов статистика ограничена.
            </p>
            <div className="guide-table-wrap">
              <table className="guide-api-table">
                <thead><tr><th>Метод</th><th>Путь</th><th>Назначение</th></tr></thead>
                <tbody>
                  <ApiOperation method="GET" path="/api/arcface/status" description="Проверить доступность модели, индекса и число векторов." />
                  <ApiOperation method="POST" path="/api/recognize?top_k=5&threshold=0.7" description="Загрузить одно фото и получить список кандидатов." />
                  <ApiOperation method="POST" path="/api/employees/{employee_guid}/arcface-index" description="Рассчитать эмбеддинг по сохранённому эталонному фото сотрудника и добавить его в индекс." />
                  <ApiOperation method="POST" path="/api/refresh-index" description="Перезагрузить основной индекс и добавленные через интерфейс эмбеддинги без перезапуска API." />
                </tbody>
              </table>
            </div>
            <pre className="guide-code"><code>{`POST /api/recognize?top_k=5&threshold=0.7
Content-Type: multipart/form-data
file: <одно изображение>

{
  "found": true,
  "threshold": 0.7,
  "top_match": {
    "uuid": "UUID сотрудника",
    "cosine_similarity": 0.91
  },
  "results": []
}

POST /api/refresh-index`}</code></pre>
            <p>
              Чтобы добавить сохранённое фото без ручного копирования файлов,
              выберите сотрудника с эталонным фото и нажмите «Добавить в индекс».
              Эмбеддинг сохраняется отдельно от основного обучающего набора и
              остаётся доступен после обновления или полной перестройки индекса.
              Повторное добавление того же фото не создаёт дубликат.
            </p>
            <p>
              При создании сессии можно отдельно отметить «Пополнить индекс
              ResNet50 + ArcFace». В индекс попадут признаки выбранных кадров
              только после успешной верификации; повторная загрузка тех же
              признаков не создаёт дубликатов. Это пополняет FAISS-индекс поиска,
              но не переобучает веса модели.
            </p>
            <p>
              После добавления папок сотрудников и фото в
              <code> artifacts/api-server/data/arcface_dataset/&lt;uuid&gt;/</code>
              запустите <code>cd artifacts/api-server &amp;&amp; uv run python -m app.train_and_embed</code>,
              затем обновите индекс. Скрипт пересчитывает эмбеддинги; он не
              переобучает веса модели.
            </p>

            <h3 className="guide-subheading">Очередь ручной проверки</h3>
            <div className="guide-table-wrap">
              <table className="guide-api-table">
                <thead><tr><th>Метод</th><th>Путь</th><th>Назначение</th></tr></thead>
                <tbody>
                  <ApiOperation method="GET" path="/api/verification/review-queue" description="Сессии, связанные с задачами в статусе open или in_review." />
                  <ApiOperation method="POST" path="/api/verification/sessions/{session_id}/review-decision" description="Подтвердить, отклонить или запросить повторную проверку." />
                  <ApiOperation method="GET" path="/api/verification/sessions/{session_id}/status" description="Статус сессии, системные значения, Ground Truth и состояние последней задачи." />
                  <ApiOperation method="GET" path="/api/verification/sessions/{session_id}" description="Полная карточка сессии верификации." />
                  <ApiOperation method="GET" path="/api/review/tasks" description="Задачи очереди ручной проверки." />
                  <ApiOperation method="POST" path="/api/review/tasks/{task_id}/decision" description="Совместимый метод решения по идентификатору задачи." />
                </tbody>
              </table>
            </div>

            <h3 className="guide-subheading">Принять решение по сессии</h3>
            <p>
              Для <code>approve</code> и <code>reject</code> обе метки Ground Truth
              обязательны. Для <code>request_retry</code> можно передать только
              комментарий; существующие метки при этом не заменяются.
            </p>
            <pre className="guide-code"><code>{`POST /api/verification/sessions/{session_id}/review-decision
Content-Type: application/json

{
  "status": "approve",
  "ground_truth_match": true,
  "ground_truth_liveness": true,
  "comment": "Проверено специалистом"
}`}</code></pre>
            <p className="guide-contract-label">Ответ — карточка задачи проверки:</p>
            <pre className="guide-code"><code>{`{
  "id": "123e4567-e89b-42d3-a456-426614174000",
  "kind": "verification",
  "subject_ref": "идентификатор сотрудника",
  "status": "resolved",
  "priority": "high",
  "summary": "Описание случая",
  "created_at": "2026-10-25T10:00:00Z",
  "resolved_at": "2026-10-25T10:05:00Z",
  "resolution": "approve",
  "reviewer_note": "Проверено специалистом",
  "verification_session_id": "323e4567-e89b-42d3-a456-426614174001"
}`}</code></pre>

            <h3 className="guide-subheading">Получить очередь и статус</h3>
            <p>Запрос очереди не требует тела. Каждый элемент содержит только краткую карточку <code>verification_session</code>.</p>
            <pre className="guide-code"><code>{`GET /api/verification/review-queue

[
  {
    "verification_session": {
      "id": "f00d510f-0b82-473c-95f5-57065ceb8e63",
      "employee_ref": "2158",
      "status": "manual_review",
      "match_score": 0.699604,
      "liveness_score": 0.7999,
      "quality_score": 0.8939,
      "decision": "review"
    }
  }
]

GET /api/verification/sessions/{session_id}/status`}</code></pre>
            <p>Ответ метода статуса:</p>
            <pre className="guide-code"><code>{`{
  "session_id": "323e4567-e89b-42d3-a456-426614174001",
  "status": "manual_review",
  "decision": "review",
  "is_match": true,
  "is_live": true,
  "ground_truth_match": null,
  "ground_truth_liveness": null,
  "review_status": "open",
  "review_resolution": null,
  "reviewer_note": null
}`}</code></pre>

            <div className="guide-api-errors">
              <b>Типовые ответы</b>
              <span><code>404</code> — сессия не найдена.</span>
              <span><code>409</code> — сессия не находится в активной очереди.</span>
              <span><code>422</code> — для подтверждения или отклонения не заданы обе метки.</span>
            </div>

            <details className="guide-endpoint-catalog">
              <summary>Краткий каталог остальных маршрутов</summary>
              <div className="guide-table-wrap">
                <table className="guide-api-table">
                  <thead><tr><th>Раздел</th><th>Основные маршруты</th></tr></thead>
                  <tbody>
                    <tr><td>Вход и состояние</td><td><code>GET /api/auth/user</code>, <code>GET /api/healthz</code></td></tr>
                    <tr><td>Сотрудники и фото</td><td><code>GET /api/employees</code>, <code>GET /api/employees/{'{employee_guid}'}/photo</code>, <code>POST /api/employees/{'{employee_guid}'}/photo/upload-url</code>, <code>POST /api/employees/{'{employee_guid}'}/photo</code></td></tr>
                    <tr><td>HEALTH_ID</td><td><code>GET /api/health-results/eligible-employees</code>, <code>GET /api/health-results</code>, <code>POST /api/health-results</code>, <code>GET /api/health-results/{'{result_id}'}</code></td></tr>
                    <tr><td>Верификация</td><td><code>GET /api/verification/sessions</code>, <code>POST /api/verification/sessions</code>, <code>POST /api/verification/sessions/{'{session_id}'}/complete</code></td></tr>
                    <tr><td>ArcFace</td><td><code>GET /api/arcface/status</code>, <code>POST /api/recognize</code>, <code>POST /api/employees/{'{employee_guid}'}/arcface-index</code>, <code>POST /api/refresh-index</code></td></tr>
                    <tr><td>Метрики</td><td><code>GET /api/verification/metrics</code>, <code>POST /api/verification/metrics/calculate</code></td></tr>
                    <tr><td>Мониторинг дрейфа</td><td><code>GET /api/drift/summary</code>, <code>GET /api/drift/history</code>, <code>POST /api/drift/run</code>, <code>POST /api/drift/control</code></td></tr>
                    <tr><td>Управление моделью</td><td><code>GET /api/governance/model-card</code>, <code>GET /api/governance/versions</code>, <code>GET /api/governance/models</code>, <code>POST /api/governance/models</code></td></tr>
                    <tr><td>Сводка и активность</td><td><code>GET /api/dashboard/summary</code>, <code>GET /api/activity</code></td></tr>
                  </tbody>
                </table>
              </div>
            </details>
          </GuideSection>
        </div>
      </div>
    </div>
  );
}
