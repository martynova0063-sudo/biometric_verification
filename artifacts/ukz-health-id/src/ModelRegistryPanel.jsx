import { useEffect, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { Archive, Check, CircleAlert, Pencil, Plus, RotateCcw, Save, X } from 'lucide-react';
import {
  getGetModelCardQueryKey,
  getGetModelDefaultsQueryKey,
  getListModelVersionsQueryKey,
  getListRegisteredModelsQueryKey,
  useActivateRegisteredModel,
  useArchiveRegisteredModel,
  useCreateRegisteredModel,
  useGetModelDefaults,
  useListRegisteredModels,
  useUpdateRegisteredModel,
} from '@workspace/api-client-react';
import './model-registry.css';

const toDraft = (model, fields) => ({
  model_name: model.model_name,
  version: model.version,
  purpose: model.purpose,
  change_summary: model.change_summary,
  parameters: Object.fromEntries(fields.map(({ key, value_type }) => {
    const value = model.parameters[key];
    return [key, value_type === 'array' ? value.join(', ') : value_type === 'number' ? String(value) : value];
  })),
});

const toParameterInputs = (parameters, fields) => Object.fromEntries(fields.map(({ key, value_type }) => {
  const value = parameters[key];
  return [key, value_type === 'array' ? value.join(', ') : value_type === 'number' ? String(value) : value];
}));

const toRequest = (draft, fields) => ({
  ...draft,
  parameters: Object.fromEntries(fields.map(({ key, value_type }) => {
    const value = draft.parameters[key];
    if (value_type === 'number') return [key, Number(value)];
    if (value_type === 'array') return [key, String(value).split(',').map((part) => Number(part.trim()))];
    return [key, value];
  })),
});

const statusName = (status) => ({ current: 'Активная', draft: 'Черновик', archived: 'В архиве' }[status] || status);

export default function ModelRegistryPanel({ onClose }) {
  const queryClient = useQueryClient();
  const defaults = useGetModelDefaults({
    query: { enabled: true, queryKey: getGetModelDefaultsQueryKey() },
  });
  const models = useListRegisteredModels({
    query: { enabled: true, queryKey: getListRegisteredModelsQueryKey() },
  });
  const createModel = useCreateRegisteredModel();
  const updateModel = useUpdateRegisteredModel();
  const activateModel = useActivateRegisteredModel();
  const archiveModel = useArchiveRegisteredModel();

  const [editingId, setEditingId] = useState(null);
  const [draft, setDraft] = useState(null);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [initialSelectionDone, setInitialSelectionDone] = useState(false);
  const fields = defaults.data?.parameter_fields || [];
  const pending = createModel.isPending || updateModel.isPending || activateModel.isPending || archiveModel.isPending;

  useEffect(() => {
    if (draft || !editingId) return;
    if (editingId === 'new' && defaults.data) {
      setDraft(toDraft({
        model_name: defaults.data.model_name,
        version: '',
        purpose: defaults.data.purpose,
        change_summary: defaults.data.change_summary,
        parameters: defaults.data.parameters,
      }, fields));
      return;
    }
    const selected = models.data?.find((model) => model.id === editingId);
    if (selected) setDraft(toDraft(selected, fields));
  }, [editingId, draft, defaults.data, models.data, fields]);

  useEffect(() => {
    if (initialSelectionDone || editingId || !models.data) return;
    const currentModel = models.data.find((model) => model.status === 'current');
    if (currentModel) {
      setDraft(toDraft(currentModel, fields));
      setEditingId(currentModel.id);
    }
    setInitialSelectionDone(true);
  }, [initialSelectionDone, editingId, models.data, fields]);

  const refreshModelData = () => {
    queryClient.invalidateQueries({ queryKey: getListRegisteredModelsQueryKey() });
    queryClient.invalidateQueries({ queryKey: getGetModelCardQueryKey() });
    queryClient.invalidateQueries({ queryKey: getListModelVersionsQueryKey() });
  };

  const beginNew = () => {
    setError('');
    setNotice('');
    setDraft(null);
    setEditingId('new');
  };

  const beginEdit = (model) => {
    setError('');
    setNotice('');
    setDraft(toDraft(model, fields));
    setEditingId(model.id);
  };

  const save = (event) => {
    event.preventDefault();
    setError('');
    setNotice('');
    const data = toRequest(draft, fields);
    const callbacks = {
      onSuccess: () => {
        refreshModelData();
        setEditingId(null);
        setDraft(null);
        setNotice('Изменения сохранены.');
      },
      onError: (requestError) => {
        const detail = requestError?.data?.detail;
        setError(typeof detail === 'string' ? detail : 'Не удалось сохранить модель. Проверьте версию и значения параметров.');
      },
    };
    if (editingId === 'new') createModel.mutate({ data }, callbacks);
    else updateModel.mutate({ modelId: editingId, data }, callbacks);
  };

  const activate = (model) => {
    setError('');
    activateModel.mutate({ modelId: model.id }, {
      onSuccess: () => {
        refreshModelData();
        setNotice(`Модель ${model.version} теперь активна.`);
      },
      onError: () => setError('Не удалось активировать модель.'),
    });
  };

  const archive = (model) => {
    if (!window.confirm(`Архивировать модель ${model.model_name} ${model.version}? Её можно будет снова активировать.`)) return;
    setError('');
    archiveModel.mutate({ modelId: model.id }, {
      onSuccess: () => {
        refreshModelData();
        setNotice(`Модель ${model.version} перемещена в архив.`);
      },
      onError: (requestError) => {
        const detail = requestError?.data?.detail;
        setError(typeof detail === 'string' ? detail : 'Не удалось архивировать модель.');
      },
    });
  };

  const updateTextField = (key, value) => setDraft((current) => ({ ...current, [key]: value }));
  const updateParameter = (key, value) => setDraft((current) => ({
    ...current,
    parameters: { ...current.parameters, [key]: value },
  }));

  return <section className="registry-panel soft-card" data-testid="panel-model-registry">
    <header className="registry-head">
      <div>
        <div className="eyebrow">Реестр и настройки</div>
        <h2 className="display">Модели и параметры</h2>
        <p>Новая модель получает системные значения из verification_config.py. Она начнёт использоваться для верификации только после активации.</p>
      </div>
      <button className="icon-button registry-close" type="button" aria-label="Закрыть карточку модели" data-testid="button-close-model-registry" onClick={onClose}><X size={17} /></button>
    </header>

    {notice && <div className="registry-notice" role="status" data-testid="status-model-registry"><Check size={15} />{notice}</div>}
    {error && <div className="form-alert form-alert-error registry-error" role="alert" data-testid="error-model-registry"><CircleAlert size={15} />{error}</div>}

    {models.isLoading || defaults.isLoading ? <div className="registry-loading" aria-label="Загрузка настроек модели">Загрузка модели и системных значений…</div>
      : models.isError || defaults.isError ? <div className="form-alert form-alert-error">Не удалось загрузить модели и системные параметры. Обновите карточку модели.</div>
        : <div className="registry-layout">
          <div className="registry-list">
            <div className="registry-list-head">
              <span className="eyebrow">Версии модели</span>
              <button className="button button-primary" type="button" data-testid="button-add-model" onClick={beginNew} disabled={pending}><Plus size={14} /> Добавить</button>
            </div>
            {models.data?.map((model) => <article key={model.id} className={`registry-model ${model.status === 'current' ? 'registry-model-current' : ''}`} data-testid={`card-model-${model.id}`}>
              <div className="registry-model-top"><StatusText status={model.status} modelId={model.id} /><button className="icon-button registry-edit" type="button" aria-label={`Изменить ${model.model_name} ${model.version}`} title="Изменить" data-testid={`button-edit-model-${model.id}`} onClick={() => beginEdit(model)}><Pencil size={14} /></button></div>
              <h3>{model.model_name}</h3>
              <div className="registry-version mono">{model.version}</div>
              <p>{model.change_summary}</p>
              <div className="registry-model-actions">
                {model.status !== 'current' && <button className="button button-secondary" type="button" data-testid={`button-activate-model-${model.id}`} disabled={pending} onClick={() => activate(model)}><Check size={13} /> Активировать</button>}
                {model.status !== 'archived' && model.status !== 'current' && <button className="button button-danger" type="button" data-testid={`button-archive-model-${model.id}`} disabled={pending} onClick={() => archive(model)}><Archive size={13} /> В архив</button>}
                {model.status === 'archived' && <button className="button button-secondary" type="button" data-testid={`button-restore-model-${model.id}`} disabled={pending} onClick={() => activate(model)}><RotateCcw size={13} /> Восстановить</button>}
              </div>
            </article>)}
            {!models.data?.length && <p className="registry-empty">Моделей пока нет.</p>}
          </div>

          <div className="registry-editor">
            {!editingId ? <div className="registry-empty-editor"><Pencil size={20} /><b>Выберите модель для изменения</b><span>Или добавьте новую версию с системными параметрами по умолчанию.</span></div>
              : !draft ? <div className="registry-loading">Подготовка формы…</div>
                : <form onSubmit={save} data-testid="form-model-editor">
                  <div className="registry-editor-head">
                    <div><div className="eyebrow">{editingId === 'new' ? 'Новая версия' : 'Изменение конфигурации'}</div><h3>{editingId === 'new' ? 'Добавить модель' : 'Параметры модели'}</h3></div>
                    <button className="icon-button registry-edit" type="button" aria-label="Закрыть форму" data-testid="button-cancel-model-edit" onClick={() => { setEditingId(null); setDraft(null); }}><X size={16} /></button>
                  </div>
                  <div className="registry-meta-fields">
                    <label className="field"><span>Название модели</span><input required maxLength="120" data-testid="input-model-name" value={draft.model_name} onChange={(event) => updateTextField('model_name', event.target.value)} /></label>
                    <label className="field"><span>Версия</span><input required maxLength="60" data-testid="input-model-version" placeholder="Например, 1.1.0-research" value={draft.version} onChange={(event) => updateTextField('version', event.target.value)} /></label>
                    <label className="field registry-wide-field"><span>Назначение</span><textarea required maxLength="2000" rows="2" data-testid="input-model-purpose" value={draft.purpose} onChange={(event) => updateTextField('purpose', event.target.value)} /></label>
                    <label className="field registry-wide-field"><span>Описание версии</span><textarea required maxLength="2000" rows="2" data-testid="input-model-summary" value={draft.change_summary} onChange={(event) => updateTextField('change_summary', event.target.value)} /></label>
                  </div>
                  {Object.entries(fields.reduce((groups, field) => {
                    (groups[field.group] ||= []).push(field);
                    return groups;
                  }, {})).map(([group, groupFields]) => <fieldset className="registry-parameter-group" key={group}>
                    <legend>{group}</legend>
                    <div className="registry-parameter-grid">
                      {groupFields.map((field) => <label className={`registry-param-field ${field.value_type === 'boolean' ? 'registry-param-toggle' : ''}`} key={field.key}>
                        <span>{field.label}<small className="mono">{field.key}</small></span>
                        {field.value_type === 'boolean'
                          ? <input type="checkbox" data-testid={`input-model-param-${field.key}`} checked={Boolean(draft.parameters[field.key])} onChange={(event) => updateParameter(field.key, event.target.checked)} />
                          : field.value_type === 'select'
                            ? <select data-testid={`input-model-param-${field.key}`} value={draft.parameters[field.key] ?? ''} onChange={(event) => updateParameter(field.key, event.target.value)}>
                              {!draft.parameters[field.key] && <option value="" disabled>Выберите вариант</option>}
                              {field.options?.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                            </select>
                          : <input type={field.value_type === 'number' ? 'number' : 'text'} step={field.value_type === 'number' ? 'any' : undefined} data-testid={`input-model-param-${field.key}`} value={draft.parameters[field.key] ?? ''} onChange={(event) => updateParameter(field.key, event.target.value)} />}
                      </label>)}
                    </div>
                  </fieldset>)}
                  <div className="registry-form-actions">
                    <button className="button button-secondary" type="button" data-testid="button-reset-model-defaults" disabled={pending || !defaults.data} onClick={() => setDraft((current) => ({ ...current, parameters: toParameterInputs(defaults.data.parameters, fields) }))}><RotateCcw size={14} /> Системные значения</button>
                    <button className="button button-primary" type="submit" data-testid="button-save-model" disabled={pending}>{pending ? 'Сохраняем…' : <><Save size={14} /> Сохранить</>}</button>
                  </div>
                </form>}
          </div>
        </div>}
  </section>;
}

function StatusText({ status, modelId }) {
  return <span className={`registry-status registry-status-${status}`} data-testid={`status-model-${modelId}`}><i />{statusName(status)}</span>;
}