"use client";

import { useEffect, useRef, useState } from "react";

import type { SettingsDictionary } from "@/app/[locale]/settings/page";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  ApiError,
  OfflineError,
  chatgptPlanModels,
  chatgptPlanStatus,
  selectChatGPTPlan,
  type ChatGPTPlanAccount,
  type ChatGPTPlanModel,
} from "@/lib/api";
import { useResource } from "@/lib/use-resource";

function accountName(account: ChatGPTPlanAccount): string {
  if (account.label && account.email) return `${account.label} (${account.email})`;
  return account.label || account.email || account.client_id;
}

function distinctAccountName(account: ChatGPTPlanAccount, accounts: ChatGPTPlanAccount[]): string {
  const name = accountName(account);
  // Email and user-supplied labels can match across separate workspace grants.
  // Keep the billing identity unambiguous without exposing any credentials.
  return accounts.some((entry) => entry.client_id !== account.client_id && accountName(entry) === name)
    ? `${name} [${account.client_id}]`
    : name;
}

function failureMessage(thrown: unknown, t: SettingsDictionary): string {
  if (thrown instanceof OfflineError) return t.chatgpt_offline;
  if (thrown instanceof ApiError) {
    if (thrown.status === 403) return t.chatgpt_forbidden;
    if (thrown.code === "chatgpt_not_connected") return t.chatgpt_sign_in_required;
    if (thrown.code === "chatgpt_model_unavailable") return t.chatgpt_model_unavailable;
    if (thrown.code === "chatgpt_unavailable") return t.chatgpt_unavailable;
  }
  return t.chatgpt_failed;
}

/** Local metadata is safe to read on mount. Upstream model discovery is click-only. */
export function ChatGPTPlanSettings({ t }: { t: SettingsDictionary }) {
  const status = useResource(chatgptPlanStatus);
  const [chosenClientId, setChosenClientId] = useState<string | null>(null);
  const [model, setModel] = useState("");
  const [catalogue, setCatalogue] = useState<{
    clientId: string;
    models: ChatGPTPlanModel[];
  } | null>(null);
  const [loadingModels, setLoadingModels] = useState(false);
  const [saving, setSaving] = useState(false);
  const [notice, setNotice] = useState<{ text: string; bad: boolean } | null>(null);
  // Account changes and unmounts invalidate late replies. A ref also closes the gap
  // before React renders disabled buttons after a repeated click.
  const request = useRef(0);
  const pending = useRef(false);

  useEffect(() => () => {
    request.current += 1;
  }, []);

  const data = status.data;
  const accounts = data?.accounts ?? [];
  const savedAccount = accounts.find((account) => account.client_id === data?.selected_client_id);
  const clientId = chosenClientId ?? (
    savedAccount?.connected ? savedAccount.client_id : accounts.find((account) => account.connected)?.client_id
  ) ?? "";
  const account = accounts.find((entry) => entry.client_id === clientId);
  const models = catalogue?.clientId === clientId ? catalogue.models : null;
  const active = data?.provider === "chatgpt_plan";
  const validModel = models?.some((entry) => entry.slug === model) ?? false;
  const alreadySelected = active && data?.selected_client_id === clientId && data?.model === model;
  const controlsDisabled = status.loading || saving;

  function changeAccount(value: string) {
    request.current += 1;
    pending.current = false;
    setChosenClientId(value);
    setCatalogue(null);
    setModel("");
    setLoadingModels(false);
    setNotice(null);
  }

  function refreshStatus() {
    if (pending.current) return;
    changeAccount("");
    setChosenClientId(null);
    status.reload();
  }

  async function refreshModels() {
    if (pending.current || !account?.connected || controlsDisabled) return;
    pending.current = true;
    const version = ++request.current;
    setLoadingModels(true);
    setCatalogue(null);
    setModel("");
    setNotice(null);
    try {
      const response = await chatgptPlanModels(clientId);
      if (request.current !== version) return;
      setCatalogue({ clientId, models: response.models });
      // Keep an existing selection only when this account still advertises it.
      if (data?.selected_client_id === clientId && response.models.some((entry) => entry.slug === data.model)) {
        setModel(data.model ?? "");
      }
    } catch (thrown) {
      if (request.current === version) setNotice({ text: failureMessage(thrown, t), bad: true });
    } finally {
      if (request.current === version) {
        pending.current = false;
        setLoadingModels(false);
      }
    }
  }

  async function selectPlan() {
    if (pending.current || !account?.connected || !validModel || alreadySelected || controlsDisabled) return;
    pending.current = true;
    const version = ++request.current;
    setSaving(true);
    setNotice(null);
    try {
      await selectChatGPTPlan(clientId, model);
      if (request.current !== version) return;
      setNotice({ text: t.chatgpt_saved, bad: false });
      status.reload();
    } catch (thrown) {
      if (request.current === version) setNotice({ text: failureMessage(thrown, t), bad: true });
    } finally {
      if (request.current === version) {
        pending.current = false;
        setSaving(false);
      }
    }
  }

  return (
    <section
      aria-labelledby="chatgpt-plan-title"
      className="border-od-line bg-od-panel-deep-3 flex flex-col gap-4 rounded-[10px] border p-[18px]"
    >
      <div>
        <h3 id="chatgpt-plan-title" className="text-od-text-3 m-0 text-[15px] font-semibold">
          {t.chatgpt_title}
        </h3>
        <p className="text-od-muted-5 mt-2 text-[13px] text-pretty">{t.chatgpt_description}</p>
        <p id="chatgpt-plan-scope" className="text-od-muted-5 mt-2 text-[13px] text-pretty">
          {t.chatgpt_scope}
        </p>
      </div>

      {status.loading && data === null ? (
        <div role="status" className="flex flex-col gap-2">
          <span className="text-od-muted-5 text-[13px]">{t.live_loading}</span>
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-2/3" />
        </div>
      ) : null}

      {status.error ? (
        <p role="alert" className="m-0 text-[13px] text-[color:var(--od-red-text-6)]">
          {status.error.kind === "offline" ? t.chatgpt_offline : status.error.kind === "forbidden" ? t.chatgpt_forbidden : t.chatgpt_failed}
        </p>
      ) : null}

      {data ? (
        <>
          <dl className="m-0 grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-2 text-[13px]">
            <dt className="text-od-muted-5">{t.chatgpt_status}</dt>
            <dd className="text-od-text-2 m-0">{active ? t.chatgpt_active : t.chatgpt_inactive}</dd>
            <dt className="text-od-muted-5">{t.chatgpt_saved_account}</dt>
            <dd className="text-od-text-2 m-0 break-words"><bdi>{savedAccount ? distinctAccountName(savedAccount, accounts) : t.chatgpt_none}</bdi></dd>
            <dt className="text-od-muted-5">{t.chatgpt_saved_model}</dt>
            <dd className="text-od-text-2 m-0 break-words"><bdi>{active && data.model ? data.model : t.chatgpt_none}</bdi></dd>
          </dl>

          {!accounts.some((entry) => entry.connected) ? (
            <p role="status" className="text-od-muted-5 m-0 text-[13px]">{t.chatgpt_sign_in_required}</p>
          ) : data.selected_client_id && !data.connected ? (
            <p role="status" className="text-od-muted-5 m-0 text-[13px]">{t.chatgpt_saved_disconnected}</p>
          ) : null}

          {accounts.length > 0 ? (
            <fieldset disabled={controlsDisabled} className="m-0 flex min-w-0 flex-col gap-3 border-0 p-0">
              <legend className="sr-only">{t.chatgpt_title}</legend>
              <label htmlFor="chatgpt-account" className="text-od-text-3 text-[13px] font-medium">{t.chatgpt_account}</label>
              <select
                id="chatgpt-account"
                value={clientId}
                onChange={(event) => changeAccount(event.target.value)}
                className="bg-od-canvas-2 border-od-border-6 text-od-text-2 w-full min-w-0 rounded-[7px] border p-[9px_12px] text-[13px] disabled:opacity-50"
              >
                <option value="" disabled>{t.chatgpt_choose_account}</option>
                {accounts.map((entry) => (
                  <option key={entry.client_id} value={entry.client_id} disabled={!entry.connected}>
                    {distinctAccountName(entry, accounts)}{entry.connected ? "" : ` (${t.chatgpt_sign_in_short})`}
                  </option>
                ))}
              </select>

              <p id="chatgpt-models-help" className="text-od-muted-5 m-0 text-[12.5px] text-pretty">{t.chatgpt_models_help}</p>
              <div className="flex flex-wrap gap-2">
                <Button type="button" variant="outline" onClick={refreshModels} disabled={loadingModels || !account?.connected} aria-describedby="chatgpt-models-help">
                  {loadingModels ? t.chatgpt_loading_models : t.chatgpt_refresh_models}
                </Button>
              </div>

              <label htmlFor="chatgpt-model" className="text-od-text-3 text-[13px] font-medium">{t.chatgpt_model}</label>
              <select
                id="chatgpt-model"
                value={model}
                disabled={loadingModels || !models?.length}
                onChange={(event) => { setModel(event.target.value); setNotice(null); }}
                className="bg-od-canvas-2 border-od-border-6 text-od-text-2 w-full min-w-0 rounded-[7px] border p-[9px_12px] text-[13px] disabled:opacity-50"
              >
                <option value="" disabled>{models === null ? t.chatgpt_refresh_first : t.chatgpt_choose_model}</option>
                {models?.map((entry) => <option key={entry.slug} value={entry.slug}>{entry.display_name || entry.slug}</option>)}
              </select>
              {models?.length === 0 ? <p role="status" className="text-od-muted-5 m-0 text-[13px]">{t.chatgpt_no_models}</p> : null}

              <div className="flex flex-wrap gap-2">
                <Button type="button" onClick={selectPlan} disabled={loadingModels || !account?.connected || !validModel || alreadySelected} aria-describedby="chatgpt-plan-scope">
                  {saving ? t.live_saving : alreadySelected ? t.chatgpt_selected : t.chatgpt_use}
                </Button>
              </div>
            </fieldset>
          ) : null}
        </>
      ) : null}

      {notice ? <p role={notice.bad ? "alert" : "status"} className="m-0 text-[13px]" style={{ color: notice.bad ? "var(--od-red-text-6)" : "var(--od-muted-5)" }}>{notice.text}</p> : null}

      <div className="flex flex-col gap-2 text-[13px]">
        <p className="text-od-muted-5 m-0 text-pretty">{t.chatgpt_local_sign_in}</p>
        <code dir="ltr" className="mono ltr-data text-od-text-2 break-words">python -m scripts.chatgpt_auth sign-in</code>
        <p className="text-od-muted-5 m-0 text-pretty">{t.chatgpt_vm_transfer}</p>
        <div className="flex flex-wrap gap-x-4 gap-y-2">
          <a href="https://developers.openai.com/siwc/token-sharing-open-source/self-hosted-vms" target="_blank" rel="noopener noreferrer" className="text-od-text-2 underline">{t.chatgpt_setup_guide}</a>
          <a href="https://chatgpt.com/settings/usage" target="_blank" rel="noopener noreferrer" className="text-od-text-2 underline">{t.chatgpt_usage}</a>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button type="button" variant="outline" onClick={refreshStatus} disabled={status.loading || loadingModels || saving}>
          {status.loading ? t.live_loading : t.chatgpt_refresh_status}
        </Button>
      </div>
    </section>
  );
}
