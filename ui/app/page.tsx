"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";

type ChatRole = "assistant" | "user";

type ChatMessage = {
  role: ChatRole;
  content: string;
  sources?: string[];
};

type StreamTokenEvent = { type: "token"; content: string };
type StreamSourcesEvent = { type: "sources"; content: string[] };
type StreamEvent = StreamTokenEvent | StreamSourcesEvent;

const defaultBaseUrl =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const defaultApiKey = process.env.NEXT_PUBLIC_API_KEY ?? "change-me";

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content:
        "Hi! I am your RAG copilot. Ask questions about your indexed documents and I will answer with sources.",
    },
  ]);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState("demo-ui-session");
  const [topK, setTopK] = useState(5);
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [apiKey, setApiKey] = useState(defaultApiKey);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState("");
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const [documents, setDocuments] = useState<string[]>([]);
  const [selectedSources, setSelectedSources] = useState<string[]>([]);

  const canSend = input.trim().length > 0 && !isStreaming;

  const updateLastAssistantMessage = (
    updater: (message: ChatMessage) => ChatMessage,
  ) => {
    setMessages((prev) => {
      if (!prev.length) return prev;
      const lastIndex = prev.length - 1;
      const last = prev[lastIndex];
      if (last.role !== "assistant") return prev;
      const updated = updater(last);
      return [...prev.slice(0, lastIndex), updated];
    });
  };
  const latestSources = useMemo<string[]>(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].sources?.length) {
        return messages[i].sources ?? [];
      }
    }
    return [];
  }, [messages]);
  const activeDocument = selectedSources[0] ?? "";
  const activeDocumentName =
    activeDocument.split("/").pop()?.split("\\").pop() ?? "this document";
  const starterPrompts = useMemo(
    () => [
      `Summarize ${activeDocumentName} in 5 bullet points.`,
      `What are the key highlights from ${activeDocumentName}?`,
      `List action items or takeaways from ${activeDocumentName}.`,
    ],
    [activeDocumentName],
  );

  const fetchDocuments = useCallback(async () => {
    try {
      const response = await fetch(`${baseUrl}/documents`, {
        headers: { "x-api-key": apiKey },
      });
      if (!response.ok) {
        throw new Error("Unable to fetch documents.");
      }
      const payload = (await response.json()) as { documents: string[] };
      setDocuments(payload.documents);
      setSelectedSources((prev) =>
        prev.filter((entry) => payload.documents.includes(entry)),
      );
    } catch {
      setDocuments([]);
    }
  }, [apiKey, baseUrl]);

  useEffect(() => {
    const timer = setTimeout(() => {
      void fetchDocuments();
    }, 0);
    return () => clearTimeout(timer);
  }, [fetchDocuments]);

  const handleSend = async (e: FormEvent, forcedPrompt?: string) => {
    e.preventDefault();
    const prompt = (forcedPrompt ?? input).trim();
    if (!prompt || isStreaming) return;

    setError("");
    setInput("");
    setIsStreaming(true);

    setMessages((prev) => [
      ...prev,
      { role: "user", content: prompt },
      { role: "assistant", content: "" },
    ]);

    try {
      const response = await fetch(`${baseUrl}/ask/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": apiKey,
        },
        body: JSON.stringify({
          query: prompt,
          session_id: sessionId,
          top_k: topK,
          source_filters: selectedSources,
          history: [],
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`Request failed (${response.status}). Check API key or backend URL.`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let done = false;
      let collectedSources: string[] = [];

      while (!done) {
        const { value, done: streamDone } = await reader.read();
        done = streamDone;
        buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });

        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";

        for (const event of events) {
          const line = event
            .split("\n")
            .find((entry) => entry.startsWith("data: "));
          if (!line) continue;
          const payload = line.replace("data: ", "").trim();
          if (payload === "[DONE]") continue;

          const parsed = JSON.parse(payload) as StreamEvent;

          if (parsed.type === "token" && typeof parsed.content === "string") {
            updateLastAssistantMessage((last) => ({
              ...last,
              content: last.content + parsed.content,
            }));
          }

          if (parsed.type === "sources" && Array.isArray(parsed.content)) {
            collectedSources = parsed.content;
            updateLastAssistantMessage((last) => ({
              ...last,
              sources: parsed.content,
            }));
          }
        }
      }

      if (!collectedSources.length) {
        updateLastAssistantMessage((last) => ({ ...last, sources: [] }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error.");
      updateLastAssistantMessage((last) =>
        last.content
          ? last
          : {
              ...last,
              content: "I could not reach the API. Verify settings and try again.",
            },
      );
    } finally {
      setIsStreaming(false);
    }
  };

  const handleUpload = async (file: File | null) => {
    if (!file) return;
    setUploadStatus("");
    setError("");
    setUploading(true);
    try {
      const form = new FormData();
      form.append("file", file);
      const response = await fetch(`${baseUrl}/documents/upload`, {
        method: "POST",
        headers: { "x-api-key": apiKey },
        body: form,
      });
      if (!response.ok) {
        throw new Error(`Upload failed (${response.status}).`);
      }
      const payload = (await response.json()) as {
        filename: string;
        source_id: string;
        chunks_indexed: number;
      };
      setUploadStatus(
        `Uploaded ${payload.filename} and indexed ${payload.chunks_indexed} chunks.`,
      );
      await fetchDocuments();
      setSelectedSources([payload.source_id]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex w-full max-w-7xl gap-6 px-4 py-6 lg:px-8">
        <section className="hidden w-80 shrink-0 rounded-2xl border border-white/10 bg-slate-900/70 p-5 backdrop-blur lg:block">
          <h1 className="text-lg font-semibold">RAG Studio</h1>
          <p className="mt-2 text-sm text-slate-300">
            Stream answers from your FastAPI backend with citations.
          </p>

          <div className="mt-6 space-y-4">
            <label className="block text-xs text-slate-400">
              Backend URL
              <input
                className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm outline-none ring-indigo-500/40 focus:ring"
                value={baseUrl}
                onChange={(e) => setBaseUrl(e.target.value)}
              />
            </label>
            <label className="block text-xs text-slate-400">
              API Key
              <input
                className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm outline-none ring-indigo-500/40 focus:ring"
                value={apiKey}
                onChange={(e) => {
                  setApiKey(e.target.value);
                }}
                onBlur={fetchDocuments}
              />
            </label>
            <label className="block text-xs text-slate-400">
              Session ID
              <input
                className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm outline-none ring-indigo-500/40 focus:ring"
                value={sessionId}
                onChange={(e) => setSessionId(e.target.value)}
              />
            </label>
            <label className="block text-xs text-slate-400">
              Top-K Chunks
              <input
                type="number"
                min={1}
                max={20}
                className="mt-1 w-full rounded-lg border border-white/10 bg-slate-950 px-3 py-2 text-sm outline-none ring-indigo-500/40 focus:ring"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              />
            </label>
          </div>

          <div className="mt-6 rounded-xl border border-white/10 bg-slate-950/60 p-3">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                Query Scope
              </h2>
              <button
                type="button"
                onClick={fetchDocuments}
                className="text-[11px] text-indigo-300 hover:text-indigo-200"
              >
                Refresh
              </button>
            </div>
            <p className="mt-1 text-xs text-slate-500">
              Select documents to use as retrieval context.
            </p>
            <div className="mt-3 max-h-36 space-y-2 overflow-auto pr-1">
              {documents.length ? (
                documents.map((doc) => (
                  <label key={doc} className="flex items-center gap-2 text-xs text-slate-300">
                    <input
                      type="checkbox"
                      checked={selectedSources.includes(doc)}
                      onChange={(e) => {
                        setSelectedSources((prev) =>
                          e.target.checked
                            ? [...prev, doc]
                            : prev.filter((entry) => entry !== doc),
                        );
                      }}
                    />
                    <span className="truncate">{doc}</span>
                  </label>
                ))
              ) : (
                <p className="text-xs text-slate-500">No indexed docs found.</p>
              )}
            </div>
            <p className="mt-2 text-[11px] text-slate-500">
              {selectedSources.length
                ? `Using ${selectedSources.length} selected source(s).`
                : "No filter selected: all indexed docs will be used."}
            </p>
            {selectedSources.length === 1 ? (
              <p className="mt-1 text-[11px] text-emerald-300">
                Suggestions and answers are focused on the selected document.
              </p>
            ) : null}
          </div>

          <div className="mt-6 rounded-xl border border-white/10 bg-slate-950/60 p-3">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Upload Document
            </h2>
            <p className="mt-1 text-xs text-slate-500">Supports .txt, .md, .pdf, .docx</p>
            <input
              type="file"
              accept=".txt,.md,.pdf,.docx"
              onChange={(e) => handleUpload(e.target.files?.[0] ?? null)}
              disabled={uploading}
              className="mt-3 block w-full text-xs text-slate-300 file:mr-3 file:rounded-md file:border-0 file:bg-indigo-500 file:px-3 file:py-2 file:text-xs file:font-semibold file:text-white hover:file:bg-indigo-400 disabled:opacity-60"
            />
            {uploadStatus ? (
              <p className="mt-2 text-xs text-emerald-300">{uploadStatus}</p>
            ) : null}
          </div>

          <div className="mt-6">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-slate-400">
              Latest Sources
            </h2>
            <div className="mt-2 space-y-2">
              {latestSources.length ? (
                latestSources.map((source) => (
                  <div
                    key={source}
                    className="rounded-lg border border-emerald-400/30 bg-emerald-500/10 px-3 py-2 text-xs text-emerald-200"
                  >
                    {source}
                  </div>
                ))
              ) : (
                <p className="text-xs text-slate-500">Sources appear after first response.</p>
              )}
            </div>
          </div>
        </section>

        <section className="flex min-h-[85vh] flex-1 flex-col rounded-2xl border border-white/10 bg-slate-900/70 backdrop-blur">
          <header className="flex items-center justify-between border-b border-white/10 px-4 py-3 sm:px-6">
            <div>
              <h2 className="text-base font-semibold">Document Chat</h2>
              <p className="text-xs text-slate-400">
                {isStreaming ? "Streaming response..." : "Ready"}
              </p>
            </div>
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300">
              {sessionId}
            </span>
          </header>

          <div className="flex-1 space-y-4 overflow-y-auto px-4 py-5 sm:px-6">
            {messages.map((message, idx) => (
              <div
                key={`${message.role}-${idx}`}
                className={`flex items-start gap-3 ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                {message.role === "assistant" ? (
                  <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 text-xs font-semibold text-emerald-300">
                    AI
                  </div>
                ) : null}
                <article
                  className={`max-w-3xl rounded-2xl px-4 py-3 ${
                    message.role === "user"
                      ? "bg-indigo-500 text-white"
                      : "border border-white/10 bg-slate-800 text-slate-100"
                  }`}
                >
                  <p className="whitespace-pre-wrap text-sm leading-6">{message.content}</p>
                  {message.sources?.length ? (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {message.sources.map((source) => (
                        <span
                          key={source}
                          className="rounded-full bg-emerald-500/20 px-2 py-1 text-[11px] text-emerald-200"
                        >
                          {source}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </article>
                {message.role === "user" ? (
                  <div className="mt-1 flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-500/30 text-xs font-semibold text-indigo-200">
                    You
                  </div>
                ) : null}
              </div>
            ))}
          </div>

          <div className="border-t border-white/10 px-4 py-4 sm:px-6">
            <div className="mb-3 flex flex-wrap gap-2">
              {starterPrompts.map((prompt) => (
                <button
                  key={prompt}
                  type="button"
                  onClick={(e) => handleSend(e, prompt)}
                  disabled={isStreaming}
                  className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-200 transition hover:bg-white/10 disabled:opacity-40"
                >
                  {prompt}
                </button>
              ))}
            </div>

            <form onSubmit={(e) => handleSend(e)} className="flex gap-2">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about your indexed documents..."
                className="flex-1 rounded-xl border border-white/10 bg-slate-950 px-4 py-3 text-sm outline-none ring-indigo-500/40 placeholder:text-slate-500 focus:ring"
              />
              <button
                type="submit"
                disabled={!canSend}
                className="rounded-xl bg-indigo-500 px-5 py-3 text-sm font-semibold text-white transition hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                Send
              </button>
            </form>
            {error ? <p className="mt-2 text-xs text-rose-300">{error}</p> : null}
          </div>
        </section>
      </div>
    </main>
  );
}
