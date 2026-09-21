import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API_BASE = "http://localhost:8000";

const LOADING_PHRASES = [
  "Thinking…",
  "Searching the episodes…",
  "Looking for the right moment…",
  "Piecing it together…",
  "Checking the transcript…",
  "Almost there…",
];

const BUILD_STEPS = [
  { key: "transcribe", label: "Transcribe" },
  { key: "chunk", label: "Chunk" },
  { key: "index", label: "Embed" },
];

function formatTime(seconds) {
  if (seconds == null) return "?";
  const total = Math.round(seconds);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function App() {
  const [episodes, setEpisodes] = useState([]);
  const [episodesError, setEpisodesError] = useState(null);
  const [currentEpisode, setCurrentEpisode] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatError, setChatError] = useState(null);
  const [phraseIndex, setPhraseIndex] = useState(0);
  const [architecture, setArchitecture] = useState("fixed"); // "fixed" | "agent" - see ChatRequest in chat_routes.py

  const [chatHealth, setChatHealth] = useState(null);
  const [buildStatus, setBuildStatus] = useState(null);
  const [buildError, setBuildError] = useState(null);

  const audioRef = useRef(null);
  const sessionIdRef = useRef(crypto.randomUUID());
  const messagesEndRef = useRef(null);
  const pendingSeekRef = useRef(null);
  const inputRef = useRef(null);

  const building = buildStatus?.state === "running";

  function fetchEpisodes() {
    fetch(`${API_BASE}/episodes`)
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json();
      })
      .then((data) => {
        setEpisodes(data);
        setEpisodesError(null);
      })
      .catch((err) => setEpisodesError(err.message));
  }

  function fetchChatHealth() {
    fetch(`${API_BASE}/chat/health`)
      .then((res) => res.json())
      .then(setChatHealth)
      .catch(() => {});
  }

  useEffect(() => {
    fetchEpisodes();
    fetchChatHealth();
    fetch(`${API_BASE}/pipeline/status`)
      .then((res) => res.json())
      .then(setBuildStatus)
      .catch(() => {});
    inputRef.current?.focus();
  }, []);

  // Poll build progress while a build is running.
  useEffect(() => {
    if (!building) return;
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/pipeline/status`);
        const data = await res.json();
        setBuildStatus(data);
        if (data.state === "completed" || data.state === "failed") {
          fetchEpisodes();
          fetchChatHealth();
        }
      } catch {
        // transient network hiccup while polling - try again next tick
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [building]);

  // Cycle "still working" phrases so a long answer doesn't feel stalled.
  useEffect(() => {
    if (!loading) {
      setPhraseIndex(0);
      return;
    }
    const interval = setInterval(() => {
      setPhraseIndex((i) => (i + 1) % LOADING_PHRASES.length);
    }, 1700);
    return () => clearInterval(interval);
  }, [loading]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  async function startBuild() {
    setBuildError(null);
    try {
      const res = await fetch(`${API_BASE}/pipeline/build`, { method: "POST" });
      const data = await res.json();
      if (data.status === "started" || data.status === "already_running") {
        setBuildStatus({ state: "running", steps: [], error: null });
      }
    } catch (err) {
      setBuildError(err.message);
    }
  }

  function loadEpisode(episode, seekTo) {
    const isSameEpisode = currentEpisode?.episode_number === episode.episode_number;
    setCurrentEpisode(episode);
    const audio = audioRef.current;
    if (!audio) return;

    if (isSameEpisode) {
      if (seekTo != null) audio.currentTime = seekTo;
      audio.play();
      return;
    }

    pendingSeekRef.current = seekTo ?? null;
    audio.src = `${API_BASE}/episodes/${episode.episode_number}/audio`;
    audio.load();
  }

  function handleAudioLoadedMetadata() {
    const audio = audioRef.current;
    if (!audio) return;
    if (pendingSeekRef.current != null) {
      audio.currentTime = pendingSeekRef.current;
      pendingSeekRef.current = null;
    }
    audio.play();
  }

  function playCitation(citation) {
    const episode = episodes.find((e) => e.episode_number === citation.episode_number);
    if (!episode) return;
    loadEpisode(episode, citation.start);
  }

  async function sendMessage(e) {
    e.preventDefault();
    const query = input.trim();
    if (!query || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: query }]);
    setInput("");
    setLoading(true);
    setChatError(null);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionIdRef.current, query, architecture }),
      });
      if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
      const data = await res.json();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.answer,
          citations: data.citations || [],
          refused: data.refused,
          queryType: data.query_type,
          retrievalMode: data.retrieval_mode,
          architecture,
        },
      ]);
    } catch (err) {
      setChatError(err.message);
    } finally {
      setLoading(false);
      inputRef.current?.focus();
    }
  }

  return (
    <div className="app">
      <aside className="episodes-panel">
        <div className="brand">
          <h1 className="panel-title">Tracer</h1>
          <p className="panel-subtitle">Physics podcast companion</p>
        </div>

        <div className="build-panel">
          <div className="build-status-row">
            <span className={`status-dot ${chatHealth?.status === "ok" ? "ok" : "warn"}`} />
            <span className="build-status-text">
              {chatHealth
                ? `${chatHealth.indexed_chunk_count ?? 0} chunks indexed`
                : "Checking index…"}
            </span>
            <button className="build-button" onClick={startBuild} disabled={building}>
              {building ? "Building…" : "Build Index"}
            </button>
          </div>

          {buildStatus && buildStatus.state !== "idle" && (
            <div className="build-progress">
              {BUILD_STEPS.map(({ key, label }) => {
                const doneStep = buildStatus.steps?.find((s) => s.step === key);
                const stepIdx = BUILD_STEPS.findIndex((s) => s.key === key);
                const isCurrent =
                  !doneStep && building && stepIdx === (buildStatus.steps?.length ?? 0);
                const failed = doneStep && doneStep.returncode !== 0;
                const cls = failed ? "failed" : doneStep ? "done" : isCurrent ? "current" : "pending";
                return (
                  <div key={key} className={`build-step ${cls}`}>
                    <span className="step-icon" />
                    <span>{label}</span>
                  </div>
                );
              })}
            </div>
          )}

          {buildStatus?.state === "failed" && (
            <p className="error-text">Build failed: {buildStatus.error}</p>
          )}
          {buildError && <p className="error-text">{buildError}</p>}
        </div>

        <div className="player">
          <div className="player-label">
            {currentEpisode
              ? `Episode ${currentEpisode.episode_number} — ${currentEpisode.title}`
              : "No episode selected"}
          </div>
          <audio
            ref={audioRef}
            controls
            onLoadedMetadata={handleAudioLoadedMetadata}
            onPlay={() => setIsPlaying(true)}
            onPause={() => setIsPlaying(false)}
          />
        </div>

        {episodesError && <p className="error-text">Couldn't load episodes: {episodesError}</p>}

        <ul className="episode-list">
          {episodes.map((ep) => {
            const active = currentEpisode?.episode_number === ep.episode_number;
            return (
              <li
                key={ep.episode_number}
                className={active ? "episode-item active" : "episode-item"}
                onClick={() => loadEpisode(ep)}
              >
                <span className="episode-number">Ep {ep.episode_number}</span>
                <span className="episode-title">{ep.title}</span>
                {active && isPlaying ? (
                  <span className="now-playing" aria-label="Now playing">
                    <span />
                    <span />
                    <span />
                  </span>
                ) : (
                  <span className="episode-duration">{formatTime(ep.duration_seconds)}</span>
                )}
              </li>
            );
          })}
        </ul>
      </aside>

      <main className="chat-panel">
        <div className="architecture-toggle">
          <span className="architecture-label">Architecture:</span>
          <div className="architecture-switch" role="group" aria-label="Choose retrieval architecture">
            <button
              type="button"
              className={architecture === "fixed" ? "arch-option active" : "arch-option"}
              onClick={() => setArchitecture("fixed")}
              title="The original intent-classifier + fixed router pipeline"
            >
              Fixed pipeline
            </button>
            <button
              type="button"
              className={architecture === "agent" ? "arch-option active" : "arch-option"}
              onClick={() => setArchitecture("agent")}
              title="Experimental ReAct agent (create_agent + LangGraph) - see REACT_AGENT_DESIGN.md"
            >
              ReAct agent (experimental)
            </button>
          </div>
        </div>

        <div className="messages">
          {messages.length === 0 && (
            <p className="empty-state">Ask a question about the podcast episodes to get started.</p>
          )}
          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              {msg.role === "assistant" && msg.architecture && (
                <span className={`arch-badge arch-badge-${msg.architecture}`}>
                  {msg.architecture === "agent" ? "ReAct agent" : "Fixed pipeline"}
                </span>
              )}
              {msg.role === "assistant" ? (
                <div className="message-text markdown">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
                </div>
              ) : (
                <div className="message-text plain">{msg.text}</div>
              )}
              {msg.citations && msg.citations.length > 0 && (
                <div className="citations">
                  {msg.citations.map((c, ci) => (
                    <button key={ci} className="citation-chip" onClick={() => playCitation(c)}>
                      Ep {c.episode_number} · {formatTime(c.start)}–{formatTime(c.end)}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="message-text typing">
                {LOADING_PHRASES[phraseIndex]}
                <span className="dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </span>
              </div>
            </div>
          )}
          {chatError && <p className="error-text">Something went wrong: {chatError}</p>}
          <div ref={messagesEndRef} />
        </div>

        <form className="input-row" onSubmit={sendMessage}>
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about the episodes..."
            disabled={loading}
          />
          <button type="submit" disabled={loading || !input.trim()}>
            Send
          </button>
        </form>
      </main>
    </div>
  );
}
