const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const MODE_LABELS = {
  multimodal: "Multi-modal",
  visual:     "Visual Only",
  text:       "Text Only",
};

export default function StatsBar({ results, queryMode, queryImage, loading }) {
  const topScore = results[0]?.final_score ?? 0;
  const avgScore = results.length
    ? (results.reduce((s, r) => s + r.final_score, 0) / results.length).toFixed(3)
    : 0;

  return (
    <div className="stats-bar">
      {queryImage && (
        <div className="stats-query-img">
          <img src={queryImage} alt="query" />
          <span>Your query</span>
        </div>
      )}
      <div className="stats-items">
        <div className="stat">
          <span className="stat-label">Matches</span>
          <span className="stat-value">{loading ? "—" : results.length}</span>
        </div>
        <div className="stat">
          <span className="stat-label">Top Score</span>
          <span className="stat-value stat-value--amber">{loading ? "—" : topScore.toFixed(3)}</span>
        </div>       
        <div className="stat">
          <span className="stat-label">Mode</span>
          <span className="stat-value stat-value--mode">{queryMode ? MODE_LABELS[queryMode] : "—"}</span>
        </div>
      </div>
    </div>
  );
}