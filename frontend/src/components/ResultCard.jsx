import { useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

function ScoreBar({ label, value, color }) {
  if (value === null || value === undefined) return null;
  const pct = Math.max(0, Math.min(1, value));
  return (
    <div className="score-row">
      <span className="score-label">{label}</span>
      <div className="score-track">
        <div className={`score-fill score-fill--${color}`} style={{ width: `${pct * 100}%` }} />
      </div>
      <span className="score-num">{value.toFixed(3)}</span>
    </div>
  );
}

export default function ResultCard({ result, isTop }) {
  const [imgError, setImgError] = useState(false);

  const {
    rank, material_id, final_score,
    visual_score, text_score, confidence,
    storage_bin, total_stock, storage_location, plant, duration,
  } = result;

  return (
    <div className={`result-card ${isTop ? "result-card--top" : ""}`}>
      {isTop && <div className="top-badge">Best Match</div>}

      {/* METHOD 8: Confidence warning */}
      {isTop && confidence === "low" && (
        <div className="confidence-warn">
          ⚠ Two close matches — try adding more details or a clearer photo
        </div>
      )}

      <div className="card-rank">#{rank}</div>

      <div className="card-image-wrap">
        {!imgError ? (
          <img
            src={`${API}/api/image/${encodeURIComponent(material_id)}`}
            alt={material_id}
            className="card-image"
            onError={() => setImgError(true)}
          />
        ) : (
          <div className="card-image-placeholder">
            <svg width="32" height="32" viewBox="0 0 32 32" fill="none">
              <rect x="4" y="4" width="24" height="24" rx="4" stroke="currentColor" strokeWidth="1" opacity="0.3" />
              <path d="M10 22l5-6 4 5 3-3 4 4" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.4" />
            </svg>
          </div>
        )}
      </div>

      <div className="card-body">
        <div className="card-id">
          <span className="card-id-label">Material ID</span>
          <span className="card-id-value">{material_id}</span>
        </div>

        <div className="card-scores">
          <ScoreBar label="Match"  value={final_score}  color="amber" />
          <ScoreBar label="Visual" value={visual_score} color="blue" />
          <ScoreBar label="Text"   value={text_score}   color="teal" />
        </div>

        <div className="card-meta">
          {storage_bin && (
            <div className="meta-item">
              <span className="meta-icon">⬡</span>
              <span className="meta-key">Bin</span>
              <span className="meta-val">{storage_bin}</span>
            </div>
          )}
          {total_stock !== undefined && total_stock !== null && (
            <div className="meta-item">
              <span className="meta-icon">◈</span>
              <span className="meta-key">Stock</span>
              <span className="meta-val">{total_stock}</span>
            </div>
          )}
          {plant && (
            <div className="meta-item">
              <span className="meta-icon">◎</span>
              <span className="meta-key">Plant</span>
              <span className="meta-val">{plant}</span>
            </div>
          )}
          {storage_location && (
            <div className="meta-item">
              <span className="meta-icon">◷</span>
              <span className="meta-key">Loc</span>
              <span className="meta-val">{storage_location}</span>
            </div>
          )}
          {duration !== undefined && duration !== null && (
            <div className="meta-item">
              <span className="meta-icon">◑</span>
              <span className="meta-key">Dur</span>
              <span className="meta-val">{duration}y</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}