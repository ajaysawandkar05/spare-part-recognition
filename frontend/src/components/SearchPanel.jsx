import { useState, useRef, useCallback } from "react";

const PART_TYPES = [
  "All",
  "photoelectric sensor",
  "relay",
  "actuator",
  "pneumatic",
  "connector",
  "bracket",
  "cable",
  "screw",
  "handle",
  "valve",
  "spring",
  "pin",
  "shaft",
  "drill bit",
  "cutting",
  "terminal block",
  "control panel",
];

export default function SearchPanel({ onSearch, loading }) {
  const [image, setImage]               = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [text, setText]                 = useState("");
  const [dragging, setDragging]         = useState(false);
  const [visualWeight, setVisualWeight] = useState(0.5);
  const [topK, setTopK]                 = useState(5);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [partType, setPartType]         = useState("All");
  const fileRef = useRef();

  const setFile = (file) => {
    if (!file) return;
    setImage(file);
    setImagePreview(URL.createObjectURL(file));
  };

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) setFile(file);
  }, []);

  const handleDragOver  = (e) => { e.preventDefault(); setDragging(true); };
  const handleDragLeave = () => setDragging(false);

  const clearImage = () => {
    setImage(null);
    setImagePreview(null);
    if (fileRef.current) fileRef.current.value = "";
  };

  const handleSubmit = () => {
    onSearch({
      image,
      text,
      visualWeight,
      textWeight: parseFloat((1 - visualWeight).toFixed(2)),
      topK,
      filterPartType: partType === "All" ? null : partType,
    });
  };

  const canSearch = (image || text.trim()) && !loading;
  const mode = image && text ? "multimodal" : image ? "visual" : text ? "text" : null;

  return (
    <section className="search-panel">
      <div className="search-grid">

        {/* Upload Zone */}
        <div
          className={`upload-zone ${dragging ? "upload-zone--drag" : ""} ${imagePreview ? "upload-zone--filled" : ""}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onClick={() => !imagePreview && fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            style={{ display: "none" }}
            onChange={(e) => setFile(e.target.files?.[0])}
          />
          {imagePreview ? (
            <div className="image-preview-wrap">
              <img src={imagePreview} alt="Query" className="image-preview" />
              <button className="clear-btn" onClick={(e) => { e.stopPropagation(); clearImage(); }}>
                ✕
              </button>
              <div className="preview-label">Query Image</div>
            </div>
          ) : (
            <div className="upload-placeholder">
              <svg className="upload-icon" viewBox="0 0 48 48" fill="none">
                <rect x="6" y="6" width="36" height="36" rx="6" stroke="currentColor" strokeWidth="1.5" />
                <path d="M24 32V20M24 20l-5 5M24 20l5 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                <path d="M14 38c0-4 2-6 10-6s10 2 10 6" stroke="currentColor" strokeWidth="1" strokeLinecap="round" opacity="0.4" />
              </svg>
              <p className="upload-title">Drop image here</p>
              <p className="upload-sub">or click to browse · JPG, PNG, WEBP</p>
            </div>
          )}
        </div>

        {/* Right column */}
        <div className="search-controls">
          <div className="mode-badge-wrap">
            {mode && (
              <span className={`mode-badge mode-badge--${mode}`}>
                {mode === "multimodal" ? "⬡ Multi-modal" : mode === "visual" ? "◈ Visual" : "◎ Text"}
              </span>
            )}
          </div>

          {/* Part type filter — METHOD 1 */}
          <div className="filter-row">
            <span className="filter-label">Part type</span>
            <select
              className="filter-select"
              value={partType}
              onChange={(e) => setPartType(e.target.value)}
            >
              {PART_TYPES.map((t) => (
                <option key={t} value={t}>{t}</option>
              ))}
            </select>
          </div>

          <div className="text-input-wrap">
            <textarea
              className="text-input"
              placeholder="Describe the part… e.g. 'Omron M18 infrared sensor'"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={4}
              onKeyDown={(e) => { if (e.key === "Enter" && e.ctrlKey) handleSubmit(); }}
            />
          </div>

          <button
            className="advanced-toggle"
            onClick={() => setShowAdvanced(!showAdvanced)}
          >
            {showAdvanced ? "▲" : "▼"} Advanced options
          </button>

          {showAdvanced && (
            <div className="advanced-panel">
              <div className="slider-row">
                <span className="slider-label">
                  Results
                  <span className="slider-value">{topK}</span>
                </span>
                <input
                  type="range" min="1" max="10" step="1"
                  value={topK} onChange={(e) => setTopK(Number(e.target.value))}
                />
              </div>
              <div className="slider-row">
                <span className="slider-label">
                  Visual weight
                  <span className="slider-value">{Math.round(visualWeight * 100)}%</span>
                </span>
                <input
                  type="range" min="0" max="1" step="0.05"
                  value={visualWeight} onChange={(e) => setVisualWeight(Number(e.target.value))}
                  disabled={!image || !text}
                />
              </div>
              <div className="weight-display">
                <span>Visual {Math.round(visualWeight * 100)}%</span>
                <div className="weight-bar">
                  <div className="weight-bar-fill weight-bar-fill--visual" style={{ width: `${visualWeight * 100}%` }} />
                </div>
                <span>Text {Math.round((1 - visualWeight) * 100)}%</span>
              </div>
            </div>
          )}

          <button
            className={`search-btn ${loading ? "search-btn--loading" : ""}`}
            onClick={handleSubmit}
            disabled={!canSearch}
          >
            {loading ? (
              <span className="spinner" />
            ) : (
              <>
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                  <circle cx="7" cy="7" r="5" stroke="currentColor" strokeWidth="1.5" />
                  <path d="M11 11l3 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                Find Matching Parts
              </>
            )}
          </button>
        </div>
      </div>
    </section>
  );
}