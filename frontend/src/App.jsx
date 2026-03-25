import { useState, useCallback, useRef } from "react";
import SearchPanel from "./components/SearchPanel";
import ResultsGrid from "./components/ResultsGrid";
import Header from "./components/Header";
import StatsBar from "./components/StatsBar";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [queryMode, setQueryMode] = useState(null);
  const [error, setError] = useState(null);
  const [queryImage, setQueryImage] = useState(null); // preview URL

  const handleSearch = useCallback(async ({ image, text, visualWeight, textWeight, topK, filterPartType }) => {
    if (!image && !text.trim()) return;
    setLoading(true);
    setError(null);

    const formData = new FormData();
    if (image) {
      formData.append("image", image);
      setQueryImage(URL.createObjectURL(image));
    } else {
      setQueryImage(null);
    }
    if (text) formData.append("text", text);
    formData.append("top_k", topK);
    formData.append("visual_weight", visualWeight);
    formData.append("text_weight", textWeight);
    if (filterPartType) formData.append("filter_part_type", filterPartType);

    try {
      const res = await fetch(`${API}/api/query`, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server error: ${res.status}`);
      const data = await res.json();
      setResults(data.results);
      setQueryMode(data.query_mode);
    } catch (err) {
      setError(err.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }, []);

  return (
    <div className="app">
      <Header />
      <main className="main">
        <SearchPanel onSearch={handleSearch} loading={loading} />
        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠</span>
            {error}
          </div>
        )}
        {(results.length > 0 || loading) && (
          <StatsBar results={results} queryMode={queryMode} queryImage={queryImage} loading={loading} />
        )}
        <ResultsGrid results={results} loading={loading} />
      </main>
    </div>
  );
}