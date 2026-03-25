import ResultCard from "./ResultCard";

export default function ResultsGrid({ results, loading }) {
  if (loading) {
    return (
      <div className="results-grid">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="result-card result-card--skeleton">
            <div className="skeleton-img" />
            <div className="skeleton-body">
              <div className="skeleton-line skeleton-line--wide" />
              <div className="skeleton-line" />
              <div className="skeleton-line skeleton-line--short" />
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (results.length === 0) return null;

  return (
    <div className="results-section">
      <div className="results-header">
        <h2 className="results-title">Matching Parts</h2>
        <span className="results-count">{results.length} results</span>
      </div>
      <div className="results-grid">
        {results.map((result) => (
          <ResultCard key={result.material_id} result={result} isTop={result.rank === 1} />
        ))}
      </div>
    </div>
  );
}