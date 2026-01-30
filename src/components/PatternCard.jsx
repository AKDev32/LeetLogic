import { useState } from "react";
import { ChevronDown, ChevronUp, Code2 } from "lucide-react";
import PatternsCodeBlock from "./PatternsCodeBlock";
import PatternProblemsTable from "./PatternProblemsTable";

const PatternCard = ({ pattern }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="pattern-detail-card">
      <div
        className="pattern-detail-header"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="pattern-header-content">
          <Code2 size={20} className="pattern-icon" />
          <div>
            <h3 className="pattern-detail-title">{pattern.title}</h3>
            <p className="pattern-detail-subtitle">{pattern.description}</p>
          </div>
        </div>
        <button className="pattern-expand-btn">
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>

      {expanded && (
        <div className="pattern-detail-content">
          {/* Code Template */}
          {pattern.code && (
            <PatternsCodeBlock
              code={pattern.code}
              title="Template"
              description="Copy and adapt this template for similar problems"
            />
          )}

          {/* Problems List */}
          {pattern.problems && (
            <div className="pattern-problems-section">
              <h4 className="section-title">Practice Problems</h4>
              {pattern.categories ? (
                pattern.categories.map((cat, idx) => (
                  <PatternProblemsTable
                    key={idx}
                    category={cat.category}
                    problems={cat.problems}
                  />
                ))
              ) : (
                <PatternProblemsTable problems={pattern.problems} />
              )}
            </div>
          )}

          {/* Use Cases */}
          {pattern.useCases && (
            <div className="pattern-use-cases">
              <h4 className="section-title">When to Use</h4>
              <ul className="use-cases-list">
                {pattern.useCases.map((useCase, idx) => (
                  <li key={idx}>{useCase}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PatternCard;
