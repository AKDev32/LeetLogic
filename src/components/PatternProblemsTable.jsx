import { ExternalLink } from "lucide-react";

const PatternProblemsTable = ({ problems, category }) => {
  const getSlug = (problemName) => {
    return problemName
      .toLowerCase()
      .replace(/[^a-z0-9\s-]/g, "")
      .replace(/\s+/g, "-")
      .replace(/-+/g, "-");
  };

  return (
    <div className="pattern-problems-table">
      {category && <h4 className="problems-category">{category}</h4>}
      <div className="problems-list">
        {problems.map((problem, idx) => (
          <a
            key={idx}
            href={`https://leetcode.com/problems/${getSlug(problem)}/`}
            target="_blank"
            rel="noopener noreferrer"
            className="problem-item"
          >
            <span className="problem-number">{idx + 1}</span>
            <span className="problem-name">{problem}</span>
            <ExternalLink size={14} className="problem-link-icon" />
          </a>
        ))}
      </div>
    </div>
  );
};

export default PatternProblemsTable;
