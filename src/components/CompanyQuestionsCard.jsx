import { useState, useMemo } from "react";
import {
  ChevronDown,
  ChevronUp,
  ExternalLink,
  TrendingUp,
  Building2,
  Search,
  X,
} from "lucide-react";

const CompanyQuestionsCard = ({ company }) => {
  const [expanded, setExpanded] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState(0);
  const [imageError, setImageError] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case "Easy":
        return "#10b981";
      case "Medium":
        return "#f59e0b";
      case "Hard":
        return "#ef4444";
      default:
        return "#9ca3af";
    }
  };

  const getFrequencyColor = (frequency) => {
    switch (frequency) {
      case "Very High":
        return "#ef4444";
      case "High":
        return "#f59e0b";
      case "Medium":
        return "#3b82f6";
      default:
        return "#9ca3af";
    }
  };

  // Check if logo is an image path or emoji
  const isImageLogo =
    company.logo &&
    (company.logo.endsWith(".png") ||
      company.logo.endsWith(".jpg") ||
      company.logo.endsWith(".svg") ||
      company.logo.endsWith(".webp") ||
      company.logo.endsWith(".avif") ||
      company.logo.startsWith("http"));

  // Filter questions based on search query
  const filteredQuestions = useMemo(() => {
    if (!searchQuery.trim()) {
      return company.categories[selectedCategory].questions;
    }

    const query = searchQuery.toLowerCase();
    return company.categories[selectedCategory].questions.filter(
      (question) =>
        question.name.toLowerCase().includes(query) ||
        question.leetcodeNum.toString().includes(query) ||
        question.difficulty.toLowerCase().includes(query) ||
        question.topics.some((topic) => topic.toLowerCase().includes(query)),
    );
  }, [company.categories, selectedCategory, searchQuery]);

  const clearSearch = () => {
    setSearchQuery("");
  };

  return (
    <div className="company-card card">
      <div className="card-header" onClick={() => setExpanded(!expanded)}>
        <div className="company-header-content">
          <div className="company-logo-wrapper">
            {isImageLogo && !imageError ? (
              <img
                src={company.logo}
                alt={`${company.name} logo`}
                className="company-logo-image"
                onError={() => setImageError(true)}
              />
            ) : (
              <span className="company-logo-emoji">{company.logo}</span>
            )}
            <div className="company-info">
              <h3 className="company-name">{company.name}</h3>
              <div className="company-meta">
                <span className="company-stat">
                  <Building2 size={14} />
                  {company.totalQuestions}+ questions
                </span>
                <span
                  className="frequency-badge"
                  style={{
                    background: `${getFrequencyColor(company.frequency)}20`,
                    color: getFrequencyColor(company.frequency),
                    border: `1px solid ${getFrequencyColor(company.frequency)}40`,
                  }}
                >
                  <TrendingUp size={12} />
                  {company.frequency} Frequency
                </span>
              </div>
            </div>
          </div>
        </div>
        <button className="expand-btn">
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>

      {expanded && (
        <div className="card-content">
          {/* Search Bar */}
          <div className="question-search-bar">
            <Search size={18} className="search-icon" />
            <input
              type="text"
              placeholder="Search by name, number, difficulty, or topic..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="question-search-input"
              onClick={(e) => e.stopPropagation()}
            />
            {searchQuery && (
              <button className="clear-search-btn" onClick={clearSearch}>
                <X size={18} />
              </button>
            )}
          </div>

          {/* Category Tabs */}
          <div className="category-tabs">
            {company.categories.map((cat, idx) => (
              <button
                key={idx}
                className={`category-tab ${selectedCategory === idx ? "active" : ""}`}
                onClick={() => {
                  setSelectedCategory(idx);
                  setSearchQuery(""); // Clear search when switching categories
                }}
              >
                {cat.category}
                <span className="question-count">{cat.questions.length}</span>
              </button>
            ))}
          </div>

          {/* Search Results Info */}
          {searchQuery && (
            <div className="search-results-info">
              Found {filteredQuestions.length} question
              {filteredQuestions.length !== 1 ? "s" : ""}
              {filteredQuestions.length > 0 && ` matching "${searchQuery}"`}
            </div>
          )}

          {/* Questions List */}
          <div className="questions-list">
            {filteredQuestions.length > 0 ? (
              filteredQuestions.map((question) => (
                <div key={question.id} className="question-item">
                  <div className="question-header">
                    <div className="question-title-section">
                      <span className="leetcode-num">
                        #{question.leetcodeNum}
                      </span>
                      <span className="question-name">{question.name}</span>
                    </div>
                    <a
                      href={`https://leetcode.com/problems/${question.name.toLowerCase().replace(/\s+/g, "-").replace(/[()]/g, "")}/`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="leetcode-link"
                      onClick={(e) => e.stopPropagation()}
                    >
                      <ExternalLink size={16} />
                    </a>
                  </div>

                  <div className="question-meta">
                    <span
                      className="difficulty-badge"
                      style={{
                        background: `${getDifficultyColor(question.difficulty)}20`,
                        color: getDifficultyColor(question.difficulty),
                        border: `1px solid ${getDifficultyColor(question.difficulty)}40`,
                      }}
                    >
                      {question.difficulty}
                    </span>
                    <span
                      className="frequency-indicator"
                      style={{ color: getFrequencyColor(question.frequency) }}
                    >
                      {question.frequency}
                    </span>
                    {question.acceptance && (
                      <span className="acceptance-rate">
                        ✓ {question.acceptance}
                      </span>
                    )}
                  </div>

                  <div className="topic-tags">
                    {question.topics.map((topic, idx) => (
                      <span key={idx} className="topic-tag">
                        {topic}
                      </span>
                    ))}
                  </div>
                </div>
              ))
            ) : (
              <div className="no-results">
                <Search size={48} className="no-results-icon" />
                <p className="no-results-text">
                  No questions found matching "{searchQuery}"
                </p>
                <p className="no-results-hint">
                  Try searching by problem name, number, difficulty, or topic
                </p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default CompanyQuestionsCard;
