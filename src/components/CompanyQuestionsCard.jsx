import { useState } from 'react';
import { ChevronDown, ChevronUp, ExternalLink, TrendingUp, Building2 } from 'lucide-react';

const CompanyQuestionsCard = ({ company }) => {
  const [expanded, setExpanded] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState(0);

  const getDifficultyColor = (difficulty) => {
    switch (difficulty) {
      case 'Easy': return '#10b981';
      case 'Medium': return '#f59e0b';
      case 'Hard': return '#ef4444';
      default: return '#9ca3af';
    }
  };

  const getFrequencyColor = (frequency) => {
    switch (frequency) {
      case 'Very High': return '#ef4444';
      case 'High': return '#f59e0b';
      case 'Medium': return '#3b82f6';
      default: return '#9ca3af';
    }
  };

  return (
    <div className="company-card card">
      <div className="card-header" onClick={() => setExpanded(!expanded)}>
        <div className="company-header-content">
          <div className="company-logo-wrapper">
            <span className="company-logo-emoji">{company.logo}</span>
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
                    border: `1px solid ${getFrequencyColor(company.frequency)}40`
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
          {/* Category Tabs */}
          <div className="category-tabs">
            {company.categories.map((cat, idx) => (
              <button
                key={idx}
                className={`category-tab ${selectedCategory === idx ? 'active' : ''}`}
                onClick={() => setSelectedCategory(idx)}
              >
                {cat.category}
                <span className="question-count">{cat.questions.length}</span>
              </button>
            ))}
          </div>

          {/* Questions List */}
          <div className="questions-list">
            {company.categories[selectedCategory].questions.map((question) => (
              <div key={question.id} className="question-item">
                <div className="question-header">
                  <div className="question-title-section">
                    <span className="leetcode-num">#{question.leetcodeNum}</span>
                    <span className="question-name">{question.name}</span>
                  </div>
                  <a
                    href={`https://leetcode.com/problems/${question.name.toLowerCase().replace(/\s+/g, '-')}/`}
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
                      border: `1px solid ${getDifficultyColor(question.difficulty)}40`
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
                </div>

                <div className="topic-tags">
                  {question.topics.map((topic, idx) => (
                    <span key={idx} className="topic-tag">{topic}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default CompanyQuestionsCard;