import { useState } from 'react';
import { ChevronDown, ChevronUp, Clock, Database, Zap, TrendingUp, AlertCircle } from 'lucide-react';

const ComplexityCard = ({ complexity }) => {
  const [expanded, setExpanded] = useState(false);
  const [showCode, setShowCode] = useState(false);

  const getIcon = () => {
    switch(complexity.notation) {
      case 'O(1)': return Zap;
      case 'O(log n)': 
      case 'O(n log n)': return TrendingUp;
      case 'O(n²)':
      case 'O(2ⁿ)':
      case 'O(n!)': return AlertCircle;
      default: return Clock;
    }
  };

  const Icon = getIcon();

  return (
    <div className="complexity-card-modern">
      <div 
        className="complexity-card-header"
        style={{ borderLeftColor: complexity.color }}
        onClick={() => setExpanded(!expanded)}
      >
        <div className="complexity-header-left">
          <div className="complexity-icon-wrapper" style={{ background: `${complexity.color}20` }}>
            <Icon size={20} style={{ color: complexity.color }} />
          </div>
          <div className="complexity-title-section">
            <span className="complexity-notation" style={{ color: complexity.color }}>
              {complexity.notation}
            </span>
            <span className="complexity-name">{complexity.name}</span>
          </div>
        </div>
        <button className="complexity-expand-btn">
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>

      {expanded && (
        <div className="complexity-card-content">
          <p className="complexity-description">{complexity.description}</p>

          {/* Stats Section */}
          {(complexity.maxN || complexity.operations) && (
            <div className="complexity-stats">
              {complexity.maxN && (
                <div className="complexity-stat">
                  <Database size={16} />
                  <span>Max Input: <strong>{complexity.maxN}</strong></span>
                </div>
              )}
              {complexity.operations && (
                <div className="complexity-stat">
                  <Clock size={16} />
                  <span>Operations: <strong>{complexity.operations}</strong></span>
                </div>
              )}
            </div>
          )}

          {/* Examples */}
          <div className="complexity-examples-section">
            <h4 className="complexity-section-title">Common Examples</h4>
            <div className="complexity-examples-grid">
              {complexity.examples.map((example, idx) => (
                <div key={idx} className="complexity-example-item">
                  <span className="example-bullet">→</span>
                  <span>{example}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Code Example */}
          {complexity.code && (
            <div className="complexity-code-section">
              <div className="code-section-header">
                <h4 className="complexity-section-title">Code Example</h4>
                <button 
                  className="toggle-code-btn"
                  onClick={() => setShowCode(!showCode)}
                >
                  {showCode ? 'Hide' : 'Show'} Code
                </button>
              </div>
              
              {showCode && (
                <div className="complexity-code-block">
                  <pre><code>{complexity.code}</code></pre>
                </div>
              )}
            </div>
          )}

          {/* Performance Badge */}
          <div className="complexity-performance">
            <span className="performance-label">Performance:</span>
            <span 
              className="performance-badge"
              style={{ 
                background: `${complexity.color}20`,
                color: complexity.color,
                borderColor: `${complexity.color}40`
              }}
            >
              {complexity.performance || getPerformanceText(complexity.notation)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

const getPerformanceText = (notation) => {
  const perfMap = {
    'O(1)': 'Excellent',
    'O(log n)': 'Great',
    'O(n)': 'Good',
    'O(n log n)': 'Fair',
    'O(n²)': 'Moderate',
    'O(2ⁿ)': 'Poor',
    'O(n!)': 'Very Poor'
  };
  return perfMap[notation] || 'Variable';
};

export default ComplexityCard;