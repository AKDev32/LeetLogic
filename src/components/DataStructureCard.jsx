import { useState } from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
import CodeBlock from './CodeBlock';

const DataStructureCard = ({ dataStructure }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="data-structure-card card">
      <div className="card-header" onClick={() => setExpanded(!expanded)}>
        <div>
          <h3 className="card-title">{dataStructure.title}</h3>
          {dataStructure.complexity && (
            <div className="complexity-badges">
              {Object.entries(dataStructure.complexity).map(([key, value]) => (
                <span key={key} className="complexity-badge">
                  <span className="complexity-key">{key}:</span> {value}
                </span>
              ))}
            </div>
          )}
        </div>
        <button className="expand-btn">
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>
      
      {expanded && (
        <div className="card-content">
          {dataStructure.techniques && (
            <div className="techniques">
              <h4 className="content-subtitle">Common Techniques:</h4>
              <div className="technique-tags">
                {dataStructure.techniques.map((tech, idx) => (
                  <span key={idx} className="technique-tag">{tech}</span>
                ))}
              </div>
            </div>
          )}
          
          {dataStructure.code && (
            <div className="code-section">
              <CodeBlock code={dataStructure.code} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DataStructureCard;