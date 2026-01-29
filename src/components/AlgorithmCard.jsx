import { useState } from 'react';
import { ChevronDown, ChevronUp, Clock, HardDrive, CheckCircle, XCircle } from 'lucide-react';
import CodeBlock from './CodeBlock';

const AlgorithmCard = ({ section }) => {
  const [expandedItems, setExpandedItems] = useState({});

  const toggleItem = (idx) => {
    setExpandedItems(prev => ({
      ...prev,
      [idx]: !prev[idx]
    }));
  };

  return (
    <div className="algorithm-section">
      <h3 className="algorithm-section-title">{section.title}</h3>
      <div className="algorithm-items">
        {section.items.map((item, idx) => (
          <div key={idx} className="algorithm-item card">
            <div className="card-header" onClick={() => toggleItem(idx)}>
              <div>
                <h4 className="algorithm-name">{item.name}</h4>
                <div className="algorithm-meta">
                  <span className="meta-item">
                    <Clock size={14} />
                    <span>Time: {item.time}</span>
                  </span>
                  <span className="meta-item">
                    <HardDrive size={14} />
                    <span>Space: {item.space}</span>
                  </span>
                  {item.stable !== undefined && (
                    <span className="meta-item">
                      {item.stable ? (
                        <CheckCircle size={14} className="stable-icon" />
                      ) : (
                        <XCircle size={14} className="unstable-icon" />
                      )}
                      <span>{item.stable ? 'Stable' : 'Unstable'}</span>
                    </span>
                  )}
                </div>
              </div>
              <button className="expand-btn">
                {expandedItems[idx] ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
              </button>
            </div>
            
            {expandedItems[idx] && item.code && (
              <div className="card-content">
                <CodeBlock code={item.code} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AlgorithmCard;