import React from 'react';

const ComplexityCard = ({ section }) => {
  return (
    <div className="complexity-card card">
      <h3 className="card-title">{section.title}</h3>
      <div className="complexity-items">
        {section.items.map((item, idx) => (
          <div key={idx} className="complexity-item">
            <div className="complexity-header">
              <span 
                className="complexity-notation"
                style={{ color: item.color || '#8b5cf6' }}
              >
                {item.name}
              </span>
              <span className="complexity-desc">{item.description}</span>
            </div>
            <div className="complexity-examples">
              {item.examples.map((example, i) => (
                <span key={i} className="complexity-example">
                  {example}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ComplexityCard;