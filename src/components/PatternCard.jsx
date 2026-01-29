import { Tag, CheckSquare } from 'lucide-react';

const PatternCard = ({ pattern }) => {
  return (
    <div className="pattern-card card">
      <h3 className="card-title">{pattern.title}</h3>
      <p className="pattern-description">{pattern.description}</p>
      
      {pattern.useCases && (
        <div className="pattern-section">
          <h4 className="pattern-section-title">
            <Tag size={16} />
            Use Cases
          </h4>
          <ul className="pattern-list">
            {pattern.useCases.map((useCase, idx) => (
              <li key={idx}>{useCase}</li>
            ))}
          </ul>
        </div>
      )}
      
      {pattern.examples && (
        <div className="pattern-section">
          <h4 className="pattern-section-title">
            <CheckSquare size={16} />
            Example Problems
          </h4>
          <div className="example-tags">
            {pattern.examples.map((example, idx) => (
              <span key={idx} className="example-tag">{example}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default PatternCard;