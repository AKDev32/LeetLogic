import { CheckCircle } from 'lucide-react';

const TipsCard = ({ section }) => {
  return (
    <div className="tips-card card">
      <h3 className="card-title">{section.title}</h3>
      <ul className="tips-list">
        {section.items.map((item, idx) => (
          <li key={idx} className="tip-item">
            <CheckCircle size={18} className="tip-icon" />
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
};

export default TipsCard;