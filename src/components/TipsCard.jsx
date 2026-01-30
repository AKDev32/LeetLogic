import { useState } from "react";
import { ChevronDown, ChevronUp, Square, CheckSquare } from "lucide-react";

const TipsCard = ({ section }) => {
  const [expanded, setExpanded] = useState(false);
  const [checkedItems, setCheckedItems] = useState(new Set());

  const toggleCheck = (idx) => {
    const newChecked = new Set(checkedItems);
    if (newChecked.has(idx)) {
      newChecked.delete(idx);
    } else {
      newChecked.add(idx);
    }
    setCheckedItems(newChecked);
  };

  const checkedCount = checkedItems.size;
  const totalCount = section.items.length;
  const progressPercentage = (checkedCount / totalCount) * 100;

  return (
    <div className="tips-checklist-card">
      <div
        className="tips-checklist-header"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="tips-checklist-info">
          <h3 className="tips-checklist-title">{section.title}</h3>
          <div className="tips-checklist-meta">
            <span className="tips-checklist-count">
              {checkedCount}/{totalCount} completed
            </span>
            <div className="tips-progress-bar">
              <div
                className="tips-progress-fill"
                style={{ width: `${progressPercentage}%` }}
              />
            </div>
          </div>
        </div>
        <button className="tips-checklist-expand">
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </button>
      </div>

      {expanded && (
        <div className="tips-checklist-content">
          <ul className="tips-checklist-items">
            {section.items.map((item, idx) => {
              const isChecked = checkedItems.has(idx);
              return (
                <li
                  key={idx}
                  className={`tips-checklist-item ${isChecked ? "checked" : ""}`}
                  onClick={() => toggleCheck(idx)}
                >
                  <div className="tips-checkbox">
                    {isChecked ? (
                      <CheckSquare
                        size={20}
                        className="checkbox-icon checked"
                      />
                    ) : (
                      <Square size={20} className="checkbox-icon" />
                    )}
                  </div>
                  <span className="tips-checklist-text">{item}</span>
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
};

export default TipsCard;
