import { useState } from "react";
import { Copy, Check } from "lucide-react";

const PatternsCodeBlock = ({ code, title, description }) => {
  const [activeLanguage, setActiveLanguage] = useState("javascript");
  const [copied, setCopied] = useState(false);

  const languages = [
    { id: "javascript", name: "JavaScript", color: "#f7df1e" },
    { id: "python", name: "Python", color: "#3776ab" },
    { id: "java", name: "Java", color: "#007396" },
    { id: "cpp", name: "C++", color: "#00599c" },
  ];

  const handleCopy = () => {
    navigator.clipboard.writeText(code[activeLanguage]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="patterns-code-block">
      {title && (
        <div className="code-block-header-info">
          <h4 className="code-block-title">{title}</h4>
          {description && (
            <p className="code-block-description">{description}</p>
          )}
        </div>
      )}
      <div className="code-block-editor">
        <div className="code-editor-header">
          <div className="language-selector">
            {languages.map((lang) => (
              <button
                key={lang.id}
                className={`lang-btn ${activeLanguage === lang.id ? "active" : ""}`}
                onClick={() => setActiveLanguage(lang.id)}
                style={
                  activeLanguage === lang.id
                    ? { borderBottomColor: lang.color }
                    : {}
                }
              >
                <span
                  className="lang-dot"
                  style={{ background: lang.color }}
                ></span>
                {lang.name}
              </button>
            ))}
          </div>
          <button className="code-copy-btn" onClick={handleCopy}>
            {copied ? <Check size={16} /> : <Copy size={16} />}
            <span>{copied ? "Copied!" : "Copy"}</span>
          </button>
        </div>
        <pre className="code-editor-content">
          <code className={`language-${activeLanguage}`}>
            {code[activeLanguage]}
          </code>
        </pre>
      </div>
    </div>
  );
};

export default PatternsCodeBlock;
