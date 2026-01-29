import { useState } from 'react';
import { Copy, Check } from 'lucide-react';

const CodeBlock = ({ code }) => {
  const [activeLanguage, setActiveLanguage] = useState('javascript');
  const [copied, setCopied] = useState(false);
  
  const languages = Object.keys(code);

  const handleCopy = () => {
    navigator.clipboard.writeText(code[activeLanguage]);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="code-block">
      <div className="code-header">
        <div className="language-tabs">
          {languages.map(lang => (
            <button
              key={lang}
              className={`language-tab ${activeLanguage === lang ? 'active' : ''}`}
              onClick={() => setActiveLanguage(lang)}
            >
              {lang}
            </button>
          ))}
        </div>
        <button className="copy-btn" onClick={handleCopy}>
          {copied ? <Check size={16} /> : <Copy size={16} />}
          <span>{copied ? 'Copied!' : 'Copy'}</span>
        </button>
      </div>
      <pre className="code-content">
        <code>{code[activeLanguage]}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;