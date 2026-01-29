import { useState } from 'react';
import { Activity, Database, Code2, Lightbulb, Brain, Menu, X, Github, ExternalLink } from 'lucide-react';
import { cheatsheetData } from './data/cheatsheetData';
import ComplexityCard from './components/ComplexityCard';
import DataStructureCard from './components/DataStructureCard';
import AlgorithmCard from './components/AlgorithmCard';
import PatternCard from './components/PatternCard';
import TipsCard from './components/TipsCard';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('bigO');
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const tabs = [
    { id: 'bigO', label: 'Big O', icon: Activity },
    { id: 'dataStructures', label: 'Data Structures', icon: Database },
    { id: 'algorithms', label: 'Algorithms', icon: Code2 },
    { id: 'patterns', label: 'Patterns', icon: Lightbulb },
    { id: 'tips', label: 'Tips', icon: Brain }
  ];

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <button 
              className="menu-toggle"
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              {sidebarOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
            <h1 className="logo">
              <span className="logo-bracket">&lt;</span>
              LeetLogic
              <span className="logo-bracket">/&gt;</span>
            </h1>
          </div>
          <div className="header-right">
            <a 
              href="https://github.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="header-link"
            >
              <Github size={20} />
              <span>GitHub</span>
            </a>
            <a 
              href="https://leetcode.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="header-link"
            >
              <ExternalLink size={20} />
              <span>LeetCode</span>
            </a>
          </div>
        </div>
      </header>

      <div className="main-container">
        <aside className={`sidebar ${sidebarOpen ? 'open' : 'closed'}`}>
          <nav className="sidebar-nav">
            {tabs.map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  className={`sidebar-item ${activeTab === tab.id ? 'active' : ''}`}
                  onClick={() => setActiveTab(tab.id)}
                >
                  <Icon size={20} />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>
        </aside>

        <main className="content">
          <div className="content-inner">
            {activeTab === 'bigO' && (
              <section className="section">
                <div className="section-header">
                  <Activity className="section-icon" size={32} />
                  <div>
                    <h2 className="section-title">{cheatsheetData.bigO.title}</h2>
                    <p className="section-subtitle">Time and space complexity analysis</p>
                  </div>
                </div>
                <div className="grid">
                  {cheatsheetData.bigO.sections.map((section, idx) => (
                    <ComplexityCard key={idx} section={section} />
                  ))}
                </div>
              </section>
            )}

            {activeTab === 'dataStructures' && (
              <section className="section">
                <div className="section-header">
                  <Database className="section-icon" size={32} />
                  <div>
                    <h2 className="section-title">{cheatsheetData.dataStructures.title}</h2>
                    <p className="section-subtitle">Common data structures and their implementations</p>
                  </div>
                </div>
                <div className="data-structures-grid">
                  {cheatsheetData.dataStructures.sections.map((ds, idx) => (
                    <DataStructureCard key={idx} dataStructure={ds} />
                  ))}
                </div>
              </section>
            )}

            {activeTab === 'algorithms' && (
              <section className="section">
                <div className="section-header">
                  <Code2 className="section-icon" size={32} />
                  <div>
                    <h2 className="section-title">{cheatsheetData.algorithms.title}</h2>
                    <p className="section-subtitle">Essential algorithms with code examples</p>
                  </div>
                </div>
                <div className="algorithms-container">
                  {cheatsheetData.algorithms.sections.map((section, idx) => (
                    <AlgorithmCard key={idx} section={section} />
                  ))}
                </div>
              </section>
            )}

            {activeTab === 'patterns' && (
              <section className="section">
                <div className="section-header">
                  <Lightbulb className="section-icon" size={32} />
                  <div>
                    <h2 className="section-title">{cheatsheetData.patterns.title}</h2>
                    <p className="section-subtitle">Problem-solving patterns and techniques</p>
                  </div>
                </div>
                <div className="patterns-grid">
                  {cheatsheetData.patterns.sections.map((pattern, idx) => (
                    <PatternCard key={idx} pattern={pattern} />
                  ))}
                </div>
              </section>
            )}

            {activeTab === 'tips' && (
              <section className="section">
                <div className="section-header">
                  <Brain className="section-icon" size={32} />
                  <div>
                    <h2 className="section-title">{cheatsheetData.tips.title}</h2>
                    <p className="section-subtitle">Strategies for tackling coding interviews</p>
                  </div>
                </div>
                <div className="tips-grid">
                  {cheatsheetData.tips.sections.map((section, idx) => (
                    <TipsCard key={idx} section={section} />
                  ))}
                </div>
              </section>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;