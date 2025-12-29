import { useState, useEffect } from 'react'
import axios from 'axios'
import { Activity, FileText, TrendingUp, AlertCircle, CheckCircle, Loader } from 'lucide-react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [reportText, setReportText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [systemStatus, setSystemStatus] = useState(null)

  useEffect(() => {
    checkSystemHealth()
  }, [])

  const checkSystemHealth = async () => {
    try {
      const response = await axios.get(`${API_URL}/api/health`)
      setSystemStatus(response.data)
    } catch (err) {
      console.error('System health check failed:', err)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!reportText.trim()) {
      setError('Please enter a medical report')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await axios.post(`${API_URL}/api/code`, {
        report_text: reportText
      })

      if (response.data.success) {
        setResult(response.data)
      } else {
        setError(response.data.error || 'Coding failed')
      }
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to connect to server')
    } finally {
      setLoading(false)
    }
  }

  const handleClear = () => {
    setReportText('')
    setResult(null)
    setError(null)
  }

  const getCategoryColor = (category) => {
    switch (category) {
      case 'primary_diagnosis':
        return '#e74c3c'
      case 'secondary_diagnosis':
        return '#f39c12'
      case 'symptom':
        return '#3498db'
      case 'procedure':
        return '#27ae60'
      case 'complication':
        return '#8e44ad'
      default:
        return '#95a5a6'
    }
  }

  const getCategoryLabel = (category) => {
    switch (category) {
      case 'primary_diagnosis':
        return 'Primary Diagnosis'
      case 'secondary_diagnosis':
        return 'Secondary Diagnosis'
      case 'symptom':
        return 'Symptom'
      case 'procedure':
        return 'Procedure'
      case 'complication':
        return 'Complication'
      default:
        return 'Diagnosis'
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="logo">
            <Activity size={32} />
            <h1>AI Medical Coding System</h1>
          </div>
          <div className="badge">Hybrid RAG + Gemini AI</div>
        </div>
        {systemStatus && (
          <div className="system-status">
            <CheckCircle size={16} />
            <span>{systemStatus.total_codes?.toLocaleString()} ICD-10 Codes Loaded</span>
          </div>
        )}
      </header>

      <main className="main">
        <div className="container">
          <div className="input-section">
            <div className="section-header">
              <FileText size={24} />
              <h2>Medical Report</h2>
            </div>
            
            <form onSubmit={handleSubmit}>
              <textarea
                className="report-input"
                placeholder="Enter medical report here...&#10;&#10;Example:&#10;PATIENT GOT ADMITTED WITH C/O-EXCERTIONAL DYSPNEA SINCE 2-3 DAYS ASSOCIATED WITH COLD, COUGH+, MILD FEVERISH SINCE 1 DAY. DIAGNOSED AS LOWER RESPIRATORY TRACT INFECTION..."
                value={reportText}
                onChange={(e) => setReportText(e.target.value)}
                rows={12}
              />

              <div className="button-group">
                <button 
                  type="submit" 
                  className="btn btn-primary"
                  disabled={loading || !reportText.trim()}
                >
                  {loading ? (
                    <>
                      <Loader className="spinner" size={20} />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <TrendingUp size={20} />
                      Generate Codes
                    </>
                  )}
                </button>
                
                <button 
                  type="button" 
                  className="btn btn-secondary"
                  onClick={handleClear}
                  disabled={loading}
                >
                  Clear
                </button>
              </div>
            </form>

            {error && (
              <div className="alert alert-error">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}
          </div>

          {result && (
            <div className="results-section">
              <div className="section-header">
                <CheckCircle size={24} />
                <h2>Coding Results</h2>
              </div>

              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-value">{result.total_codes}</div>
                  <div className="stat-label">Total Codes</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{result.avg_confidence}%</div>
                  <div className="stat-label">Avg Confidence</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{result.extracted_statements.length}</div>
                  <div className="stat-label">Statements</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">95%</div>
                  <div className="stat-label">Target Accuracy</div>
                </div>
              </div>

              {result.extracted_statements.length > 0 && (
                <div className="statements-box">
                  <h3>Extracted Clinical Statements:</h3>
                  <ul>
                    {result.extracted_statements.map((stmt, idx) => (
                      <li key={idx}>{stmt}</li>
                    ))}
                  </ul>
                </div>
              )}

              <div className="codes-list">
                {result.codes.map((code, idx) => (
                  <div key={idx} className="code-card">
                    <div className="code-header">
                      <div className="code-number" style={{ backgroundColor: getCategoryColor(code.category) }}>
                        {code.code}
                      </div>
                      <div className="code-category">
                        {getCategoryLabel(code.category)}
                      </div>
                      <div className="confidence-badge">
                        {code.confidence}%
                      </div>
                    </div>
                    <div className="code-description">{code.description}</div>
                    <div className="code-source">
                      <span className="source-label">Source:</span> "{code.source}"
                    </div>
                  </div>
                ))}
              </div>

              {result.codes.length === 0 && (
                <div className="alert alert-info">
                  <AlertCircle size={20} />
                  <span>No codes found with sufficient confidence (&gt;60%)</span>
                </div>
              )}
            </div>
          )}
        </div>
      </main>

      <footer className="footer">
        <p>Comprehensive AI Medical Coding | Hybrid RAG + Gemini AI | All Medical Conditions Treated Equally | Zero Bias</p>
      </footer>
    </div>
  )
}

export default App
