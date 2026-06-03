import { useState, useRef, useCallback } from 'react'
import ModelViewer from './components/ModelViewer'
import './App.css'

const API = 'http://localhost:8000'

function DropZone({ accept, onFile, file }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f) onFile(f)
  }

  return (
    <div
      className={`drop-zone ${dragging ? 'dragging' : ''}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && inputRef.current.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
        style={{ display: 'none' }}
      />
      {file ? (
        <div className="file-info">
          <span className="file-icon">📄</span>
          <span className="file-name">{file.name}</span>
          <span className="file-size">{(file.size / 1024).toFixed(0)} KB</span>
          <span className="drop-hint">Click or drop to replace</span>
        </div>
      ) : (
        <div className="drop-prompt">
          <span className="drop-icon">↑</span>
          <span>Drop file here or click to browse</span>
          <span className="drop-hint">Accepts: {accept}</span>
        </div>
      )}
    </div>
  )
}

function XenonitePanel() {
  const [file, setFile] = useState(null)
  const [originalUrl, setOriginalUrl] = useState(null)
  const [resultUrl, setResultUrl] = useState(null)
  const [resultBlob, setResultBlob] = useState(null)
  const [status, setStatus] = useState('idle') // idle | processing | done | error
  const [error, setError] = useState(null)

  // Density slider: maps to seed_ratio 0.005 – 0.04
  const [density, setDensity] = useState(50)
  const seedRatio = (0.005 + (density / 100) * 0.035).toFixed(4)

  function handleFile(f) {
    setFile(f)
    setResultUrl(null)
    setResultBlob(null)
    setStatus('idle')
    setError(null)
    const url = URL.createObjectURL(f)
    setOriginalUrl(url)
  }

  async function handleProcess() {
    if (!file) return
    setStatus('processing')
    setResultUrl(null)
    setError(null)

    const form = new FormData()
    form.append('file', file)
    form.append('seed_ratio', seedRatio)

    try {
      const res = await fetch(`${API}/xenonite`, { method: 'POST', body: form })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Server error ${res.status}`)
      }
      const blob = await res.blob()
      setResultBlob(blob)
      setResultUrl(URL.createObjectURL(blob))
      setStatus('done')
    } catch (e) {
      setError(e.message)
      setStatus('error')
    }
  }

  function handleDownload() {
    if (!resultBlob) return
    const a = document.createElement('a')
    a.href = URL.createObjectURL(resultBlob)
    a.download = file.name.replace(/\.[^.]+$/, '') + '_xenonite.stl'
    a.click()
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span className="badge">Effect A</span>
        <h2>Xenonite Pattern</h2>
        <p className="panel-desc">
          Upload a 3D model (.stl, .obj) and convert it into a hollow xenonite
          lattice structure — geometric poles and circular nodes following the
          surface of your mesh.
        </p>
      </div>

      <DropZone
        accept=".stl,.obj,.glb,.gltf,.ply"
        onFile={handleFile}
        file={file}
      />

      {file && (
        <div className="controls">
          <label className="slider-label">
            <span>Pattern density</span>
            <span className="slider-value">{density}%</span>
          </label>
          <input
            type="range"
            min={10}
            max={100}
            value={density}
            onChange={(e) => setDensity(Number(e.target.value))}
            className="slider"
          />

          <button
            className="process-btn"
            onClick={handleProcess}
            disabled={status === 'processing'}
          >
            {status === 'processing' ? 'Generating…' : 'Generate Xenonite'}
          </button>
        </div>
      )}

      {error && <div className="error-box">{error}</div>}

      <div className={`viewers ${originalUrl ? 'has-viewers' : ''}`}>
        {originalUrl && (
          <ModelViewer url={originalUrl} label="Original" color="#8888aa" />
        )}
        {resultUrl && (
          <ModelViewer url={resultUrl} label="Xenonite" color="#b8a0ff" />
        )}
      </div>

      {resultUrl && (
        <button className="download-btn" onClick={handleDownload}>
          ↓ Download xenonite.stl
        </button>
      )}
    </div>
  )
}

function TexturePanel() {
  return (
    <div className="panel panel-texture">
      <div className="panel-header">
        <span className="badge badge-b">Effect B</span>
        <h2>Rocky's Texture</h2>
        <p className="panel-desc">
          Upload an image and apply Rocky's signature textured display effect.
          Algorithm steps coming soon.
        </p>
      </div>
      <div className="coming-soon">
        <span className="coming-icon">🔜</span>
        <span>Effect steps in progress</span>
      </div>
    </div>
  )
}

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1 className="site-title">Effect Studio</h1>
        <p className="site-subtitle">Transform your files with generative effects</p>
      </header>

      <main className="panels">
        <XenonitePanel />
        <div className="divider" />
        <TexturePanel />
      </main>

      <footer className="app-footer">
        Effect Studio — more effects coming soon
      </footer>
    </div>
  )
}
