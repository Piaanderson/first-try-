import { useState, useRef } from 'react'
import './App.css'

function UploadPanel({ id, label, badge, accept, description, onFile, file, processing, result, resultLabel }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped) onFile(dropped)
  }

  function handleChange(e) {
    const selected = e.target.files[0]
    if (selected) onFile(selected)
  }

  return (
    <div className={`panel ${dragging ? 'dragging' : ''}`}>
      <div className="panel-header">
        <span className="badge">{badge}</span>
        <h2>{label}</h2>
        <p className="panel-desc">{description}</p>
      </div>

      <div
        className="drop-zone"
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
          onChange={handleChange}
          style={{ display: 'none' }}
        />
        {file ? (
          <div className="file-info">
            <span className="file-icon">📄</span>
            <span className="file-name">{file.name}</span>
            <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
          </div>
        ) : (
          <div className="drop-prompt">
            <span className="drop-icon">↑</span>
            <span>Drop file here or click to browse</span>
            <span className="drop-hint">Accepts: {accept}</span>
          </div>
        )}
      </div>

      {file && (
        <button
          className="process-btn"
          onClick={() => onFile(file, true)}
          disabled={processing}
        >
          {processing ? 'Processing…' : 'Apply Effect'}
        </button>
      )}

      {result && (
        <div className="result-area">
          <div className="result-label">{resultLabel}</div>
          <div className="result-preview">{result}</div>
        </div>
      )}
    </div>
  )
}

// Stub: replace with real xenonite processing logic when algorithm is ready
async function processXenonite(file) {
  await new Promise(r => setTimeout(r, 1200))
  return `[Xenonite pattern output for "${file.name}" will appear here]`
}

// Stub: replace with real texture processing logic when steps are provided
async function processTexture(file) {
  await new Promise(r => setTimeout(r, 1200))
  return `[Textured display output for "${file.name}" will appear here]`
}

export default function App() {
  const [modelFile, setModelFile] = useState(null)
  const [modelProcessing, setModelProcessing] = useState(false)
  const [modelResult, setModelResult] = useState(null)

  const [imageFile, setImageFile] = useState(null)
  const [imageProcessing, setImageProcessing] = useState(false)
  const [imageResult, setImageResult] = useState(null)

  async function handleModel(file, run = false) {
    setModelFile(file)
    if (!run) return
    setModelProcessing(true)
    setModelResult(null)
    const out = await processXenonite(file)
    setModelResult(out)
    setModelProcessing(false)
  }

  async function handleImage(file, run = false) {
    setImageFile(file)
    if (!run) return
    setImageProcessing(true)
    setImageResult(null)
    const out = await processTexture(file)
    setImageResult(out)
    setImageProcessing(false)
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-inner">
          <h1 className="site-title">Effect Studio</h1>
          <p className="site-subtitle">Transform your files with generative effects</p>
        </div>
      </header>

      <main className="panels">
        <UploadPanel
          id="xenonite"
          badge="Effect A"
          label="Xenonite Pattern"
          accept=".obj,.stl,.glb,.gltf,.ply,.fbx"
          description="Upload a 3D model and convert it into a xenonite pattern. Supports OBJ, STL, GLB, GLTF, PLY, and FBX files."
          onFile={handleModel}
          file={modelFile}
          processing={modelProcessing}
          result={modelResult}
          resultLabel="Xenonite Output"
        />

        <div className="divider" />

        <UploadPanel
          id="texture"
          badge="Effect B"
          label="Rocky's Texture"
          accept="image/*"
          description="Upload an image and apply Rocky's signature textured display effect. Accepts PNG, JPG, WEBP, and other image formats."
          onFile={handleImage}
          file={imageFile}
          processing={imageProcessing}
          result={imageResult}
          resultLabel="Textured Output"
        />
      </main>

      <footer className="app-footer">
        <span>Effect Studio — more effects coming soon</span>
      </footer>
    </div>
  )
}
