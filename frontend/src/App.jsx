import './App.css'

function App() {
  return (
    <div className="container">
      <header>
        <h1>Video Transcript Generator</h1>
        <p>Generate transcripts, translate, ask questions and create clips from any video stream.</p>
      </header>
      <section className="app-frame">
        <iframe
          src="http://localhost:8501"
          title="Video Transcript App"
          width="100%"
          height="800"
        />
      </section>
    </div>
  )
}

export default App
