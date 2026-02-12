import React, { useState } from 'react';
import axios from 'axios';
import { Shield, Upload, CheckCircle, AlertCircle } from 'lucide-react';

export default function App() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append('video', file);

    try {
      // Connects to your Node.js Bridge (Port 5000)
      const res = await axios.post('http://localhost:5001/api/detect', formData);
      setResult(res.data);
    } catch (err) {
      alert("Backend Bridge not connected. Start Node.js first!");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col items-center justify-center p-6">
      <div className="flex items-center gap-3 mb-10">
        <Shield className="w-10 h-10 text-blue-500" />
        <h1 className="text-4xl font-black tracking-tight uppercase">Deepfake Guardian</h1>
      </div>

      <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl shadow-2xl w-full max-w-lg">
        <div className="border-2 border-dashed border-slate-700 rounded-xl p-10 text-center hover:border-blue-500 transition-colors">
          <input type="file" onChange={(e) => setFile(e.target.files[0])} className="hidden" id="video-upload" />
          <label htmlFor="video-upload" className="cursor-pointer">
            <Upload className="w-12 h-12 mx-auto mb-4 text-slate-500" />
            <p className="text-slate-400">{file ? file.name : "Select video to scan"}</p>
          </label>
        </div>

        <button onClick={handleUpload} disabled={loading} className="w-full mt-8 bg-blue-600 hover:bg-blue-500 py-4 rounded-xl font-bold uppercase tracking-widest transition-all active:scale-95">
          {loading ? "AI is Analyzing..." : "Verify Content"}
        </button>
      </div>

      {result && (
        <div className={`mt-8 p-6 rounded-2xl w-full max-w-lg flex items-center gap-4 border ${result.label === 'FAKE' ? 'bg-red-950/30 border-red-500/50' : 'bg-emerald-950/30 border-emerald-500/50'}`}>
          {result.label === 'FAKE' ? <AlertCircle className="text-red-500 w-8 h-8" /> : <CheckCircle className="text-emerald-500 w-8 h-8" />}
          <div>
            <p className="text-sm uppercase font-bold text-slate-400">Analysis Result</p>
            <h2 className={`text-3xl font-black ${result.label === 'FAKE' ? 'text-red-500' : 'text-emerald-500'}`}>{result.label} ({result.confidence}%)</h2>
          </div>
        </div>
      )}
    </div>
  );
}