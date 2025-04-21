import React, { useState } from 'react';
import { MessageSquare, Upload, BarChart2, PieChart, Clock } from 'lucide-react';

const WhatsAppChatAnalyzer = () => {
  const [uploadStatus, setUploadStatus] = useState("");

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    setUploadStatus("Uploading...");

    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      console.log("✅ Backend Response:", data);
      setUploadStatus("Upload successful! Check console for results.");
    } catch (error) {
      console.error("❌ Upload failed:", error);
      setUploadStatus("Upload failed. Check console.");
    }
  };

  return (
    <div className="flex flex-col items-center w-full max-w-5xl mx-auto p-6">
      <div className="flex justify-center mb-6">
        <div className="bg-green-500 p-4 rounded-full">
          <MessageSquare size={48} color="white" />
        </div>
      </div>

      <h1 className="text-3xl font-bold text-green-600 mb-4">WhatsApp Chat Analyzer</h1>
      <p className="text-lg text-gray-700 mb-6 max-w-2xl text-center">
        Upload a WhatsApp chat file (.txt) to begin your analysis.
      </p>

      <input type="file" accept=".txt" onChange={handleFileUpload} className="mb-4" />
      <p className="text-sm text-gray-600 mb-8">{uploadStatus}</p>

      {/* Keep your feature section below */}
      {/* ... existing How to Use, Features etc. remain unchanged ... */}
    </div>
  );
};

export default WhatsAppChatAnalyzer;
