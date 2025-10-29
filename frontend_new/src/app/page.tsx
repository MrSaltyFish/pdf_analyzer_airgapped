"use client";

import React, { useState } from "react";
import axios from "axios";

export default function RAGChatPage() {
  const [messages, setMessages] = useState<{ role: string; text: string }[]>(
    []
  );
  const [input, setInput] = useState("");
  const [persona, setPersona] = useState("General user");
  const [uploading, setUploading] = useState(false);
  const [loading, setLoading] = useState(false);

  // === File Upload ===
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post(
        "http://localhost:8000/api/v1/upload/single-file",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      console.log("✅ File uploaded:", res.data);
      alert("✅ File uploaded successfully!");
    } catch (err) {
      console.error("❌ Upload failed:", err);
      alert("❌ Upload failed. Check console.");
    } finally {
      setUploading(false);
    }
  };

  // === Send Chat Message ===
  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMsg = { role: "user", text: input };
    setMessages((prev) => [...prev, newMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:8000/api/v1/query", {
        query: input,
        persona: persona,
        top_k: 5,
      });

      const botMsg = {
        role: "assistant",
        text: res.data.final_answer || JSON.stringify(res.data),
      };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "⚠️ Error fetching response." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendFAISSDelete = async () => {
    const res = axios.delete(
      "http://localhost:8000/api/v1/vector-db/delete-all"
    );
  };

  return (
    <main className="flex flex-col h-screen max-w-3xl mx-auto p-6 bg-gray-50">
      <h1 className="text-2xl font-bold mb-4 text-gray-800">
        📄 RAG Chat Assistant
      </h1>

      {/* Upload Section */}
      <div className="flex items-center mb-3">
        <input
          type="file"
          accept=".pdf"
          onChange={handleFileUpload}
          className="block w-full text-sm text-gray-700"
        />
        {uploading && (
          <span className="ml-3 text-blue-600 text-sm animate-pulse">
            Uploading...
          </span>
        )}
      </div>

      {/* Persona Input */}
      <input
        type="text"
        value={persona}
        onChange={(e) => setPersona(e.target.value)}
        placeholder="Persona (default: General user)"
        className="border border-gray-300 rounded-lg p-2 mb-4 w-full"
      />

      {/* Chat Window */}
      <div className="flex-1 overflow-y-auto border rounded-lg p-4 bg-white shadow-sm">
        {messages.length === 0 && (
          <div className="text-gray-500 text-center mt-10">
            Start chatting with your RAG Assistant ✨
          </div>
        )}
        {messages.map((msg, i) => (
          <div
            key={i}
            className={`my-2 flex ${
              msg.role === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`px-4 py-2 rounded-2xl max-w-[75%] ${
                msg.role === "user"
                  ? "bg-blue-100 text-blue-800"
                  : "bg-gray-100 text-gray-800 border"
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
        {loading && (
          <div className="text-gray-400 text-sm mt-2">Thinking...</div>
        )}
      </div>

      {/* Chat Input */}
      <div className="flex mt-3">
        <input
          type="text"
          value={input}
          placeholder="Ask a question..."
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && sendMessage()}
          className="flex-1 border border-gray-300 rounded-l-lg p-2"
        />
        <button
          onClick={sendMessage}
          disabled={loading}
          className="bg-blue-600 text-white px-4 py-2 rounded-r-lg hover:bg-blue-700 transition"
        >
          Send
        </button>{" "}
        <button
          onClick={sendFAISSDelete}
          disabled={loading}
          className="bg-red-600 text-white px-4 py-2 rounded-r-lg hover:bg-blue-700 transition"
        >
          Delete FAISS
        </button>
      </div>
    </main>
  );
}
