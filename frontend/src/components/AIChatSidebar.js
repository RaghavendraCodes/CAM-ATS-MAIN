// AIChatSidebar.jsx
import React, { useState, useRef, useEffect } from "react";
import { MessageSquare, Send, X, Maximize2, Minimize2 } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

const AIChatSidebar = ({ isOpen, onClose, transcript = "", videoTitle = "" }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [sidebarWidth, setSidebarWidth] = useState(350);
  const [isHalfScreen, setIsHalfScreen] = useState(false);
  const isResizing = useRef(false);
  const messagesEndRef = useRef(null);

  // ✅ Drag to Resize (Improved)
  const handleMouseDown = () => {
    isResizing.current = true;
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  const handleMouseMove = (e) => {
    if (!isResizing.current) return;
    let newWidth = window.innerWidth - e.clientX;
    if (newWidth >= 300 && newWidth <= 700) {
      setSidebarWidth(newWidth);
      setIsHalfScreen(false); // disable half-mode when manually dragging
    }
  };

  const handleMouseUp = () => {
    isResizing.current = false;
    document.body.style.cursor = "default";
    document.body.style.userSelect = "auto";
  };

  useEffect(() => {
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  // ✅ Toggle to half screen width
  const toggleHalfScreen = () => {
    if (isHalfScreen) {
      setSidebarWidth(350);
      setIsHalfScreen(false);
    } else {
      setSidebarWidth(window.innerWidth * 0.5);
      setIsHalfScreen(true);
    }
  };

  useEffect(() => {
    if (isOpen && messages.length === 0) {
      setMessages([{ role: "assistant", content: `Hello! I'm your AI assistant for "${videoTitle}".` }]);
    }
  }, [isOpen, messages.length, videoTitle]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const getContext = () => transcript.slice(0, 4000) || "";

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userQuestion = input.trim();
    setMessages((prev) => [...prev, { role: "user", content: userQuestion }]);
    setInput("");
    setIsLoading(true);

    const context = getContext();
    const RAG_SYSTEM_PROMPT = `
- Use only the information from the transcript below.
- If the answer is not found in transcript, reply: I don't know based on this video.
- Use proper markdown formatting for:
  - Bullet points (-)
  - Numbered steps (1.)
  - Headings (#, ##)
  - Code blocks using \`\`\`python

Transcript:
${context}

Question:
${userQuestion}

Answer:

`;

    try {
      const response = await fetch("http://localhost:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [
            { role: "system", content: RAG_SYSTEM_PROMPT },
            { role: "user", content: userQuestion },
          ],
        }),
      });

      const raw = await response.text();
      const result = JSON.parse(raw);
      const aiText = result?.choices?.[0]?.message?.content?.trim() || "No valid answer found.";

      setMessages((prev) => [...prev, { role: "assistant", content: aiText }]);
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "Error connecting to AI backend." }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div
      style={{ width: isOpen ? sidebarWidth : 0 }}
      className={`fixed inset-y-0 right-0 bg-gray-900 z-50 transform transition-all duration-200
      ${isOpen ? "translate-x-0" : "translate-x-full"} flex flex-col border-l border-gray-700`}
    >
      {/* ✅ Drag Area */}
      <div
        onMouseDown={handleMouseDown}
        className="absolute left-0 top-0 w-1 h-full cursor-col-resize bg-transparent hover:bg-blue-500/50"
      ></div>

      {/* ✅ Header */}
      <div className="p-4 bg-gray-800 flex justify-between items-center border-b border-gray-700">
        <h3 className="text-lg font-bold flex items-center text-white">
          <MessageSquare className="h-5 w-5 mr-2 text-blue-400" />
          AI Assistant
        </h3>
        <div className="flex gap-2">
          {/* 🔹 Button to resize to half screen */}
          <button onClick={toggleHalfScreen} className="text-gray-400 hover:text-white p-1">
            {isHalfScreen ? <Minimize2 /> : <Maximize2 />}
          </button>
          <button onClick={onClose} className="text-gray-400 hover:text-white p-1 rounded-full hover:bg-gray-700">
            <X className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* ✅ Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
            <div className={`max-w-lg p-3 rounded-lg text-sm ${
              msg.role === "user" ? "bg-blue-600 text-white" : "bg-gray-700 text-gray-100"
            }`}>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const lang = /language-(\w+)/.exec(className || "");
                    return !inline ? (
                      <SyntaxHighlighter style={vscDarkPlus} language={lang?.[1] || "text"} {...props}>
                        {String(children).replace(/\n$/, "")}
                      </SyntaxHighlighter>
                    ) : (
                      <code className="bg-gray-800 px-1 py-0.5 rounded">{children}</code>
                    );
                  },
                }}
              >
                {msg.content}
              </ReactMarkdown>
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="max-w-lg p-3 rounded-lg text-sm bg-gray-700 text-gray-100">
              <span className="animate-pulse">Assistant is typing...</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* ✅ Input */}
      <div className="p-4 border-t border-gray-700">
        <form onSubmit={handleSend} className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask something..."
            className="flex-1 p-2 rounded-lg bg-gray-700 text-white border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="p-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-500"
          >
            <Send className="h-5 w-5" />
          </button>
        </form>
      </div>
    </div>
  );
};

export default AIChatSidebar;
