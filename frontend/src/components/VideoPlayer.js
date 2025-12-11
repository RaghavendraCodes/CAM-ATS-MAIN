import React, { useState, useEffect, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import axios from "axios";
import Webcam from "react-webcam";
import YouTube from "react-youtube";
import {
  Play,
  Pause,
  Square,
  Camera,
  AlertTriangle,
  Eye,
  Volume2,
  VolumeX,
  Settings,
  Home,
  MessageSquare, // <--- NEW ICON IMPORT
} from "lucide-react";
import AIChatSidebar from "./AIChatSidebar"; // Adjust path as necessary

const VideoPlayer = ({ user }) => {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const webcamRef = useRef(null);
  const intervalRef = useRef(null);
  const playerRef = useRef(null);
  const alertSoundRef = useRef(null);

  const [session, setSession] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [currentScore, setCurrentScore] = useState(100);
  const [analysisCount, setAnalysisCount] = useState(0);
  const [lastAnalysis, setLastAnalysis] = useState(null);
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [isPlaying, setIsPlaying] = useState(false);
  const [sessionEnded, setSessionEnded] = useState(false);

  const [isChatOpen, setIsChatOpen] = useState(false); 
  const [videoTranscript, setVideoTranscript] = useState(null);
  const [isTranscriptLoading, setIsTranscriptLoading] = useState(false); // New loading state for the AI button

  useEffect(() => {
    fetchSession();
    setupVisibilityDetection();
    
    // Cleanup on unmount
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [sessionId]);
  
const fetchTranscript = async (url) => {
  if (!url) {
    console.warn("No URL provided for transcript");
    return;
  }

  setIsTranscriptLoading(true);
  setVideoTranscript(null);

  try {
    console.log("🔄 Fetching transcript for:", url);

    const response = await axios.post(`/api/transcript`, { 
      video_url: url 
    });

    console.log("✅ Transcript API Response:", response.data);

    if (response.data?.transcript_text) {
      setVideoTranscript(response.data.transcript_text);
    } else {
      console.error("❌ No 'transcript_text' in backend response");
      setVideoTranscript("Transcript unavailable from backend.");
    }
  } catch (error) {
    console.error("❌ Failed to fetch transcript:", error);
    setVideoTranscript("Transcript unavailable. The AI assistant is limited to general knowledge.");
  } finally {
    setIsTranscriptLoading(false);
  }
};

const fetchSession = async () => {
  try {
    const response = await axios.get(`/sessions/${sessionId}`);
    console.log("✅ Session data:", response.data);

    setSession(response.data);

    if (response.data?.video_url) {
      console.log("🎥 Video URL found:", response.data.video_url);
      await fetchTranscript(response.data.video_url);
    } else {
      console.warn("⚠️ No video_url found in session.");
    }
  } catch (error) {
    console.error("❌ Failed to fetch session:", error);
    navigate("/dashboard");
  }
};


  const setupVisibilityDetection = () => {
    document.addEventListener('visibilitychange', handleVisibilityChange);
  };

  const handleVisibilityChange = () => {
    if (document.hidden && isRecording) {
      // Tab switched - report to backend
      reportTabSwitch();
    }
  };

  const reportTabSwitch = async () => {
    try {
      await axios.post(`/sessions/${sessionId}/tab-switch`);
      addAlert({
        type: "tab_switch",
        description: "Tab switching detected - stay focused!",
        timestamp: new Date(),
      });
    } catch (error) {
      console.error("Failed to report tab switch:", error);
    }
  };

  const startRecording = () => {
    setIsRecording(true);
    // Capture image every 3 seconds
    intervalRef.current = setInterval(captureAndAnalyze, 3000);
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  const endSession = async () => {
    stopRecording();
    try {
      const response = await axios.post(`/sessions/${sessionId}/end`);
      setCurrentScore(response.data.final_score);
      setSessionEnded(true);
    } catch (error) {
      console.error("Failed to end session:", error);
    }
  };

  const captureAndAnalyze = useCallback(async () => {
    if (!webcamRef.current || !cameraEnabled) return;

    try {
      const imageSrc = webcamRef.current.getScreenshot();
      if (!imageSrc) return;

      const response = await axios.post("/analyze/image", {
        session_id: sessionId,
        image_data: imageSrc,
      });

      const { analysis, alerts: newAlerts, attention_score } = response.data;
      
      setLastAnalysis(analysis);
      setAnalysisCount(prev => prev + 1);
      
      // Update score based on attention
      setCurrentScore(prev => Math.max(0, prev - (newAlerts.length * 2)));
      
      // Add new alerts
      newAlerts.forEach(alert => {
        addAlert({
          type: alert.alert_type,
          description: alert.description,
          timestamp: new Date(),
        });
      });

    } catch (error) {
      console.error("Analysis failed:", error);
    }
  }, [sessionId, cameraEnabled]);

  const addAlert = (alert) => {
    setAlerts(prev => [alert, ...prev].slice(0, 10)); // Keep last 10 alerts
    
    // Play alert sound
    if (soundEnabled && alertSoundRef.current) {
      alertSoundRef.current.play().catch(() => {
        // Ignore audio play errors
      });
    }
  };

  const getVideoId = (url) => {
    if (!url) return null;
    const match = url.match(/(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)/);
    return match ? match[1] : null;
  };

  const onPlayerReady = (event) => {
    playerRef.current = event.target;
  };

  const onPlayerStateChange = (event) => {
    // YouTube player state: 1 = playing, 2 = paused
    setIsPlaying(event.data === 1);
    
    if (event.data === 1 && !isRecording) {
      startRecording();
    } else if (event.data === 2 && isRecording) {
      stopRecording();
    }
  };

  const getAlertColor = (alertType) => {
    const colors = {
      no_face_detected: "text-red-400",
      multiple_faces: "text-orange-400",
      mobile_detected: "text-yellow-400",
      head_pose_deviation: "text-blue-400",
      eye_gaze_deviation: "text-purple-400",
      poor_lighting: "text-gray-400",
      tab_switch: "text-red-500",
      yawning: "text-yellow-500",
      laughing: "text-green-400",
    };
    return colors[alertType] || "text-gray-300";
  };

  if (!session) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading session...</div>
      </div>
    );
  }

  const videoId = getVideoId(session.video_url);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}  
        <div className="flex items-center space-x-4 ml-5 justify-items-end">
            {/* NEW AI ASSISTANT BUTTON */}
            <button
                onClick={() => setIsChatOpen(true)}
                className="px-2 py-1 mt-2 mb-2 bg-blue-600 hover:bg-blue-700 rounded-md flex items-center space-x-2 transition-colors"
            >
                <MessageSquare className="h-4 w-4" />
                <span>AI Assistant</span>
            </button>

            <button
                onClick={() => navigate("/dashboard")}
                className="px-4 py-1 bg-gray-600 hover:bg-gray-700 rounded-md flex items-center space-x-2 transition-colors"
            >
                <Home className="h-4 w-4" />
                <span>Dashboard</span>
            </button>
        </div>
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">{session.title}</h1>
            <p className="text-gray-400">
              Started: {new Date(session.start_time).toLocaleString()}
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <div className="text-sm text-gray-400">Current Score</div>
              <div className={`text-xl font-bold ${
                currentScore >= 80 ? 'text-green-400' : 
                currentScore >= 60 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {currentScore}/100
              </div>
            </div>
            <button
              onClick={() => navigate("/dashboard")}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded-md flex items-center space-x-2 transition-colors"
            >
              <Home className="h-4 w-4" />
              <span>Dashboard</span>
            </button>
          </div>
        </div>
      </div>

      <div className="flex h-screen">
        {/* Main Video Area */}
        <div className="flex-1 p-6">
          <div className="bg-gray-800 rounded-lg overflow-hidden h-full">
            {videoId ? (
              <YouTube
                videoId={videoId}
                onReady={onPlayerReady}
                onStateChange={onPlayerStateChange}
                opts={{
                  width: '100%',
                  height: '500',
                  playerVars: {
                    autoplay: 0,
                    controls: 1,
                    rel: 0,
                  },
                }}
                className="w-full"
              />
            ) : (
              <div className="h-96 flex items-center justify-center bg-gray-700">
                <p className="text-gray-400">Invalid video URL</p>
              </div>
            )}

            {/* Session Controls */}
            <div className="p-4 border-t border-gray-700">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  <div className={`flex items-center space-x-2 px-3 py-2 rounded-md ${
                    isRecording ? 'bg-red-600' : 'bg-gray-600'
                  }`}>
                    <Camera className="h-4 w-4" />
                    <span className="text-sm">
                      {isRecording ? 'Monitoring Active' : 'Monitoring Paused'}
                    </span>
                  </div>
                  
                  <div className="text-sm text-gray-400">
                    Analyses: {analysisCount}
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => setCameraEnabled(!cameraEnabled)}
                    className={`p-2 rounded-md transition-colors ${
                      cameraEnabled ? 'bg-green-600 hover:bg-green-700' : 'bg-red-600 hover:bg-red-700'
                    }`}
                  >
                    <Eye className="h-4 w-4" />
                  </button>
                  
                  <button
                    onClick={() => setSoundEnabled(!soundEnabled)}
                    className={`p-2 rounded-md transition-colors ${
                      soundEnabled ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 hover:bg-gray-700'
                    }`}
                  >
                    {soundEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
                  </button>

                  <button
                    onClick={endSession}
                    className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md flex items-center space-x-2 transition-colors"
                  >
                    <Square className="h-4 w-4" />
                    <span>End Session</span>
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <div className="w-80 bg-gray-800 border-l border-gray-700 flex flex-col">
          {/* Camera Feed */}
          <div className="p-4 border-b border-gray-700">
            <h3 className="text-lg font-semibold mb-3 flex items-center">
              <Camera className="h-5 w-5 mr-2 text-blue-500" />
              Camera Feed
            </h3>
            {cameraEnabled ? (
              <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={{
                  width: 320,
                  height: 240,
                  facingMode: "user",
                }}
                className="w-full rounded-lg"
              />
            ) : (
              <div className="w-full h-48 bg-gray-700 rounded-lg flex items-center justify-center">
                <p className="text-gray-400">Camera disabled</p>
              </div>
            )}
          </div>

          {/* Real-time Alerts */}
          <div className="flex-1 p-4 overflow-y-auto">
            <h3 className="text-lg font-semibold mb-3 flex items-center">
              <AlertTriangle className="h-5 w-5 mr-2 text-yellow-500" />
              Live Alerts ({alerts.length})
            </h3>
            <div className="space-y-2">
              {alerts.map((alert, index) => (
                <div
                  key={index}
                  className="p-3 bg-gray-700 rounded-lg border-l-4 border-yellow-500"
                >
                  <div className={`font-medium text-sm ${getAlertColor(alert.type)}`}>
                    {alert.description}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    {alert.timestamp.toLocaleTimeString()}
                  </div>
                </div>
              ))}
              {alerts.length === 0 && (
                <div className="text-center text-gray-400 py-8">
                  <Eye className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>All good! No alerts detected.</p>
                </div>
              )}
            </div>
          </div>

          {/* Analysis Status */}
          {lastAnalysis && (
            <div className="p-4 border-t border-gray-700">
              <h4 className="text-sm font-semibold mb-2">Last Analysis</h4>
              <div className="text-xs text-gray-400 space-y-1">
                <div>Face Detected: {lastAnalysis.face_detected ? "✓" : "✗"}</div>
                <div>Attention Score: {Math.round(lastAnalysis.attention_score * 100)}%</div>
                <div>Expression: {lastAnalysis.facial_expression}</div>
              </div>
            </div>
          )}
        </div>
      </div>

      <AIChatSidebar 
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        transcript={videoTranscript}
        videoTitle={session.title}
    />

      {/* Session End Modal */}
      {sessionEnded && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 p-6 rounded-lg max-w-md w-full mx-4">
            <h2 className="text-xl font-semibold mb-4">Session Completed!</h2>
            <div className="mb-4">
              <p className="text-gray-300">Your final attention score:</p>
              <div className={`text-4xl font-bold mt-2 ${
                currentScore >= 80 ? 'text-green-400' : 
                currentScore >= 60 ? 'text-yellow-400' : 'text-red-400'
              }`}>
                {currentScore}/100
              </div>
            </div>
            <div className="text-sm text-gray-400 mb-6">
              <p>Total alerts: {alerts.length}</p>
              <p>Analyses performed: {analysisCount}</p>
            </div>
            <button
              onClick={() => navigate("/dashboard")}
              className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
            >
              Return to Dashboard
            </button>
          </div>
        </div>
      )}

      {/* Hidden audio element for alert sounds */}
      <audio ref={alertSoundRef} preload="auto">
        <source src="data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmEZAS2I2vLHdyYFLHfP8t2PPgcUYLjn6qJMDgpIoel0IUxDoU=" type="audio/wav" />
      </audio>
    </div>
  );
};

export default VideoPlayer;