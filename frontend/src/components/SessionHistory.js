import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import {
  Calendar,
  Clock,
  Target,
  AlertTriangle,
  Eye,
  TrendingUp,
  Play,
  Search,
  Filter,
  Download,
} from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from "recharts";

const SessionHistory = ({ user }) => {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedSession, setSelectedSession] = useState(null);
  const [analytics, setAnalytics] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [filterStatus, setFilterStatus] = useState("all");

  useEffect(() => {
    fetchSessions();
  }, []);

  const fetchSessions = async () => {
    try {
      const response = await axios.get("/sessions");
      setSessions(response.data);
    } catch (error) {
      console.error("Failed to fetch sessions:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchSessionAnalytics = async (sessionId) => {
    try {
      const response = await axios.get(`/sessions/${sessionId}/analytics`);
      setAnalytics(response.data);
    } catch (error) {
      console.error("Failed to fetch analytics:", error);
    }
  };

  const handleSessionClick = (session) => {
    setSelectedSession(session);
    fetchSessionAnalytics(session.id);
  };

  const formatDuration = (seconds) => {
    if (!seconds) return "0m";
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return hours > 0 ? `${hours}h ${minutes}m` : `${minutes}m`;
  };

  const getScoreColor = (score) => {
    if (score >= 80) return "text-green-400";
    if (score >= 60) return "text-yellow-400";
    return "text-red-400";
  };

  const getStatusBadge = (status) => {
    const colors = {
      active: "bg-green-600 text-green-100",
      paused: "bg-yellow-600 text-yellow-100",
      completed: "bg-blue-600 text-blue-100",
    };
    return colors[status] || "bg-gray-600 text-gray-100";
  };

  const filteredSessions = sessions.filter((session) => {
    const matchesSearch = session.title.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = filterStatus === "all" || session.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

  // Prepare chart data
  const attentionChartData = analytics?.attention_timeline?.map((point, index) => ({
    time: `${index}`,
    score: Math.round(point.score * 100),
  })) || [];

  const alertPieData = analytics?.alert_statistics?.map((stat, index) => ({
    name: stat._id.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
    value: stat.count,
    color: [
      "#ef4444", "#f97316", "#eab308", "#22c55e", 
      "#3b82f6", "#8b5cf6", "#ec4899", "#6b7280"
    ][index % 8],
  })) || [];

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white text-xl">Loading session history...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-bold mb-2">Session History</h1>
            <p className="text-gray-400">
              Track your learning progress and attention patterns over time
            </p>
          </div>
        </div>

        {/* Search and Filter */}
        <div className="flex space-x-4 mb-6">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-5 w-5" />
            <input
              type="text"
              placeholder="Search sessions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
            />
          </div>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white focus:outline-none focus:border-blue-500"
          >
            <option value="all">All Status</option>
            <option value="completed">Completed</option>
            <option value="active">Active</option>
            <option value="paused">Paused</option>
          </select>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Sessions List */}
          <div className="bg-gray-800 rounded-lg border border-gray-700">
            <div className="p-6 border-b border-gray-700">
              <h2 className="text-xl font-semibold flex items-center">
                <Calendar className="h-5 w-5 mr-2 text-blue-500" />
                Sessions ({filteredSessions.length})
              </h2>
            </div>
            <div className="max-h-96 overflow-y-auto">
              {filteredSessions.map((session) => (
                <div
                  key={session.id}
                  onClick={() => handleSessionClick(session)}
                  className={`p-4 border-b border-gray-700 cursor-pointer hover:bg-gray-700 transition-colors ${
                    selectedSession?.id === session.id ? "bg-gray-700" : ""
                  }`}
                >
                  <div className="flex justify-between items-start mb-2">
                    <h3 className="font-medium text-white">{session.title}</h3>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBadge(session.status)}`}>
                      {session.status}
                    </span>
                  </div>
                  <div className="text-sm text-gray-400 space-y-1">
                    <div className="flex items-center">
                      <Calendar className="h-3 w-3 mr-1" />
                      {new Date(session.start_time).toLocaleDateString()}
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center">
                        <Clock className="h-3 w-3 mr-1" />
                        {formatDuration(session.duration)}
                      </div>
                      <div className="flex items-center space-x-3">
                        <div className="flex items-center">
                          <Target className="h-3 w-3 mr-1" />
                          <span className={getScoreColor(session.total_score)}>
                            {session.total_score || 'N/A'}
                          </span>
                        </div>
                        <div className="flex items-center">
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          {session.alerts_count || 0}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
              {filteredSessions.length === 0 && (
                <div className="p-8 text-center text-gray-400">
                  <Eye className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>No sessions found</p>
                </div>
              )}
            </div>
          </div>

          {/* Session Details */}
          <div className="space-y-6">
            {selectedSession ? (
              <>
                {/* Session Info */}
                <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                  <div className="flex justify-between items-start mb-4">
                    <h2 className="text-xl font-semibold">{selectedSession.title}</h2>
                    {selectedSession.status === "active" && (
                      <button
                        onClick={() => navigate(`/video/${selectedSession.id}`)}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md flex items-center space-x-2 transition-colors"
                      >
                        <Play className="h-4 w-4" />
                        <span>Resume</span>
                      </button>
                    )}
                  </div>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-3">
                      <div>
                        <div className="text-sm text-gray-400">Start Time</div>
                        <div className="text-white">{new Date(selectedSession.start_time).toLocaleString()}</div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-400">Duration</div>
                        <div className="text-white">{formatDuration(selectedSession.duration)}</div>
                      </div>
                    </div>
                    <div className="space-y-3">
                      <div>
                        <div className="text-sm text-gray-400">Final Score</div>
                        <div className={`text-xl font-bold ${getScoreColor(selectedSession.total_score)}`}>
                          {selectedSession.total_score || 'N/A'}/100
                        </div>
                      </div>
                      <div>
                        <div className="text-sm text-gray-400">Total Alerts</div>
                        <div className="text-white">{selectedSession.alerts_count || 0}</div>
                      </div>
                    </div>
                  </div>
                  
                  {selectedSession.description && (
                    <div className="mt-4 pt-4 border-t border-gray-700">
                      <div className="text-sm text-gray-400 mb-1">Description</div>
                      <div className="text-gray-300">{selectedSession.description}</div>
                    </div>
                  )}
                </div>

                {/* Analytics Charts */}
                {analytics && (
                  <>
                    {/* Attention Timeline */}
                    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                      <h3 className="text-lg font-semibold mb-4 flex items-center">
                        <TrendingUp className="h-5 w-5 mr-2 text-green-500" />
                        Attention Timeline
                      </h3>
                      {attentionChartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={200}>
                          <LineChart data={attentionChartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                            <XAxis dataKey="time" stroke="#9CA3AF" />
                            <YAxis stroke="#9CA3AF" />
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: '#1F2937', 
                                border: '1px solid #374151',
                                borderRadius: '8px'
                              }}
                            />
                            <Line 
                              type="monotone" 
                              dataKey="score" 
                              stroke="#3B82F6" 
                              strokeWidth={2}
                              dot={{ fill: '#3B82F6', strokeWidth: 2 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="h-48 flex items-center justify-center text-gray-400">
                          No attention data available
                        </div>
                      )}
                    </div>

                    {/* Alert Distribution */}
                    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                      <h3 className="text-lg font-semibold mb-4 flex items-center">
                        <AlertTriangle className="h-5 w-5 mr-2 text-yellow-500" />
                        Alert Distribution
                      </h3>
                      {alertPieData.length > 0 ? (
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                          <ResponsiveContainer width="100%" height={200}>
                            <PieChart>
                              <Pie
                                data={alertPieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={40}
                                outerRadius={80}
                                paddingAngle={5}
                                dataKey="value"
                              >
                                {alertPieData.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Pie>
                              <Tooltip 
                                contentStyle={{ 
                                  backgroundColor: 'white', 
                                  border: '1px solid #374151',
                                  borderRadius: '8px'
                                }}
                              />
                            </PieChart>
                          </ResponsiveContainer>
                          <div className="space-y-2">
                            {alertPieData.map((entry, index) => (
                              <div key={index} className="flex items-center">
                                <div 
                                  className="w-3 h-3 rounded-full mr-2" 
                                  style={{ backgroundColor: entry.color }}
                                />
                                <span className="text-sm text-gray-300">{entry.name}</span>
                                <span className="ml-auto text-sm text-gray-400">{entry.value}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : (
                        <div className="h-48 flex items-center justify-center text-gray-400">
                          No alerts recorded for this session
                        </div>
                      )}
                    </div>

                    {/* Statistics */}
                    <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
                      <h3 className="text-lg font-semibold mb-4">Session Statistics</h3>
                      <div className="grid grid-cols-2 gap-4 text-sm">
                        <div>
                          <div className="text-gray-400">Average Attention</div>
                          <div className="text-white font-medium">
                            {Math.round(analytics.average_attention * 100)}%
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-400">Total Alerts</div>
                          <div className="text-white font-medium">{analytics.total_alerts}</div>
                        </div>
                      </div>
                    </div>
                  </>
                )}
              </>
            ) : (
              <div className="bg-gray-800 rounded-lg border border-gray-700 p-12 text-center">
                <Eye className="h-16 w-16 mx-auto mb-4 text-gray-600" />
                <h2 className="text-xl font-semibold text-gray-400 mb-2">Select a Session</h2>
                <p className="text-gray-500">Choose a session from the list to view detailed analytics</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionHistory;