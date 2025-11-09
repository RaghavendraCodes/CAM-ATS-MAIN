from fastapi import FastAPI, APIRouter, HTTPException, Depends, UploadFile, Request, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timedelta
import jwt
from passlib.context import CryptContext
import cv2
import numpy as np
import mediapipe as mp
import base64
import io
from PIL import Image
import asyncio
import json
from enum import Enum
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import re
from transcript_utils import extract_video_id, fetch_transcript
from rag_pipeline import get_answer
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

os.environ["USE_TF"] = "0"       # Disable TensorFlow for HuggingFace Transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"


HF_API_TOKEN = os.getenv("HF_API_TOKEN")


# Create the main app without a prefix
app = FastAPI(title="CAM ATS - Context-Aware Multimodal Attention Tracking System")
    
# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.environ.get("SECRET_KEY", "your-secret-key")

# Initialize MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Enums
class AlertType(str, Enum):
    NO_FACE_DETECTED = "no_face_detected"
    MULTIPLE_FACES = "multiple_faces"
    MOBILE_DETECTED = "mobile_detected"
    HEAD_POSE_DEVIATION = "head_pose_deviation"
    EYE_GAZE_DEVIATION = "eye_gaze_deviation"
    POOR_LIGHTING = "poor_lighting"
    LOUD_SOUND = "loud_sound"
    TAB_SWITCH = "tab_switch"
    YAWNING = "yawning"
    LAUGHING = "laughing"
    NOT_FOCUSED = "not_focused"

class SessionStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"

# Database Models
class User(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    email: str
    name: str
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

class UserCreate(BaseModel):
    email: str
    name: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Session(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    video_url: str
    title: str
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    duration: Optional[int] = None  # in seconds
    status: SessionStatus = SessionStatus.ACTIVE
    total_score: int = 100
    alerts_count: int = 0
    description: Optional[str] = None

class Alert(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_id: str
    alert_type: AlertType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: float = 0.0
    description: str
    score_deduction: int = 5

class AnalysisResult(BaseModel):
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    face_detected: bool = False
    face_count: int = 0
    mobile_detected: bool = False
    head_pose: Dict[str, float] = {}
    eye_gaze: Dict[str, float] = {}
    lighting_quality: float = 0.0
    facial_expression: str = "neutral"
    attention_score: float = 0.0

class SessionCreate(BaseModel):
    video_url: str
    title: str

class ImageAnalysisRequest(BaseModel):
    session_id: str
    image_data: str  # base64 encoded image

# Computer Vision Analysis Class
class AttentionAnalyzer:
    def __init__(self):
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=3,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Comprehensive image analysis for attention tracking"""
        results = {
            "face_detected": False,
            "face_count": 0,
            "mobile_detected": False,
            "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
            "eye_gaze": {"x": 0, "y": 0},
            "lighting_quality": 0.0,
            "facial_expression": "neutral",
            "attention_score": 0.0,
            "alerts": []
        }

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 1. Face Detection
        face_results = self.face_detection.process(rgb_image)
        if face_results.detections:
            results["face_detected"] = True
            results["face_count"] = len(face_results.detections)
            
            if results["face_count"] > 1:
                results["alerts"].append({
                    "type": AlertType.MULTIPLE_FACES,
                    "description": f"Multiple faces detected: {results['face_count']}"
                })
        else:
            results["alerts"].append({
                "type": AlertType.NO_FACE_DETECTED,
                "description": "No face detected in the image"
            })

        # 2. Face Mesh Analysis for detailed face tracking
        mesh_results = self.face_mesh.process(rgb_image)
        if mesh_results.multi_face_landmarks and len(mesh_results.multi_face_landmarks) > 0:
            landmarks = mesh_results.multi_face_landmarks[0]
            
            # Head pose estimation
            results["head_pose"] = self._estimate_head_pose(landmarks, image.shape)
            
            # Eye gaze estimation
            results["eye_gaze"] = self._estimate_eye_gaze(landmarks)
            
            # Facial expression analysis
            results["facial_expression"] = self._analyze_facial_expression(landmarks)
            
            # Check for yawning or excessive laughing
            if results["facial_expression"] in ["yawning", "laughing"]:
                results["alerts"].append({
                    "type": AlertType.YAWNING if results["facial_expression"] == "yawning" else AlertType.LAUGHING,
                    "description": f"Detected {results['facial_expression']}"
                })

        # 3. Mobile/Phone detection
        results["mobile_detected"] = self._detect_mobile_device(image)
        if results["mobile_detected"]:
            results["alerts"].append({
                "type": AlertType.MOBILE_DETECTED,
                "description": "Mobile device detected in the scene"
            })

        # 4. Lighting analysis
        results["lighting_quality"] = self._analyze_lighting(image)
        if results["lighting_quality"] < 0.3:
            results["alerts"].append({
                "type": AlertType.POOR_LIGHTING,
                "description": "Poor lighting conditions detected"
            })

        # 5. Head pose alerts
        head_pose = results["head_pose"]
        if abs(head_pose["yaw"]) > 30 or abs(head_pose["pitch"]) > 25:
            results["alerts"].append({
                "type": AlertType.HEAD_POSE_DEVIATION,
                "description": f"Head turned away: yaw={head_pose['yaw']:.1f}°, pitch={head_pose['pitch']:.1f}°"
            })

        # 6. Eye gaze alerts
        eye_gaze = results["eye_gaze"]
        if abs(eye_gaze["x"]) > 0.3 or abs(eye_gaze["y"]) > 0.3:
            results["alerts"].append({
                "type": AlertType.EYE_GAZE_DEVIATION,
                "description": "Eyes not looking at the screen"
            })

        # 7. Calculate attention score
        results["attention_score"] = self._calculate_attention_score(results)

        return results

    def _estimate_head_pose(self, landmarks, image_shape):
        """Estimate head pose from MediaPipe facial landmarks using SolvePnP"""
        try:
            h, w = image_shape[:2]

            # 3D model points of facial landmarks (in a generic model space, mm units)
            model_points = np.array([
                (0.0, 0.0, 0.0),             # Nose tip
                (0.0, -330.0, -65.0),        # Chin
                (-225.0, 170.0, -135.0),     # Left eye left corner
                (225.0, 170.0, -135.0),      # Right eye right corner
                (-150.0, -150.0, -125.0),    # Left mouth corner
                (150.0, -150.0, -125.0)      # Right mouth corner
            ], dtype=np.float32)

            # Extract corresponding 2D points from MediaPipe landmarks
            image_points = np.array([
                (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),     # Nose tip
                (landmarks.landmark[152].x * w, landmarks.landmark[152].y * h), # Chin
                (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h),   # Left eye left corner
                (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h), # Right eye right corner
                (landmarks.landmark[61].x * w, landmarks.landmark[61].y * h),   # Left mouth corner
                (landmarks.landmark[291].x * w, landmarks.landmark[291].y * h)  # Right mouth corner
            ], dtype=np.float32)

            # Camera internals (assume no lens distortion)
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float32)

            dist_coeffs = np.zeros((4, 1))  # no distortion

            # SolvePnP: estimate head pose
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return {"yaw": 0, "pitch": 0, "roll": 0}

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

            # Extract Euler angles (yaw, pitch, roll)
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)

            pitch = np.arctan2(-rotation_matrix[2, 0], sy) * 180 / np.pi
            yaw   = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]) * 180 / np.pi
            roll  = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]) * 180 / np.pi

            return {"yaw": yaw, "pitch": pitch, "roll": roll}

        except Exception as e:
            print("Head pose error:", e)
            return {"yaw": 0, "pitch": 0, "roll": 0}

    def _estimate_eye_gaze(self, landmarks):
        """Estimate eye gaze direction - detect if looking away (sides, up, down)"""
        try:
            # Iris landmark indices
            LEFT_IRIS = 468
            RIGHT_IRIS = 473
            
            # Eye corners for reference
            LEFT_EYE_LEFT = 33
            LEFT_EYE_RIGHT = 133
            RIGHT_EYE_LEFT = 263
            RIGHT_EYE_RIGHT = 362
            
            # Get iris positions
            left_iris_x = landmarks.landmark[LEFT_IRIS].x
            left_iris_y = landmarks.landmark[LEFT_IRIS].y
            right_iris_x = landmarks.landmark[RIGHT_IRIS].x
            right_iris_y = landmarks.landmark[RIGHT_IRIS].y
            
            # Get eye reference points
            left_eye_left = landmarks.landmark[LEFT_EYE_LEFT].x
            left_eye_right = landmarks.landmark[LEFT_EYE_RIGHT].x
            left_eye_top = landmarks.landmark[159].y
            left_eye_bottom = landmarks.landmark[145].y
            
            right_eye_left = landmarks.landmark[RIGHT_EYE_LEFT].x
            right_eye_right = landmarks.landmark[RIGHT_EYE_RIGHT].x
            right_eye_top = landmarks.landmark[386].y
            right_eye_bottom = landmarks.landmark[374].y
            
            # Calculate eye centers
            left_eye_center_x = (left_eye_left + left_eye_right) / 2
            left_eye_center_y = (left_eye_top + left_eye_bottom) / 2
            
            right_eye_center_x = (right_eye_left + right_eye_right) / 2
            right_eye_center_y = (right_eye_top + right_eye_bottom) / 2
            
            # Normalized iris position relative to eye center
            left_gaze_x = (left_iris_x - left_eye_center_x) / (left_eye_right - left_eye_left)
            left_gaze_y = (left_iris_y - left_eye_center_y) / (left_eye_bottom - left_eye_top)
            
            right_gaze_x = (right_iris_x - right_eye_center_x) / (right_eye_right - right_eye_left)
            right_gaze_y = (right_iris_y - right_eye_center_y) / (right_eye_bottom - right_eye_top)
            
            # Average both eyes
            gaze_x = (left_gaze_x + right_gaze_x) / 2
            gaze_y = (left_gaze_y + right_gaze_y) / 2
            
            return {"x": gaze_x, "y": gaze_y}
            
        except:
            return {"x": 0, "y": 0}

    def _analyze_facial_expression(self, landmarks):
        """Analyze facial expression for yawning, laughing, etc."""
        try:
            # Mouth landmarks
            upper_lip = landmarks.landmark[13]
            lower_lip = landmarks.landmark[14]
            left_mouth = landmarks.landmark[61]
            right_mouth = landmarks.landmark[291]
            
            # Calculate mouth opening
            mouth_height = abs(upper_lip.y - lower_lip.y)
            mouth_width = abs(left_mouth.x - right_mouth.x)
            
            # Eye landmarks for detecting if eyes are open/closed
            left_eye_top = landmarks.landmark[159]
            left_eye_bottom = landmarks.landmark[145]
            right_eye_top = landmarks.landmark[386]
            right_eye_bottom = landmarks.landmark[374]
            
            left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)
            right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)
            avg_eye_height = (left_eye_height + right_eye_height) / 2
            
            # Determine expression
            if mouth_height > 0.02 and mouth_width > 0.04:
                return "yawning"
            elif mouth_width > 0.035 and mouth_height > 0.015:
                return "laughing"
            elif avg_eye_height < 0.005:
                return "eyes_closed"
            else:
                return "neutral"
        except:
            return "neutral"

    def _detect_mobile_device(self, image):
        """Detect mobile devices in the image using basic computer vision"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use edge detection to find rectangular objects
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's rectangular and has appropriate size
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h if h > 0 else 0
                    area = cv2.contourArea(contour)
                    
                    # Mobile phone typically has aspect ratio between 0.4-0.7 and reasonable size
                    if 0.4 < aspect_ratio < 0.7 and 1000 < area < 50000:
                        return True
            
            return False
        except:
            return False

    def _analyze_lighting(self, image):
        """Analyze lighting conditions"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness metrics
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray)
            
            # Normalize to 0-1 scale
            brightness_score = mean_brightness / 255.0
            contrast_score = std_brightness / 128.0
            
            # Combine metrics (good lighting has moderate brightness and good contrast)
            lighting_quality = 0.7 * brightness_score + 0.3 * min(contrast_score, 1.0)
            
            # Penalize very dark or very bright images
            if brightness_score < 0.2 or brightness_score > 0.9:
                lighting_quality *= 0.5
                
            return min(lighting_quality, 1.0)
        except:
            return 0.5

    def _calculate_attention_score(self, results: Dict[str, Any]) -> float:
        """
        Calculates a more logical, scaled attention score based on analysis results.
        The score starts at 1.0 (100%) and deductions are applied based on the severity of infractions.
        """
        
        # =================================================================================
        # 1. Foundational Checks (Is the user present and alone?)
        # Rationale: If the basic conditions aren't met, a score is meaningless.
        # This is the most critical factor.
        # =================================================================================
        if not results.get("face_detected"):
            return 0.0  # If no face is detected, attention is zero.
        if results.get("face_count", 0) > 1:
            return 0.3  # Presence of others is a major distraction, cap score.

        # Start with a perfect score.
        score = 1.0
        
        # =================================================================================
        # 2. Major Distraction Penalties
        # Rationale: Certain actions have a high, fixed cost to attention. These are
        # typically conscious choices to disengage from the primary task.
        # =================================================================================
        # A. Mobile Phone Usage
        if results.get("mobile_detected"):
            score -= 0.60  # Using a phone is a severe distraction. Heavy penalty.
            
        # B. Yawning / Drowsiness
        if results.get("facial_expression") == "yawning":
            score -= 0.35  # Yawning is a strong indicator of low engagement or fatigue.

        # =================================================================================
        # 3. Gradual & Scaled Penalties for Gaze and Pose
        # Rationale: Penalize based on *how much* the user deviates. A small, natural 
        # movement shouldn't be punished. Only significant, sustained deviations should.
        # This addresses your request for "extreme" turns only.
        # =================================================================================
        
        # A. Head Pose (Yaw & Pitch)
        head_pose = results.get("head_pose", {})
        yaw = abs(head_pose.get("yaw", 0))
        pitch = abs(head_pose.get("pitch", 0))
        
        # Define a "safe zone" where no penalty is applied for normal movement.
        YAW_SAFE_ZONE = 30   # Degrees
        PITCH_SAFE_ZONE = 25 # Degrees
        
        # Define the "extreme" angle at which the maximum penalty is applied.
        YAW_EXTREME_ZONE = 75   # A near 90-degree turn
        PITCH_EXTREME_ZONE = 60 # Looking almost straight up or down

        # Only apply a penalty if the turn is outside the safe zone.
        if yaw > YAW_SAFE_ZONE:
            # The penalty scales linearly from 0 (at the edge of the safe zone)
            # to a max of 0.5 (at the extreme zone).
            yaw_penalty = 0.5 * (yaw - YAW_SAFE_ZONE) / (YAW_EXTREME_ZONE - YAW_SAFE_ZONE)
            score -= min(yaw_penalty, 0.5) # Cap the penalty

        if pitch > PITCH_SAFE_ZONE:
            pitch_penalty = 0.4 * (pitch - PITCH_SAFE_ZONE) / (PITCH_EXTREME_ZONE - PITCH_SAFE_ZONE)
            score -= min(pitch_penalty, 0.4) # Cap the penalty

        # B. Eye Gaze Deviation
        eye_gaze = results.get("eye_gaze", {})
        gaze_deviation = abs(eye_gaze.get("x", 0))

        # Gaze is more sensitive, so it has a smaller safe zone.
        GAZE_SAFE_ZONE = 0.4
        GAZE_EXTREME_ZONE = 0.9 # Looking at the far edge of the screen

        if gaze_deviation > GAZE_SAFE_ZONE:
            gaze_penalty = 0.3 * (gaze_deviation - GAZE_SAFE_ZONE) / (GAZE_EXTREME_ZONE - GAZE_SAFE_ZONE)
            score -= min(gaze_penalty, 0.3)

        # =================================================================================
        # 4. Environmental Factor Penalties
        # Rationale: Environmental factors affect attention but may be outside the
        # user's immediate control. They should receive a smaller, fixed penalty.
        # =================================================================================
        if results.get("lighting_quality", 1.0) < 0.3:
            score -= 0.15  # Poor lighting makes focus harder and can indicate a poor setup.

        # Final score is clamped between 0.0 and 1.0.
        return max(score, 0.0)

# Initialize analyzer
analyzer = AttentionAnalyzer()

# Helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        user = await db.users.find_one({"id": user_id})
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return User(**user)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data:image/jpeg;base64, prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_string)
        
        # Convert to PIL Image then to OpenCV
        pil_image = Image.open(io.BytesIO(image_data))
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

# API Routes
@api_router.post("/auth/register", response_model=Dict[str, Any])
async def register_user(user_data: UserCreate):
    """Register a new user"""
    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        name=user_data.name,
        password_hash=hashed_password
    )
    
    await db.users.insert_one(user.dict())
    
    # Create access token
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=timedelta(days=30)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user.id, "email": user.email, "name": user.name}
    }

@api_router.post("/auth/login", response_model=Dict[str, Any])
async def login_user(login_data: UserLogin):
    """Login user and return access token"""
    user = await db.users.find_one({"email": login_data.email})
    if not user or not verify_password(login_data.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    access_token = create_access_token(
        data={"sub": user["id"]}, expires_delta=timedelta(days=30)
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {"id": user["id"], "email": user["email"], "name": user["name"]}
    }

@api_router.post("/sessions", response_model=Session)
async def create_session(session_data: SessionCreate, current_user: User = Depends(get_current_user)):
    """Create a new learning session"""
    session = Session(
        user_id=current_user.id,
        video_url=session_data.video_url,
        title=session_data.title
    )
    
    await db.sessions.insert_one(session.dict())
    return session

@api_router.get("/sessions", response_model=List[Session])
async def get_user_sessions(current_user: User = Depends(get_current_user)):
    """Get all sessions for the current user"""
    sessions = await db.sessions.find({"user_id": current_user.id}).sort("start_time", -1).to_list(100)
    return [Session(**session) for session in sessions]

@api_router.get("/sessions/{session_id}", response_model=Session)
async def get_session(session_id: str, current_user: User = Depends(get_current_user)):
    """Get a specific session"""
    session = await db.sessions.find_one({"id": session_id, "user_id": current_user.id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return Session(**session)

@api_router.post("/sessions/{session_id}/end")
async def end_session(session_id: str, current_user: User = Depends(get_current_user)):
    """End a learning session"""
    session = await db.sessions.find_one({"id": session_id, "user_id": current_user.id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    end_time = datetime.utcnow()
    start_time = session["start_time"]
    duration = int((end_time - start_time).total_seconds())
    
    # Calculate final score based on alerts
    alerts_count = await db.alerts.count_documents({"session_id": session_id})
    final_score = max(100 - (alerts_count * 5), 0)
    
    # Generate session description
    description = f"Session completed with {alerts_count} alerts and final score of {final_score}/100"
    
    await db.sessions.update_one(
        {"id": session_id},
        {
            "$set": {
                "end_time": end_time,
                "duration": duration,
                "status": SessionStatus.COMPLETED,
                "total_score": final_score,
                "alerts_count": alerts_count,
                "description": description
            }
        }
    )
    
    return {"message": "Session ended successfully", "final_score": final_score}

@api_router.post("/analyze/image")
async def analyze_image(request: ImageAnalysisRequest, current_user: User = Depends(get_current_user)):
    """Analyze uploaded image for attention tracking"""
    try:
        # Verify session exists and belongs to user
        session = await db.sessions.find_one({
            "id": request.session_id, 
            "user_id": current_user.id,
            "status": SessionStatus.ACTIVE
        })
        if not session:
            raise HTTPException(status_code=404, detail="Active session not found")
        
        # Decode and analyze image
        image = decode_base64_image(request.image_data)
        analysis_results = analyzer.analyze_image(image)
        
        # Store analysis results
        analysis_record = AnalysisResult(
            session_id=request.session_id,
            face_detected=analysis_results["face_detected"],
            face_count=analysis_results["face_count"],
            mobile_detected=analysis_results["mobile_detected"],
            head_pose=analysis_results["head_pose"],
            eye_gaze=analysis_results["eye_gaze"],
            lighting_quality=analysis_results["lighting_quality"],
            facial_expression=analysis_results["facial_expression"],
            attention_score=analysis_results["attention_score"]
        )
        
        await db.analysis_results.insert_one(analysis_record.dict())
        
        # Process alerts
        alerts_created = []
        for alert_data in analysis_results["alerts"]:
            alert = Alert(
                session_id=request.session_id,
                user_id=current_user.id,
                alert_type=alert_data["type"],
                description=alert_data["description"],
                confidence=0.8  # Default confidence
            )
            await db.alerts.insert_one(alert.dict())
            alerts_created.append(alert)
        
        return {
            "analysis": analysis_record.dict(),
            "alerts": [alert.dict() for alert in alerts_created],
            "attention_score": analysis_results["attention_score"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/sessions/{session_id}/alerts", response_model=List[Alert])
async def get_session_alerts(session_id: str, current_user: User = Depends(get_current_user)):
    """Get all alerts for a session"""
    alerts = await db.alerts.find({
        "session_id": session_id, 
        "user_id": current_user.id
    }).sort("timestamp", -1).to_list(1000)
    return [Alert(**alert) for alert in alerts]

@api_router.get("/sessions/{session_id}/analytics")
async def get_session_analytics(session_id: str, current_user: User = Depends(get_current_user)):
    """Get comprehensive analytics for a session"""
    # Get session
    session = await db.sessions.find_one({"id": session_id, "user_id": current_user.id})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Get alerts grouped by type
    pipeline = [
        {"$match": {"session_id": session_id}},
        {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    alert_stats = await db.alerts.aggregate(pipeline).to_list(100)
    
    # Get analysis results timeline
    analysis_results = await db.analysis_results.find({
        "session_id": session_id
    }).sort("timestamp", 1).to_list(1000)
    
    # Calculate attention score over time
    attention_timeline = []
    for result in analysis_results:
        attention_timeline.append({
            "timestamp": result["timestamp"],
            "score": result["attention_score"]
        })
    
    return {
        "session": Session(**session).dict(),
        "alert_statistics": alert_stats,
        "attention_timeline": attention_timeline,
        "total_alerts": len(alert_stats),
        "average_attention": sum([r["attention_score"] for r in analysis_results]) / len(analysis_results) if analysis_results else 0
    }

@api_router.post("/sessions/{session_id}/tab-switch")
async def report_tab_switch(session_id: str, current_user: User = Depends(get_current_user)):
    """Report tab switching event"""
    session = await db.sessions.find_one({
        "id": session_id, 
        "user_id": current_user.id,
        "status": SessionStatus.ACTIVE
    })
    if not session:
        raise HTTPException(status_code=404, detail="Active session not found")
    
    alert = Alert(
        session_id=session_id,
        user_id=current_user.id,
        alert_type=AlertType.TAB_SWITCH,
        description="User switched to another tab/window",
        confidence=1.0,
        score_deduction=10
    )
    
    await db.alerts.insert_one(alert.dict())
    return {"message": "Tab switch recorded", "alert_id": alert.id}

@api_router.get("/dashboard")
async def get_dashboard_data(current_user: User = Depends(get_current_user)):
    """Get comprehensive dashboard data for the user"""
    # Get recent sessions
    recent_sessions = await db.sessions.find({
        "user_id": current_user.id
    }).sort("start_time", -1).limit(10).to_list(10)
    
    # Get total statistics
    total_sessions = await db.sessions.count_documents({"user_id": current_user.id})
    total_alerts = await db.alerts.count_documents({"user_id": current_user.id})
    
    # Get alert type distribution
    pipeline = [
        {"$match": {"user_id": current_user.id}},
        {"$group": {"_id": "$alert_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]
    alert_distribution = await db.alerts.aggregate(pipeline).to_list(100)
    
    # Calculate average session duration
    completed_sessions = await db.sessions.find({
        "user_id": current_user.id,
        "status": SessionStatus.COMPLETED,
        "duration": {"$exists": True}
    }).to_list(1000)
    
    avg_duration = sum([s["duration"] for s in completed_sessions]) / len(completed_sessions) if completed_sessions else 0
    
    return {
        "user": {"id": current_user.id, "name": current_user.name, "email": current_user.email},
        "recent_sessions": [Session(**session).dict() for session in recent_sessions],
        "statistics": {
            "total_sessions": total_sessions,
            "total_alerts": total_alerts,
            "average_duration": avg_duration,
            "alert_distribution": alert_distribution
        }
    }

# Root endpoint
@api_router.get("/")
async def root():
    return {"message": "CAM ATS - Context-Aware Multimodal Attention Tracking System API"}

@app.post("/chat")
async def chat(request: Request):  # ✅ No more "Request not defined"
    try:
        data = await request.json()
        messages = data.get("messages", [])

        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {HF_API_TOKEN}",
                "Content-Type": "application/json",
            },
            json={
                "model": "meta-llama/Meta-Llama-3-8B-Instruct",
                "messages": messages,
            },
        )

        return response.json()

    except Exception as e:
        return {"error": str(e)}
    
def extract_video_id(url: str):
    """Extract the video ID from YouTube URLs like https://youtu.be/... or https://youtube.com/watch?v=..."""
    parsed = urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    if parsed.hostname in ("www.youtube.com", "youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
    return None

def fetch_transcript(video_id: str):
    try:
        transcript_data = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        return " ".join(item.text for item in transcript_data)  # ✅ FIXED
    except Exception as e:
        print("Error while fetching transcript:", e)
        return None

@api_router.post("/transcript")
async def get_transcript(request: Request):
    data = await request.json()
    video_url = data.get("video_url")
    video_id = extract_video_id(video_url)

    if not video_id:
        return {"transcript_text": "", "error": "Invalid YouTube URL"}

    try:
        transcript = fetch_transcript(video_id)
        if not transcript:
            return {"transcript_text": "", "error": "Transcript not available"}
        return {"transcript_text": transcript}
    
    except (TranscriptsDisabled, NoTranscriptFound):
        return {"transcript_text": "", "error": "Transcript disabled or not found"}


@api_router.post("/api/ask")
async def ask_video_ai(request: Request):
    data = await request.json()
    video_url = data.get("video_url")
    question = data.get("question", "")

    video_id = extract_video_id(video_url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    transcript = fetch_transcript(video_id)
    if not transcript:
        return {"answer": "Transcript not available for this video."}

    answer = get_answer(transcript, question, video_id)
    return {"answer": answer}


# Health check
@api_router.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

# Include the router in the main app
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()