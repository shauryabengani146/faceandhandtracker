/**
 * ══════════════════════════════════════════════════════════════════
 *  CYBERVISION — Main Application Component
 *
 *  Orchestrates:
 *   • CDN script loading (face-api.js + MediaPipe tasks-vision)
 *   • Webcam permission & stream setup
 *   • Concurrent face + hand detection loops (RAF-throttled to ~30fps)
 *   • Canvas rendering via canvasRenderer.ts
 *   • ASL classification via aslEngine.ts
 *   • Live telemetry sidebar + log
 * ══════════════════════════════════════════════════════════════════
 */

import {
  useRef,
  useState,
  useEffect,
  useCallback,
  useMemo,
} from 'react';
import { analyzeFingers, classifySign, type SignResult, type Landmark } from './aslEngine';
import { drawFace, drawHand, type FaceRenderData } from './canvasRenderer';
import { log, subscribeLog, type LogEntry } from './telemetryLog';

// ─────────────────────────────────────────────────────────────────
//  EXTERNAL CDN URLS
// ─────────────────────────────────────────────────────────────────
const FACE_API_CDN =
  'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.js';
const FACE_MODEL_URL =
  'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
const MEDIAPIPE_CDN =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/vision_bundle.mjs';
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const MEDIAPIPE_WASM =
  'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision/wasm';

// ─────────────────────────────────────────────────────────────────
//  TYPES / INTERFACES
// ─────────────────────────────────────────────────────────────────
type AppPhase = 'splash' | 'loading' | 'running' | 'error';

interface LoadStatus {
  faceApiScript : 'pending' | 'loading' | 'done' | 'error';
  faceModels    : 'pending' | 'loading' | 'done' | 'error';
  handModels    : 'pending' | 'loading' | 'done' | 'error';
  webcam        : 'pending' | 'loading' | 'done' | 'error';
  overall       : number; // 0–100
}

interface FaceTelemetry {
  detected: boolean;
  age: number;
  gender: string;
  genderProb: number;
  emotion: string;
  emotionScore: number;
  expressions: Record<string, number>;
}

interface HandTelemetry {
  detected: boolean;
  handedness: string;
  sign: SignResult | null;
}

// ─────────────────────────────────────────────────────────────────
//  EMOTION BAR COLORS
// ─────────────────────────────────────────────────────────────────
const EMOTION_COLORS: Record<string, string> = {
  happy   : '#00ff88',
  sad     : '#00f5ff',
  angry   : '#ff4444',
  fearful : '#ff6600',
  disgusted: '#cc00ff',
  surprised: '#ffee00',
  neutral : '#5a6a8a',
};

// ─────────────────────────────────────────────────────────────────
//  HELPER: load a script tag dynamically
// ─────────────────────────────────────────────────────────────────
function loadScript(src: string, type = 'text/javascript'): Promise<void> {
  return new Promise((resolve, reject) => {
    if (document.querySelector(`script[src="${src}"]`)) {
      resolve(); return;
    }
    const s = document.createElement('script');
    s.src  = src;
    s.type = type;
    s.crossOrigin = 'anonymous';
    s.onload  = () => resolve();
    s.onerror = () => reject(new Error(`Failed to load: ${src}`));
    document.head.appendChild(s);
  });
}

// ─────────────────────────────────────────────────────────────────
//  HELPER: load an ESM module dynamically
// ─────────────────────────────────────────────────────────────────
async function loadEsmModule(src: string): Promise<Record<string, unknown>> {
  // eslint-disable-next-line @typescript-eslint/no-implied-eval
  return import(/* @vite-ignore */ src);
}

// ─────────────────────────────────────────────────────────────────
//  MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────
export default function App() {
  // ── Refs ──
  const videoRef  = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef    = useRef<number | null>(null);
  const lastFrame = useRef<number>(0);
  const faceApiRef    = useRef<unknown>(null);      // window.faceapi after CDN load
  const handLandRef   = useRef<unknown>(null);      // MediaPipe HandLandmarker instance

  // ── State ──
  const [phase, setPhase]       = useState<AppPhase>('splash');
  const [errorMsg, setErrorMsg] = useState('');
  const [loadStatus, setLoadStatus] = useState<LoadStatus>({
    faceApiScript : 'pending',
    faceModels    : 'pending',
    handModels    : 'pending',
    webcam        : 'pending',
    overall       : 0,
  });

  const [faceTele, setFaceTele] = useState<FaceTelemetry>({
    detected: false, age: 0, gender: '', genderProb: 0,
    emotion: '', emotionScore: 0, expressions: {},
  });
  const [handTele, setHandTele] = useState<HandTelemetry[]>([]);
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [fps, setFps]         = useState(0);
  const [lockState, setLockState] = useState<'idle' | 'face' | 'hand' | 'both'>('idle');
  const fpsFrames = useRef<number[]>([]);

  // ── Subscribe to log ──
  useEffect(() => {
    const unsub = subscribeLog(setLogEntries);
    return unsub;
  }, []);

  // ── Lock state derived from telemetry ──
  useEffect(() => {
    const hasFace = faceTele.detected;
    const hasHand = handTele.some(h => h.detected);
    if (hasFace && hasHand) setLockState('both');
    else if (hasFace)       setLockState('face');
    else if (hasHand)       setLockState('hand');
    else                    setLockState('idle');
  }, [faceTele.detected, handTele]);

  // ─────────────────────────────────────────────────────────────
  //  INITIALISE — load scripts, models, webcam
  // ─────────────────────────────────────────────────────────────
  const initialise = useCallback(async () => {
    setPhase('loading');
    log('sys', 'Initialisation sequence started');

    try {
      // ── Step 1: Load face-api.js IIFE ──────────────────────────
      setLoadStatus(s => ({ ...s, faceApiScript: 'loading', overall: 5 }));
      log('sys', 'Loading face-api.js from CDN…');
      await loadScript(FACE_API_CDN);
      faceApiRef.current = (window as unknown as Record<string,unknown>)['faceapi'];
      setLoadStatus(s => ({ ...s, faceApiScript: 'done', overall: 20 }));
      log('sys', 'face-api.js script loaded ✓');

      // ── Step 2: Load face-api models ───────────────────────────
      setLoadStatus(s => ({ ...s, faceModels: 'loading', overall: 25 }));
      log('face', 'Loading face detection models from CDN…');
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const fa = faceApiRef.current as any;
      await Promise.all([
        fa.nets.tinyFaceDetector.loadFromUri(FACE_MODEL_URL),
        fa.nets.faceLandmark68TinyNet.loadFromUri(FACE_MODEL_URL),
        fa.nets.faceExpressionNet.loadFromUri(FACE_MODEL_URL),
        fa.nets.ageGenderNet.loadFromUri(FACE_MODEL_URL),
      ]);
      setLoadStatus(s => ({ ...s, faceModels: 'done', overall: 50 }));
      log('face', 'Face models loaded ✓ (TinyDetector + Landmarks + Expressions + Age/Gender)');

      // ── Step 3: Load MediaPipe hand landmarker ─────────────────
      setLoadStatus(s => ({ ...s, handModels: 'loading', overall: 55 }));
      log('hand', 'Loading MediaPipe tasks-vision from CDN…');
      const mp = await loadEsmModule(MEDIAPIPE_CDN);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const { FilesetResolver, HandLandmarker } = mp as any;

      log('hand', 'Initialising HandLandmarker (WASM)…');
      const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
      handLandRef.current = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath : HAND_MODEL_URL,
          delegate       : 'GPU',
        },
        runningMode : 'video',
        numHands    : 2,
      });
      setLoadStatus(s => ({ ...s, handModels: 'done', overall: 75 }));
      log('hand', 'HandLandmarker ready ✓ (21-pt skeleton, up to 2 hands)');

      // ── Step 4: Webcam ─────────────────────────────────────────
      setLoadStatus(s => ({ ...s, webcam: 'loading', overall: 80 }));
      log('sys', 'Requesting webcam permission…');
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width     : { ideal: 1280 },
          height    : { ideal: 720 },
          frameRate : { ideal: 30 },
          facingMode: 'user',
        },
        audio: false,
      });

      const video = videoRef.current!;
      video.srcObject = stream;
      await new Promise<void>((res, rej) => {
        video.onloadedmetadata = () => res();
        video.onerror = () => rej(new Error('Video error'));
        setTimeout(() => rej(new Error('Video timeout')), 8000);
      });
      await video.play();
      setLoadStatus(s => ({ ...s, webcam: 'done', overall: 100 }));
      log('sys', `Webcam active — ${video.videoWidth}×${video.videoHeight}px`);

      // ── Step 5: Start detection loop ───────────────────────────
      setPhase('running');
      log('sys', '▶ Detection loop started. All systems nominal.');
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setErrorMsg(msg);
      setPhase('error');
      log('warn', `INIT ERROR: ${msg}`);
    }
  }, []);

  // ─────────────────────────────────────────────────────────────
  //  DETECTION LOOP
  // ─────────────────────────────────────────────────────────────
  const TARGET_FPS        = 30;
  const FRAME_MS          = 1000 / TARGET_FPS;
  const lastLoggedSign    = useRef('');
  const lastLoggedEmotion = useRef('');

  const detectionLoop = useCallback(async (now: number) => {
    rafRef.current = requestAnimationFrame(detectionLoop);

    // Throttle to target FPS
    if (now - lastFrame.current < FRAME_MS) return;
    const delta = now - lastFrame.current;
    lastFrame.current = now;

    // FPS calculation (rolling window)
    fpsFrames.current.push(delta);
    if (fpsFrames.current.length > 30) fpsFrames.current.shift();
    const avgDelta = fpsFrames.current.reduce((a, b) => a + b, 0) / fpsFrames.current.length;
    setFps(Math.round(1000 / avgDelta));

    const video  = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2) return;

    // Sync canvas size with video
    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width  = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const fa  = faceApiRef.current as Record<string, unknown> | null;
    const hl  = handLandRef.current as Record<string, (...a: unknown[]) => unknown> | null;

    // ── Face Detection ──────────────────────────────────────────
    if (fa) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const fa_: any = fa;
        const detections = await fa_
          .detectAllFaces(video, new fa_.TinyFaceDetectorOptions({
            inputSize: 320, scoreThreshold: 0.45
          }))
          .withFaceLandmarks(true)
          .withFaceExpressions()
          .withAgeAndGender();

        if (detections && detections.length > 0) {
          const det = detections[0];
          const box = det.detection.box;
          const exprs: Record<string, number> = det.expressions as Record<string, number>;
          const topEmotion = Object.entries(exprs).sort((a, b) => b[1] - a[1])[0];

          const faceData: FaceRenderData = {
            box: { x: box.x, y: box.y, width: box.width, height: box.height },
            age        : det.age,
            gender     : det.gender,
            genderProb : det.genderProbability,
            emotion    : topEmotion[0],
            emotionScore: topEmotion[1],
            expressions: exprs,
          };

          drawFace(ctx, faceData, canvas.width);

          setFaceTele({
            detected   : true,
            age        : det.age,
            gender     : det.gender,
            genderProb : det.genderProbability,
            emotion    : topEmotion[0],
            emotionScore: topEmotion[1],
            expressions: exprs,
          });

          // Log emotion changes
          if (topEmotion[0] !== lastLoggedEmotion.current && topEmotion[1] > 0.5) {
            lastLoggedEmotion.current = topEmotion[0];
            log('face', `Face: ${topEmotion[0]} (${(topEmotion[1]*100).toFixed(0)}%) | ${det.gender} ~${Math.round(det.age)}yr`);
          }
        } else {
          setFaceTele(t => ({ ...t, detected: false }));
        }
      } catch {
        // silently skip bad frames
      }
    }

    // ── Hand Detection ──────────────────────────────────────────
    if (hl) {
      try {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const result: any = (hl as any).detectForVideo(video, now);
        const newHandTele: HandTelemetry[] = [];

        if (result && result.landmarks && result.landmarks.length > 0) {
          result.landmarks.forEach((lms: Landmark[], i: number) => {
            const handedness =
              result.handednesses?.[i]?.[0]?.displayName ?? 'Unknown';

            const fingers = analyzeFingers(lms);
            const sign    = classifySign(fingers);

            drawHand(ctx, lms, sign, canvas.width, canvas.height, handedness);

            newHandTele.push({ detected: true, handedness, sign });

            // Log sign changes
            const signKey = `${handedness}:${sign?.letter ?? 'null'}`;
            if (signKey !== lastLoggedSign.current) {
              lastLoggedSign.current = signKey;
              if (sign) {
                log('hand', `${handedness} Hand: ASL '${sign.letter}' — ${sign.name}`);
              } else {
                log('hand', `${handedness} Hand: gesture not recognized`);
              }
            }
          });
        }

        setHandTele(newHandTele);
      } catch {
        // silently skip
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Start loop when running
  useEffect(() => {
    if (phase === 'running') {
      rafRef.current = requestAnimationFrame(detectionLoop);
    }
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [phase, detectionLoop]);

  // ─────────────────────────────────────────────────────────────
  //  RESIZE — keep canvas aligned to video
  // ─────────────────────────────────────────────────────────────
  const [videoDims, setVideoDims] = useState({ w: 640, h: 480 });

  const handleVideoReady = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    const ctr = v.parentElement!;
    const maxW = ctr.clientWidth  - 0;
    const maxH = ctr.clientHeight - 0;
    const ratio = v.videoWidth / v.videoHeight;
    let w = maxW;
    let h = w / ratio;
    if (h > maxH) { h = maxH; w = h * ratio; }
    setVideoDims({ w: Math.round(w), h: Math.round(h) });
  }, []);

  useEffect(() => {
    window.addEventListener('resize', handleVideoReady);
    return () => window.removeEventListener('resize', handleVideoReady);
  }, [handleVideoReady]);

  // ─────────────────────────────────────────────────────────────
  //  DERIVED TELEMETRY
  // ─────────────────────────────────────────────────────────────
  const sortedExpressions = useMemo(() =>
    Object.entries(faceTele.expressions)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5),
    [faceTele.expressions]
  );

  // ─────────────────────────────────────────────────────────────
  //  RENDER HELPERS
  // ─────────────────────────────────────────────────────────────
  const statusLabel = (): string => {
    if (loadStatus.faceApiScript === 'loading') return 'Loading face-api.js script…';
    if (loadStatus.faceModels   === 'loading') return 'Loading face detection models…';
    if (loadStatus.handModels   === 'loading') return 'Loading hand landmarker (WASM)…';
    if (loadStatus.webcam       === 'loading') return 'Requesting webcam access…';
    return 'Finalising…';
  };

  const lockClass = (): string => {
    const map = { idle: 'idle', face: 'locked-face', hand: 'locked-hand', both: 'locked-both' };
    return map[lockState];
  };

  // ─────────────────────────────────────────────────────────────
  //  SPLASH SCREEN
  // ─────────────────────────────────────────────────────────────
  if (phase === 'splash') {
    return (
      <>
        <div className="cyber-grid" />
        <div className="splash-screen">
          <div className="splash-logo">CyberVision</div>
          <div className="splash-subtitle">Real-time Face & Sign Language AI · v2.0</div>
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '12px' }}>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-dim)', letterSpacing: '2px' }}>
              POWERED BY
            </div>
            <div style={{ display: 'flex', gap: '16px', fontSize: '0.6rem' }}>
              <span style={{ color: 'var(--neon-cyan)', border: '1px solid rgba(0,245,255,0.3)', padding: '4px 10px' }}>
                face-api.js
              </span>
              <span style={{ color: 'var(--neon-green)', border: '1px solid rgba(0,255,136,0.3)', padding: '4px 10px' }}>
                MediaPipe
              </span>
              <span style={{ color: 'var(--neon-magenta)', border: '1px solid rgba(255,0,200,0.3)', padding: '4px 10px' }}>
                ASL Engine
              </span>
            </div>
          </div>
          <button className="start-btn" onClick={initialise}>
            ▶ INITIALISE SYSTEM
          </button>
          <div className="splash-warning">
            ⚠ Requires webcam access and internet connection to load ML models.
            Processing is performed entirely in your browser — no data is sent to servers.
          </div>
        </div>
      </>
    );
  }

  // ─────────────────────────────────────────────────────────────
  //  ERROR SCREEN
  // ─────────────────────────────────────────────────────────────
  if (phase === 'error') {
    return (
      <>
        <div className="cyber-grid" />
        <div className="error-screen">
          <div className="error-icon">⚠</div>
          <div className="error-title">SYSTEM FAULT</div>
          <div className="error-msg">{errorMsg}</div>
          <div className="error-msg" style={{ fontSize: '0.6rem', opacity: 0.6 }}>
            Ensure your browser supports WebGL, WebAssembly, and has webcam access.
            Check the console for detailed error information.
          </div>
          <button className="retry-btn" onClick={() => { setPhase('splash'); setErrorMsg(''); }}>
            ◀ RETRY
          </button>
        </div>
      </>
    );
  }

  // ─────────────────────────────────────────────────────────────
  //  MAIN APPLICATION UI
  // ─────────────────────────────────────────────────────────────
  return (
    <>
      <div className="cyber-grid" />

      <div className="app-shell">
        {/* ══ HEADER ══ */}
        <header className="app-header">
          <div className="logo-text">CyberVision</div>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            {phase === 'running' && (
              <div className="header-badge badge-live">
                <div className="pulse-dot" />
                LIVE
              </div>
            )}
            <div className="header-badge badge-version">
              face-api.js · mediapipe · ASL v2
            </div>
          </div>
        </header>

        {/* ══ MAIN / VIDEO AREA ══ */}
        <main className="main-area">
          <div
            className={`video-container ${lockClass()}`}
            style={{ width: videoDims.w, height: videoDims.h }}
          >
            {/* Corner brackets */}
            <div className="corner-bracket tl" />
            <div className="corner-bracket tr" />
            <div className="corner-bracket bl" />
            <div className="corner-bracket br" />

            {/* Video */}
            <video
              ref={videoRef}
              id="video-feed"
              autoPlay
              playsInline
              muted
              width={videoDims.w}
              height={videoDims.h}
              onLoadedData={handleVideoReady}
            />

            {/* Canvas overlay */}
            <canvas
              ref={canvasRef}
              id="overlay-canvas"
              width={videoDims.w}
              height={videoDims.h}
              style={{ width: videoDims.w, height: videoDims.h }}
            />

            {/* Loading overlay */}
            {phase === 'loading' && (
              <div
                style={{
                  position: 'absolute', inset: 0,
                  display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center',
                  background: 'rgba(5,8,16,0.9)',
                  gap: 16, borderRadius: 4,
                }}
              >
                <div className="spinner" />
                <div style={{ fontSize: '0.7rem', color: 'var(--neon-cyan)', letterSpacing: '2px' }}>
                  {statusLabel()}
                </div>
                <div style={{ width: '220px' }}>
                  <div className="progress-bar-container">
                    <div
                      className="progress-bar-fill"
                      style={{ width: `${loadStatus.overall}%` }}
                    />
                  </div>
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                    marginTop: '6px', fontSize: '0.55rem', color: 'var(--text-dim)',
                  }}>
                    <span>{loadStatus.overall}%</span>
                    <span style={{ letterSpacing: '1px' }}>
                      {loadStatus.overall < 100 ? 'INITIALISING' : 'READY'}
                    </span>
                  </div>
                </div>
                {/* Substep indicators */}
                <div style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', justifyContent: 'center', maxWidth: '280px' }}>
                  {(
                    [
                      ['face-api.js', loadStatus.faceApiScript],
                      ['Face Models', loadStatus.faceModels],
                      ['Hand Models', loadStatus.handModels],
                      ['Webcam', loadStatus.webcam],
                    ] as Array<[string, string]>
                  ).map(([label, status]) => (
                    <div key={label} style={{
                      display: 'flex', alignItems: 'center', gap: '4px',
                      fontSize: '0.55rem',
                      color: status === 'done' ? 'var(--neon-green)'
                           : status === 'loading' ? 'var(--neon-cyan)'
                           : status === 'error' ? 'var(--neon-magenta)'
                           : 'var(--text-dim)',
                    }}>
                      <span>
                        {status === 'done' ? '✓'
                         : status === 'loading' ? '◈'
                         : status === 'error' ? '✗'
                         : '○'}
                      </span>
                      {label}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </main>

        {/* ══ SIDEBAR ══ */}
        <aside className="sidebar">

          {/* ── Face Telemetry Card ── */}
          <div className="telemetry-cards">
            <div className="panel-header">
              <span className="panel-icon">◈</span>
              TELEMETRY DASHBOARD
            </div>

            {/* Face Card */}
            <div className={`tele-card ${faceTele.detected ? 'active-face' : ''}`}>
              <div className="tele-card-title">
                <div className="dot dot-cyan" />
                FACE ANALYSIS
              </div>
              {faceTele.detected ? (
                <>
                  <div className="tele-row">
                    <span className="tele-label">Age</span>
                    <span className="tele-value cyan">{Math.round(faceTele.age)} yr</span>
                  </div>
                  <div className="tele-row">
                    <span className="tele-label">Gender</span>
                    <span className="tele-value cyan">
                      {faceTele.gender} ({(faceTele.genderProb * 100).toFixed(0)}%)
                    </span>
                  </div>
                  <div className="tele-row">
                    <span className="tele-label">Emotion</span>
                    <span
                      className="tele-value"
                      style={{ color: EMOTION_COLORS[faceTele.emotion] ?? 'var(--neon-cyan)', textTransform: 'capitalize' }}
                    >
                      {faceTele.emotion}
                    </span>
                  </div>

                  {/* Emotion bars */}
                  <div className="emotion-bars">
                    {sortedExpressions.map(([emo, score]) => (
                      <div className="emotion-bar-row" key={emo}>
                        <span className="emotion-bar-label">{emo}</span>
                        <div className="emotion-bar-track">
                          <div
                            className="emotion-bar-fill"
                            style={{
                              width: `${Math.round(score * 100)}%`,
                              background: EMOTION_COLORS[emo] ?? 'var(--neon-cyan)',
                              boxShadow: `0 0 4px ${EMOTION_COLORS[emo] ?? 'var(--neon-cyan)'}`,
                            }}
                          />
                        </div>
                        <span className="emotion-bar-pct">{Math.round(score * 100)}%</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <span className="no-signal">NO FACE DETECTED</span>
              )}
            </div>

            {/* Hand Cards */}
            {handTele.length === 0 ? (
              <div className="tele-card">
                <div className="tele-card-title">
                  <div className="dot dot-green" />
                  HAND TRACKING
                </div>
                <span className="no-signal">NO HANDS DETECTED</span>
              </div>
            ) : handTele.map((ht, i) => (
              <div key={i} className={`tele-card ${ht.detected ? 'active-hand' : ''}`}>
                <div className="tele-card-title">
                  <div className="dot dot-green" />
                  {ht.handedness.toUpperCase()} HAND
                </div>
                {ht.sign ? (
                  <>
                    <div className="sign-badge-large" style={{ marginBottom: '6px' }}>
                      <span className="sign-letter">{ht.sign.letter}</span>
                      <div>
                        <div className="sign-name">{ht.sign.name}</div>
                        <div style={{ fontSize: '0.5rem', color: 'rgba(0,255,136,0.5)', marginTop: '2px' }}>
                          CONF {Math.round(ht.sign.confidence * 100)}%
                        </div>
                      </div>
                    </div>
                  </>
                ) : (
                  <span className="no-signal">SCANNING GESTURE…</span>
                )}
              </div>
            ))}
          </div>

          {/* ── Telemetry Log ── */}
          <div className="log-section">
            <div className="panel-header">
              <span className="panel-icon">▸</span>
              TELEMETRY LOG
              <span style={{ marginLeft: 'auto', color: 'var(--neon-green)', fontSize: '0.55rem' }}>
                {logEntries.length} entries
              </span>
            </div>
            <div className="log-scroll">
              {logEntries.map(entry => (
                <div key={entry.id} className="log-entry">
                  <span className="log-time">{entry.time}</span>
                  <span className="log-msg">
                    {entry.level === 'face' && <span className="face-tag">[FACE] </span>}
                    {entry.level === 'hand' && <span className="hand-tag">[HAND] </span>}
                    {entry.level === 'sys'  && <span className="sys-tag">[SYS]  </span>}
                    {entry.level === 'warn' && <span className="warn-tag">[WARN] </span>}
                    {entry.msg}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </aside>

        {/* ══ FOOTER ══ */}
        <footer className="app-footer">
          <div className="footer-fps">
            <div className="fps-counter">
              <span className="fps-val">{fps}</span>
              <span className="fps-label">fps</span>
            </div>
            <span>|</span>
            <span style={{ color: lockState !== 'idle' ? 'var(--neon-green)' : 'var(--text-dim)' }}>
              {lockState === 'idle' ? 'AWAITING SIGNAL'
               : lockState === 'face' ? '◈ FACE LOCKED'
               : lockState === 'hand' ? '✋ HAND LOCKED'
               : '◈ DUAL LOCK — FACE + HAND'}
            </span>
          </div>
          <div className="footer-right">
            <span>face-api.js @vladmandic</span>
            <span>·</span>
            <span>MediaPipe tasks-vision</span>
            <span>·</span>
            <span style={{ color: 'var(--neon-cyan)' }}>100% Client-Side</span>
          </div>
        </footer>
      </div>
    </>
  );
}
