/**
 * ══════════════════════════════════════════════════════════════════
 *  ASL SIGN LANGUAGE RECOGNITION ENGINE
 *  Uses MediaPipe 21-point hand landmark data
 *  to decode static American Sign Language (ASL) gestures.
 *
 *  ── HAND LANDMARK MAP (MediaPipe) ──
 *
 *   WRIST = 0
 *
 *   THUMB:   CMC=1  MCP=2  IP=3   TIP=4
 *   INDEX:   MCP=5  PIP=6  DIP=7  TIP=8
 *   MIDDLE:  MCP=9  PIP=10 DIP=11 TIP=12
 *   RING:    MCP=13 PIP=14 DIP=15 TIP=16
 *   PINKY:   MCP=17 PIP=18 DIP=19 TIP=20
 *
 *  ── HOW TO ADD MORE SIGNS ──
 *  1. Call analyzeFingers(landmarks) to get a FingerState object.
 *  2. Check the boolean flags: thumb, index, middle, ring, pinky
 *     (true = extended, false = curled) plus thumbOut (pointing sideways).
 *  3. Add an `if` block in classifySign() with a readable comment.
 *  4. Return { letter, name, confidence } from your new branch.
 * ══════════════════════════════════════════════════════════════════
 */

/** Normalised 3D point returned by MediaPipe */
export interface Landmark {
  x: number;
  y: number;
  z: number;
}

/** Per-finger analysis result */
export interface FingerState {
  /** Thumb is mostly extended / pointing away from palm */
  thumb: boolean;
  /** Index finger is extended */
  index: boolean;
  /** Middle finger is extended */
  middle: boolean;
  /** Ring finger is extended */
  ring: boolean;
  /** Pinky finger is extended */
  pinky: boolean;
  /** Thumb is pointing laterally (sideways from the hand axis) */
  thumbOut: boolean;
}

/** Result returned from classifySign() */
export interface SignResult {
  letter: string;      // ASL letter / symbol  e.g. "A"
  name: string;        // Friendly name        e.g. "Alpha"
  confidence: number;  // 0-1 subjective score
}

// ─────────────────────────────────────────────────────────────────
//  GEOMETRY HELPERS
// ─────────────────────────────────────────────────────────────────

/** Euclidean distance between two landmarks (in normalised space) */
function dist(a: Landmark, b: Landmark): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Determine if a finger (index, middle, ring, pinky) is extended.
 *
 * Strategy: The TIP must be "above" (lower y in image coords) or
 * significantly further from the WRIST than the MCP (knuckle).
 *
 * @param tip   - landmark index of the finger tip
 * @param pip   - landmark index of the proximal inter-phalangeal joint (PIP)
 * @param mcp   - landmark index of the metacarpal-phalangeal joint (knuckle)
 * @param wrist - landmark index of the wrist (0)
 * @param lms   - full array of 21 landmarks
 */
function isFingerExtended(
  tip: number,
  pip: number,
  mcp: number,
  wrist: number,
  lms: Landmark[]
): boolean {
  // Primary rule: TIP is clearly above PIP in screen space
  // (lower y = higher on screen in canvas coords)
  const tipAbovePip = lms[tip].y < lms[pip].y;

  // Secondary rule: distance from wrist to TIP > distance from wrist to MCP
  // ensures finger is outward rather than folded under
  const dTipWrist = dist(lms[tip], lms[wrist]);
  const dMcpWrist = dist(lms[mcp], lms[wrist]);

  return tipAbovePip && dTipWrist > dMcpWrist * 0.9;
}

/**
 * Analyse the thumb extension.
 *
 * The thumb moves differently from other fingers.  We compare the
 * Tip x-position against the IP (inner joint) and MCP to determine
 * if it has been pushed laterally away from the palm.
 */
function analyzeThumb(lms: Landmark[]): { extended: boolean; out: boolean } {
  const wrist = lms[0];
  const thumbMcp = lms[2];
  const thumbTip = lms[4];
  const indexMcp = lms[5];

  // Distance from TIP to INDEX MCP – large = thumb is out
  const tipToIndexMcp = dist(thumbTip, indexMcp);

  // Distance from wrist to thumb tip vs wrist to thumb MCP
  const dTipWrist = dist(thumbTip, wrist);
  const dMcpWrist = dist(thumbMcp, wrist);
  const extended  = dTipWrist > dMcpWrist * 1.1;

  // Is thumb pointing laterally?  Compare horizontal displacement of tip vs ip
  const thumbXTravel = Math.abs(thumbTip.x - thumbMcp.x);
  const thumbYTravel = Math.abs(thumbTip.y - thumbMcp.y);
  const out = tipToIndexMcp > 0.12 && thumbXTravel > thumbYTravel * 0.6;

  return { extended, out };
}

// ─────────────────────────────────────────────────────────────────
//  PRIMARY EXPORT: analyzeFingers
// ─────────────────────────────────────────────────────────────────

/**
 * Convert the raw 21 landmarks into a simple FingerState descriptor.
 *
 * Call this FIRST, then pass the result to classifySign().
 */
export function analyzeFingers(lms: Landmark[]): FingerState {
  const WRIST = 0;

  // ── Four fingers (each has MCP, PIP, DIP, TIP) ──
  const index  = isFingerExtended(8,  6,  5,  WRIST, lms);
  const middle = isFingerExtended(12, 10, 9,  WRIST, lms);
  const ring   = isFingerExtended(16, 14, 13, WRIST, lms);
  const pinky  = isFingerExtended(20, 18, 17, WRIST, lms);

  // ── Thumb ──
  const thumbData  = analyzeThumb(lms);
  const thumb      = thumbData.extended;
  const thumbOut   = thumbData.out;

  return { thumb, index, middle, ring, pinky, thumbOut };
}

// ─────────────────────────────────────────────────────────────────
//  PRIMARY EXPORT: classifySign
// ─────────────────────────────────────────────────────────────────

/**
 * Map a FingerState to the most likely ASL letter/sign.
 *
 * Returns null if no confident match is found.
 *
 * ── ADDING NEW SIGNS ─────────────────────────────────────────────
 *  Copy the pattern:
 *
 *    // ASL 'X' — describe which fingers are extended/curled
 *    if (!f.index && !f.middle && !f.ring && !f.pinky && !f.thumb) {
 *      return { letter: 'X', name: 'X-ray', confidence: 0.8 };
 *    }
 *
 *  Place more specific checks BEFORE less specific ones.
 * ─────────────────────────────────────────────────────────────────
 */
export function classifySign(f: FingerState): SignResult | null {

  // ── ASL 'A' ──────────────────────────────────────────────────
  // Fist with thumb resting to the side of curled fingers.
  // All four fingers curled; thumb extended sideways (thumbOut).
  if (!f.index && !f.middle && !f.ring && !f.pinky && f.thumbOut) {
    return { letter: 'A', name: 'Alpha', confidence: 0.85 };
  }

  // ── ASL 'B' ──────────────────────────────────────────────────
  // Four fingers extended straight up; thumb tucked across palm.
  // All fingers up, thumb NOT out.
  if (f.index && f.middle && f.ring && f.pinky && !f.thumbOut) {
    return { letter: 'B', name: 'Bravo', confidence: 0.88 };
  }

  // ── ASL 'C' ──────────────────────────────────────────────────
  // All fingers and thumb curve to form a 'C' shape.
  // No finger is fully extended — all semi-curled with thumb out slightly.
  if (!f.index && !f.middle && !f.ring && !f.pinky && f.thumb && !f.thumbOut) {
    return { letter: 'C', name: 'Charlie', confidence: 0.75 };
  }

  // ── ASL 'D' ──────────────────────────────────────────────────
  // Index finger points up; other three fingers curl to meet thumb tip.
  if (f.index && !f.middle && !f.ring && !f.pinky && !f.thumbOut) {
    return { letter: 'D', name: 'Delta', confidence: 0.80 };
  }

  // ── ASL 'L' ──────────────────────────────────────────────────
  // Index up + thumb pointing out (L shape), other fingers curled.
  if (f.index && !f.middle && !f.ring && !f.pinky && f.thumbOut) {
    return { letter: 'L', name: 'Lima', confidence: 0.88 };
  }

  // ── ASL 'V' / Peace ──────────────────────────────────────────
  // Index and middle up; ring and pinky curled; thumb across palm.
  if (f.index && f.middle && !f.ring && !f.pinky && !f.thumbOut) {
    return { letter: 'V', name: 'Peace / Victory', confidence: 0.90 };
  }

  // ── ASL 'W' ──────────────────────────────────────────────────
  // Index, middle, and ring up; pinky and thumb curled / tucked.
  if (f.index && f.middle && f.ring && !f.pinky && !f.thumbOut) {
    return { letter: 'W', name: 'Whiskey', confidence: 0.85 };
  }

  // ── ASL 'Y' ──────────────────────────────────────────────────
  // Thumb out + pinky extended; index, middle, ring curled.
  if (!f.index && !f.middle && !f.ring && f.pinky && f.thumbOut) {
    return { letter: 'Y', name: 'Yankee', confidence: 0.87 };
  }

  // ── ASL 'I' (ILY also uses this shape with thumb) ────────────
  // Only pinky extended, no thumb lateral.
  if (!f.index && !f.middle && !f.ring && f.pinky && !f.thumbOut) {
    return { letter: 'I', name: 'India', confidence: 0.82 };
  }

  // ── ASL 'U' ──────────────────────────────────────────────────
  // Index and middle extended together; ring, pinky, thumb curled.
  // (Same as V from side — classifier will pick whichever is more stable)
  // NOTE: V and U look similar; V typically has fingers spread, U together.
  // For now mapped under V above (no spread detection yet).

  // ── Open Palm / '5' ──────────────────────────────────────────
  // All fingers AND thumb extended.
  if (f.index && f.middle && f.ring && f.pinky && f.thumbOut) {
    return { letter: '5', name: 'Open Palm / Five', confidence: 0.92 };
  }

  // ── Fist / 'S' ───────────────────────────────────────────────
  // All fingers curled; thumb not out.
  if (!f.index && !f.middle && !f.ring && !f.pinky && !f.thumb && !f.thumbOut) {
    return { letter: 'S', name: 'Sierra (Fist)', confidence: 0.80 };
  }

  // No confident match
  return null;
}
