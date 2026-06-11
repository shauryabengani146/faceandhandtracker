/**
 * ══════════════════════════════════════════════════════════════════
 *  CANVAS RENDERER — All 2D drawing routines for
 *  face boxes, hand skeletons, and info badges.
 * ══════════════════════════════════════════════════════════════════
 */

import type { Landmark, SignResult } from './aslEngine';

// ─────────────────────────────────────────────────────────────────
//  DESIGN TOKENS
// ─────────────────────────────────────────────────────────────────
const COLOR = {
  FACE_BOX     : '#00f5ff',
  FACE_BOX_DIM : 'rgba(0,245,255,0.25)',
  HAND_JOINT   : '#00ff88',
  HAND_BONE    : 'rgba(0,255,136,0.5)',
  HAND_THUMB   : '#ffee00',
  SIGN_BG      : 'rgba(0,15,12,0.85)',
  SIGN_BORDER  : '#00ff88',
  SIGN_TEXT    : '#00ff88',
  SIGN_LABEL   : 'rgba(0,255,136,0.7)',
  FACE_TAG_BG  : 'rgba(0,10,20,0.85)',
  FACE_TAG_BDR : '#00f5ff',
  FACE_TEXT    : '#00f5ff',
  EMOTION_BAR  : '#ff00c8',
};

// ─────────────────────────────────────────────────────────────────
//  FACE RENDERING
// ─────────────────────────────────────────────────────────────────

export interface FaceRenderData {
  box: { x: number; y: number; width: number; height: number };
  age: number;
  gender: string;
  genderProb: number;
  emotion: string;
  emotionScore: number;
  expressions: Record<string, number>;
}

/**
 * Draw a cyberpunk-styled face bounding box with detection data.
 *
 * The canvas is horizontally mirrored (scaleX(-1)) via CSS so we must
 * mirror the X coords ourselves: mirroredX = canvasWidth - (box.x + box.width)
 */
export function drawFace(
  ctx: CanvasRenderingContext2D,
  data: FaceRenderData,
  canvasW: number
): void {
  const { box, age, gender, emotion, emotionScore, genderProb } = data;

  // Mirror X because canvas is CSS-flipped
  const x = canvasW - (box.x + box.width);
  const y = box.y;
  const w = box.width;
  const h = box.height;

  // ── Glow effect ──
  ctx.save();
  ctx.shadowColor = COLOR.FACE_BOX;
  ctx.shadowBlur  = 14;

  // ── Main bounding box ──
  ctx.strokeStyle = COLOR.FACE_BOX;
  ctx.lineWidth   = 1.5;
  ctx.strokeRect(x, y, w, h);

  // ── Corner accent lines ──
  const cLen = Math.min(w, h) * 0.18;
  ctx.lineWidth = 3;
  const corners: Array<[number, number, number, number, number, number]> = [
    [x,     y,     x + cLen, y,      x,     y + cLen],   // TL
    [x + w, y,     x + w - cLen, y,  x + w, y + cLen],   // TR
    [x,     y + h, x + cLen, y + h,  x,     y + h - cLen], // BL
    [x + w, y + h, x + w - cLen, y + h, x + w, y + h - cLen], // BR
  ];
  for (const [ax, ay, bx, by, cx, cy] of corners) {
    ctx.beginPath();
    ctx.moveTo(bx, by);
    ctx.lineTo(ax, ay);
    ctx.lineTo(cx, cy);
    ctx.stroke();
  }
  ctx.restore();

  // ── Info badge below/inside box ──
  const lineH   = 14;
  const padding = 8;
  const tagLines = [
    `AGE  ${Math.round(age)}yr`,
    `${gender.toUpperCase()} ${(genderProb * 100).toFixed(0)}%`,
    emotion.toUpperCase(),
  ];
  const tagW = 130;
  const tagH = tagLines.length * lineH + padding * 2;
  const tagX = x;
  const tagY = y + h + 4;

  ctx.save();
  ctx.shadowColor = COLOR.FACE_BOX;
  ctx.shadowBlur  = 10;

  // Background
  ctx.fillStyle = COLOR.FACE_TAG_BG;
  ctx.fillRect(tagX, tagY, tagW, tagH);
  ctx.strokeStyle = COLOR.FACE_TAG_BDR;
  ctx.lineWidth   = 1;
  ctx.strokeRect(tagX, tagY, tagW, tagH);

  // Left accent bar
  ctx.fillStyle = COLOR.FACE_BOX;
  ctx.fillRect(tagX, tagY, 2, tagH);

  // Text
  ctx.fillStyle = COLOR.FACE_TEXT;
  ctx.font = 'bold 10px "Courier New", monospace';
  ctx.textAlign = 'left';

  tagLines.forEach((line, i) => {
    if (i === 2) {
      // Emotion gets a colour based on intensity
      const intensity = Math.min(1, emotionScore);
      ctx.fillStyle = interpolateColor(COLOR.FACE_TEXT, COLOR.EMOTION_BAR, intensity * 0.6);
    }
    ctx.fillText(line, tagX + 6, tagY + padding + (i + 1) * lineH - 2);
  });

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────
//  HAND RENDERING
// ─────────────────────────────────────────────────────────────────

/**
 * MediaPipe 21-landmark hand skeleton connections.
 * Each pair is [startIdx, endIdx].
 */
const HAND_CONNECTIONS: Array<[number, number]> = [
  // Thumb
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Index
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Middle
  [0, 9], [9, 10], [10, 11], [11, 12],
  // Ring
  [0, 13], [13, 14], [14, 15], [15, 16],
  // Pinky
  [0, 17], [17, 18], [18, 19], [19, 20],
  // Palm
  [5, 9], [9, 13], [13, 17],
];

/**
 * Draw the full hand skeleton and the ASL sign badge above the wrist.
 *
 * @param ctx       - Canvas 2D rendering context
 * @param landmarks - Array of 21 normalised MediaPipe landmarks
 * @param sign      - Classified ASL sign (or null if no match)
 * @param cw        - Canvas width  (for coordinate scaling)
 * @param ch        - Canvas height (for coordinate scaling)
 * @param handedness - 'Left' or 'Right'
 */
export function drawHand(
  ctx: CanvasRenderingContext2D,
  landmarks: Landmark[],
  sign: SignResult | null,
  cw: number,
  ch: number,
  handedness: string
): void {
  // Convert normalised [0,1] coordinates to canvas pixel coords.
  // Note: MediaPipe reports landmarks relative to the UNMIRRORED frame.
  // Since our canvas CSS mirrors it, we flip X: px.x = (1 - lm.x) * cw
  const pts = landmarks.map(lm => ({
    x: lm.x * cw,   // canvas already CSS-mirrored, so direct mapping works
    y: lm.y * ch,
  }));

  ctx.save();
  ctx.shadowColor = COLOR.HAND_JOINT;
  ctx.shadowBlur  = 8;

  // ── Bones (connections) ──
  for (const [a, b] of HAND_CONNECTIONS) {
    const isThumb = a <= 4 || b <= 4;
    ctx.beginPath();
    ctx.moveTo(pts[a].x, pts[a].y);
    ctx.lineTo(pts[b].x, pts[b].y);
    ctx.strokeStyle = isThumb ? `rgba(255,238,0,0.6)` : COLOR.HAND_BONE;
    ctx.lineWidth   = isThumb ? 2 : 1.5;
    ctx.stroke();
  }

  // ── Joints (dots) ──
  for (let i = 0; i < pts.length; i++) {
    const isTip    = [4, 8, 12, 16, 20].includes(i);
    const isWrist  = i === 0;
    const isThumbT = i <= 4;

    const radius = isTip ? 5 : isWrist ? 5 : 3;
    const color  = isThumbT ? COLOR.HAND_THUMB : COLOR.HAND_JOINT;

    ctx.beginPath();
    ctx.arc(pts[i].x, pts[i].y, radius, 0, Math.PI * 2);
    ctx.fillStyle   = color;
    ctx.shadowColor = color;
    ctx.shadowBlur  = isTip ? 12 : 6;
    ctx.fill();

    // Outer ring for tips
    if (isTip) {
      ctx.beginPath();
      ctx.arc(pts[i].x, pts[i].y, radius + 3, 0, Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth   = 0.8;
      ctx.globalAlpha = 0.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  ctx.restore();

  // ── ASL Sign Badge ──
  drawSignBadge(ctx, pts[0], sign, handedness);
}

/**
 * Render the sign-language identification badge above the wrist.
 */
function drawSignBadge(
  ctx: CanvasRenderingContext2D,
  wristPt: { x: number; y: number },
  sign: SignResult | null,
  handedness: string
): void {
  const bW  = 160;
  const bH  = 54;
  const bX  = wristPt.x - bW / 2;
  const bY  = wristPt.y + 20;

  ctx.save();
  ctx.shadowColor = COLOR.SIGN_BORDER;
  ctx.shadowBlur  = 16;

  // Background panel
  ctx.fillStyle = COLOR.SIGN_BG;
  roundRect(ctx, bX, bY, bW, bH, 3);
  ctx.fill();

  ctx.strokeStyle = sign ? COLOR.SIGN_BORDER : 'rgba(0,255,136,0.2)';
  ctx.lineWidth   = 1;
  roundRect(ctx, bX, bY, bW, bH, 3);
  ctx.stroke();

  // Left accent bar
  ctx.fillStyle = sign ? COLOR.SIGN_BORDER : 'rgba(0,255,136,0.2)';
  roundRect(ctx, bX, bY, 2, bH, 1);
  ctx.fill();

  ctx.shadowBlur = 0;

  if (sign) {
    // Big letter
    ctx.fillStyle   = COLOR.SIGN_TEXT;
    ctx.font        = 'bold 26px "Courier New", monospace';
    ctx.textAlign   = 'left';
    ctx.shadowColor = COLOR.SIGN_BORDER;
    ctx.shadowBlur  = 12;
    ctx.fillText(`[${sign.letter}]`, bX + 10, bY + 34);

    // Sign name
    ctx.font        = '9px "Courier New", monospace';
    ctx.fillStyle   = COLOR.SIGN_LABEL;
    ctx.shadowBlur  = 0;
    ctx.fillText(sign.name.toUpperCase(), bX + 56, bY + 20);

    // Confidence bar
    const barW = bW - 60;
    ctx.fillStyle = 'rgba(0,255,136,0.15)';
    ctx.fillRect(bX + 56, bY + 28, barW, 4);
    ctx.fillStyle = COLOR.SIGN_BORDER;
    ctx.shadowColor = COLOR.SIGN_BORDER;
    ctx.shadowBlur  = 6;
    ctx.fillRect(bX + 56, bY + 28, barW * sign.confidence, 4);

    // Confidence value
    ctx.font      = '8px "Courier New", monospace';
    ctx.fillStyle = 'rgba(0,255,136,0.6)';
    ctx.shadowBlur = 0;
    ctx.fillText(`${Math.round(sign.confidence * 100)}%`, bX + 56, bY + 45);

    // Handedness
    ctx.fillStyle  = 'rgba(0,255,136,0.45)';
    ctx.textAlign  = 'right';
    ctx.fillText(handedness.toUpperCase(), bX + bW - 8, bY + 45);
  } else {
    // No match
    ctx.fillStyle = 'rgba(0,255,136,0.3)';
    ctx.font      = '10px "Courier New", monospace';
    ctx.textAlign = 'left';
    ctx.fillText('SCANNING…', bX + 10, bY + 22);
    ctx.font      = '9px "Courier New", monospace';
    ctx.fillStyle = 'rgba(0,255,136,0.2)';
    ctx.fillText(handedness.toUpperCase() + ' HAND', bX + 10, bY + 38);
  }

  ctx.restore();
}

// ─────────────────────────────────────────────────────────────────
//  UTILITY: roundRect path helper (polyfill)
// ─────────────────────────────────────────────────────────────────
function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number, y: number,
  w: number, h: number,
  r: number
): void {
  if (ctx.roundRect) {
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, r);
  } else {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y,     x + w, y + r,     r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
  }
}

// ─────────────────────────────────────────────────────────────────
//  UTILITY: colour interpolation
// ─────────────────────────────────────────────────────────────────
function interpolateColor(hex1: string, hex2: string, t: number): string {
  const parse = (h: string) => {
    const n = parseInt(h.replace('#', ''), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  };
  const [r1, g1, b1] = parse(hex1);
  const [r2, g2, b2] = parse(hex2);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r},${g},${b})`;
}
