/**
 * ══════════════════════════════════════════════════════════════════
 *  TELEMETRY LOG — Lightweight reactive log store
 * ══════════════════════════════════════════════════════════════════
 */

export type LogLevel = 'sys' | 'face' | 'hand' | 'warn';

export interface LogEntry {
  id: number;
  time: string;
  level: LogLevel;
  msg: string;
}

const MAX_ENTRIES = 120;

let _counter  = 0;
let _entries: LogEntry[] = [];
let _listener: ((entries: LogEntry[]) => void) | null = null;

function timestamp(): string {
  const d = new Date();
  return [
    String(d.getHours()).padStart(2, '0'),
    String(d.getMinutes()).padStart(2, '0'),
    String(d.getSeconds()).padStart(2, '0'),
  ].join(':') + '.' + String(d.getMilliseconds()).padStart(3, '0').slice(0, 2);
}

export function log(level: LogLevel, msg: string): void {
  const entry: LogEntry = {
    id   : ++_counter,
    time : timestamp(),
    level,
    msg,
  };
  _entries = [entry, ..._entries].slice(0, MAX_ENTRIES);
  _listener?.([..._entries]);
}

export function subscribeLog(cb: (entries: LogEntry[]) => void): () => void {
  _listener = cb;
  cb([..._entries]);
  return () => { if (_listener === cb) _listener = null; };
}

export function getEntries(): LogEntry[] {
  return [..._entries];
}
