import { NextRequest, NextResponse } from "next/server";
import { execFile } from "child_process";
import path from "path";
import { promisify } from "util";

const execFileAsync = promisify(execFile);

const PYTHON_SCRIPT = path.resolve(process.cwd(), "live_scanner.py");
const TIMEOUT_MS = 180000; // 3 minutes for scanning all stocks

export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url);
    const lookback = parseInt(searchParams.get("lookback") || "10") || 10;
    const args = [PYTHON_SCRIPT];
    if (lookback !== 10) args.push("--lookback", String(lookback));
    const { stdout, stderr } = await execFileAsync("python", args, {
      timeout: TIMEOUT_MS,
      maxBuffer: 50 * 1024 * 1024,
    });

    // stderr has progress info, stdout has final JSON
    try {
      const data = JSON.parse(stdout);
      return NextResponse.json(data);
    } catch (e) {
      return NextResponse.json({ error: "Failed to parse scanner output", detail: stdout.slice(0, 500) }, { status: 500 });
    }
  } catch (err: any) {
    console.error("Live scanner error:", err.message);
    return NextResponse.json({ error: `Live scanner failed: ${err.message}` }, { status: 500 });
  }
}
