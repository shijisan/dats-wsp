import { NextResponse } from "next/server";

export async function GET(req) {
    const { searchParams } = new URL(req.url);
    const keyword = searchParams.get("keyword");

    if (!keyword) {
        return NextResponse.json({ error: "Keyword is required" }, { status: 400 });
    }

    try {
        const response = await fetch(`http://127.0.0.1:8000/api/predict?keyword=${encodeURIComponent(keyword)}`);
        const data = await response.json();

        return NextResponse.json(data);
    } catch (error) {
        return NextResponse.json({ error: "Failed to fetch prediction" }, { status: 500 });
    }
}
