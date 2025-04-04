export async function GET(req) {
    const { searchParams } = new URL(req.url);
    const keywords = searchParams.get("keywords");
    const startDate = searchParams.get("start_date");
    const endDate = searchParams.get("end_date");

    const res = await fetch(`http://127.0.0.1:8000/api/trends?keywords=${keywords}&start_date=${startDate}&end_date=${endDate}`);
    const data = await res.json();

    return Response.json(data);
}
