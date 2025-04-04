"use client";

import { useState, useEffect } from "react";
import { Chart as ChartJS } from "chart.js";
import { LineElement, PointElement, LinearScale, Title, Tooltip, Legend, CategoryScale } from "chart.js";
import SearchForm from "@/components/SearchForm";
import TrendsTable from "@/components/TrendsTable";
import CombinedTrendGraph from "@/components/CombinedTrendGraph";

// Register Chart.js components
ChartJS.register(LineElement, PointElement, LinearScale, CategoryScale, Title, Tooltip, Legend);

export default function Data() {
    const [query, setQuery] = useState("");
    const [startDate, setStartDate] = useState("");
    const [endDate, setEndDate] = useState("");
    const [trends, setTrends] = useState([]);
    const [prediction, setPrediction] = useState(null);
    const [zoomLoaded, setZoomLoaded] = useState(false);
    const [predictKeyword, setPredictKeyword] = useState("");

    useEffect(() => {
        import("chartjs-plugin-zoom").then((zoomPlugin) => {
            ChartJS.register(zoomPlugin.default);
            setZoomLoaded(true);
        }).catch(console.error);
    }, []);

    const fetchTrends = async (e) => {
        e.preventDefault();
        if (!query.trim()) return;

        const res = await fetch(
            `/api/trends?keywords=${encodeURIComponent(query)}&start_date=${startDate}&end_date=${endDate}`
        );
        const data = await res.json();

        if (data.error) {
            setTrends([]);
            return;
        }

        const formattedTrends = data.dates.map((date, index) => {
            let row = { date };
            Object.keys(data).forEach((key) => {
                if (key !== "dates") row[key] = data[key][index];
            });
            return row;
        });

        setTrends(formattedTrends);
    };

    const fetchPrediction = async (e) => {
        e.preventDefault();
        if (!predictKeyword.trim()) return;

        const res = await fetch(`/api/predict?keyword=${encodeURIComponent(predictKeyword)}`);
        const data = await res.json();

        setPrediction(data);
    };

    return (
        <main className="min-h-screen w-full flex flex-col items-center md:px-[10vw] md:py-8 bg-black">
            <div className="w-full md:p-16 flex flex-col bg-neutral-800 rounded-2xl">
                <h1 className="md:text-2xl my-4 font-medium text-white">
                    Search Keywords to Analyze Trend Pattern
                </h1>

                <SearchForm
                    query={query}
                    startDate={startDate}
                    endDate={endDate}
                    onQueryChange={setQuery}
                    onStartChange={setStartDate}
                    onEndChange={setEndDate}
                    onSubmit={fetchTrends}
                />

                {trends.length > 0 && <TrendsTable trends={trends} />}
                {trends.length > 0 && zoomLoaded && <CombinedTrendGraph trends={trends} />}
            </div>

            {/* Prediction Form */}
            <div className="w-full md:p-8 mt-8 flex flex-col bg-neutral-800 rounded-2xl">
                <h2 className="md:text-xl my-4 font-medium text-white">Predict Trend</h2>
                <form className="flex flex-col gap-4" onSubmit={fetchPrediction}>
                    <input
                        type="text"
                        value={predictKeyword}
                        onChange={(e) => setPredictKeyword(e.target.value)}
                        placeholder="Enter keyword for prediction"
                        className="p-2 rounded-md bg-neutral-700 text-white"
                    />
                    <button
                        type="submit"
                        className="p-2 bg-blue-500 text-white rounded-md hover:bg-blue-600"
                    >
                        Get Prediction
                    </button>
                </form>

                {prediction && (
                    <div className="mt-4 text-white">
                        <h3 className="text-lg font-medium">Prediction Result:</h3>
                        <pre className="bg-gray-900 p-4 rounded-md">
                            {JSON.stringify(prediction, null, 2)}
                        </pre>
                    </div>
                )}
            </div>
        </main>
    );
}
