import { Line } from "react-chartjs-2";

export default function CombinedTrendGraph({ trends }) {
    return (
        <div className="mt-8 bg-gray-900 p-4 rounded-md">
            <h2 className="text-xl text-white mb-4">Combined Trend Graphs</h2>
            <div style={{ height: "400px" }}>
                <Line
                    data={{
                        labels: trends.map((t) => t.date),
                        datasets: Object.keys(trends[0])
                            .filter(k => k !== "date")
                            .map((term, index) => ({
                                label: term,
                                data: trends.map((t) => t[term]),
                                borderColor: `rgba(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255}, 1)`, // Random colors
                                backgroundColor: `rgba(${Math.random() * 255}, ${Math.random() * 255}, ${Math.random() * 255}, 0.5)`,
                                borderWidth: 2,
                                pointRadius: 3,
                            })),
                    }}
                    options={{
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: { display: true }, // Show legend for multiple datasets
                            zoom: {
                                pan: { enabled: true, mode: "x" },
                                zoom: { wheel: { enabled: true }, mode: "x" },
                            },
                        },
                        scales: {
                            x: {
                                ticks: {
                                    color: "#fff", // White text for x-axis labels
                                },
                            },
                            y: {
                                ticks: {
                                    color: "#fff", // White text for y-axis labels
                                },
                            },
                        },
                    }}
                />
            </div>
        </div>
    );
}