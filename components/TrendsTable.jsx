export default function TrendsTable({ trends }) {
    return (
        <div className="mt-8 bg-gray-900 p-4 rounded-md overflow-x-auto max-h-[50vh]">
            <h2 className="text-xl text-white mb-4">Trend Results</h2>
            <table className="w-full text-white border-collapse">
                <thead>
                    <tr className="border-b">
                        <th className="p-2">Date</th>
                        {Object.keys(trends[0]).filter(k => k !== "date").map(term => (
                            <th key={term} className="p-2">{term} trend score</th>
                        ))}
                    </tr>
                </thead>
                <tbody className="text-center">
                    {trends.map((row, index) => (
                        <tr key={index} className="border-b">
                            <td className="p-2">{row.date}</td>
                            {Object.keys(row).filter(k => k !== "date").map(term => (
                                <td key={term} className="p-2">{row[term]}</td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
}