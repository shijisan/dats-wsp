"use client";

export default function SearchForm({ query, startDate, endDate, onQueryChange, onStartChange, onEndChange, onSubmit }) {
    // Helper function to calculate 6 months ago from today
    const getSixMonthsAgo = () => {
        const sixMonthsAgo = new Date();
        sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);
        return sixMonthsAgo.toISOString().split("T")[0]; // Format as YYYY-MM-DD
    };

    // Get today's date in YYYY-MM-DD format
    const today = new Date().toISOString().split("T")[0];

    return (
        <form className="w-full flex flex-wrap items-center md:gap-4" onSubmit={onSubmit}>
            {/* Keywords Input */}
            <div className="flex flex-col w-full md:w-1/3">
                <label className="mb-1 text-white" htmlFor="keywords">Keywords</label>
                <input
                    className="p-2 rounded-md"
                    id="keywords"
                    type="text"
                    value={query}
                    onChange={(e) => onQueryChange(e.target.value)}
                    placeholder="Enter keywords separated with commas..."
                />
            </div>

            {/* Start Date Input */}
            <div className="flex flex-col">
                <label className="mb-1 text-white" htmlFor="startDate">Start Date</label>
                <input
                    className="p-2 rounded-md"
                    type="date"
                    id="startDate"
                    value={startDate}
                    onChange={(e) => onStartChange(e.target.value)}
                    min={getSixMonthsAgo()} // Set minimum date to 6 months ago
                    max={today} // Set maximum date to today
                />
            </div>

            {/* End Date Input */}
            <div className="flex flex-col">
                <label className="mb-1 text-white" htmlFor="endDate">End Date</label>
                <input
                    className="p-2 rounded-md"
                    type="date"
                    id="endDate"
                    value={endDate}
                    onChange={(e) => onEndChange(e.target.value)}
                    min={getSixMonthsAgo()} // Set minimum date to 6 months ago
                    max={today} // Set maximum date to today
                />
            </div>

            {/* Submit Button */}
            <div>
                <button
                    type="submit"
                    className="py-3 px-6 bg-blue-500 text-white rounded-md hover:brightness-95 transition-all focus:opacity-50"
                >
                    Process
                </button>
            </div>
        </form>
    );
}