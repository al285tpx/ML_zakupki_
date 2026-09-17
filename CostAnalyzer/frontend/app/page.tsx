import SearchForm from '@/components/SearchForm';

export default function LandingPage() {
  return (
    <main className="min-h-screen bg-slate-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-5xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-extrabold text-slate-900 sm:text-5xl mb-4">
            Government Contract Analyzer
          </h1>
          <p className="text-lg text-slate-600 max-w-2xl mx-auto">
            Search for government contracts, analyze product pricing, and discover OKPD2 classifications with ML-powered insights.
          </p>
        </div>

        <div className="bg-white p-1 rounded-2xl shadow-xl border border-slate-200">
          <SearchForm />
        </div>

        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
          <div className="p-6 bg-white rounded-xl border border-slate-200 shadow-sm">
            <div className="text-blue-600 text-2xl mb-2">⚡</div>
            <h3 className="font-bold text-slate-900 mb-2">Fast Search</h3>
            <p className="text-sm text-slate-500">Asynchronous processing of thousands of contracts in seconds.</p>
          </div>
          <div className="p-6 bg-white rounded-xl border border-slate-200 shadow-sm">
            <div className="text-blue-600 text-2xl mb-2">📊</div>
            <h3 className="font-bold text-slate-900 mb-2">Price Quartiles</h3>
            <p className="text-sm text-slate-500">Automatic pricing analysis grouped by unit of measurement.</p>
          </div>
          <div className="p-6 bg-white rounded-xl border border-slate-200 shadow-sm">
            <div className="text-blue-600 text-2xl mb-2">🤖</div>
            <h3 className="font-bold text-slate-900 mb-2">ML Classification</h3>
            <p className="text-sm text-slate-500">Naive Bayes classification for missing OKPD2 product codes.</p>
          </div>
        </div>
      </div>
    </main>
  );
}
