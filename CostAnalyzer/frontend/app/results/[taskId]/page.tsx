"use client";

import React, { useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import ContractTable from '@/components/ContractTable';

interface TaskStatus {
  task_id: string;
  status: 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';
  result_url: string | null;
  error: string | null;
}

export default function ResultsPage() {
  const params = useParams();
  const router = useRouter();
  const taskId = params.taskId as string;

  const [status, setStatus] = useState<TaskStatus | null>(null);
  const [data, setData] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let pollInterval: NodeJS.Timeout;

    const checkStatus = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/status/${taskId}`);
        if (!response.ok) throw new Error('Status check failed');

        const statusData: TaskStatus = await response.json();
        setStatus(statusData);

        if (statusData.status === 'COMPLETED') {
          // Fetch final results
          const resultResponse = await fetch(`http://localhost:8000/api/results/${taskId}`);
          if (!resultResponse.ok) throw new Error('Results fetch failed');
          const resultData = await resultResponse.json();
          setData(resultData);
          setIsLoading(false);
          clearInterval(pollInterval);
        } else if (statusData.status === 'FAILED') {
          setIsLoading(false);
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error('Polling error:', error);
        setIsLoading(false);
        clearInterval(pollInterval);
      }
    };

    // Initial check
    checkStatus();

    // Poll every 3 seconds
    pollInterval = setInterval(checkStatus, 3000);

    return () => clearInterval(pollInterval);
  }, [taskId]);

  if (!status || isLoading) {
    return (
      <div className="min-h-screen bg-slate-50 flex flex-col items-center justify-center p-4">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-2xl font-bold text-slate-900 mb-2">Analyzing Contracts...</h2>
          <p className="text-slate-500">Our ML pipeline is fetching data and calculating quartiles. Please wait.</p>
          <p className="text-xs text-slate-400 mt-4 font-mono">Task ID: {taskId}</p>
        </div>
      </div>
    );
  }

  if (status.status === 'FAILED') {
    return (
      <div className="min-h-screen bg-slate-50 flex items-center justify-center p-4">
        <div className="bg-white p-8 rounded-xl shadow-sm border border-red-200 max-w-md text-center">
          <div className="text-red-500 text-4xl mb-4">⚠️</div>
          <h2 className="text-xl font-bold text-slate-900 mb-2">Analysis Failed</h2>
          <p className="text-slate-600 mb-6">{status.error || 'An unexpected error occurred during processing.'}</p>
          <button
            onClick={() => router.push('/')}
            className="bg-slate-900 text-white px-4 py-2 rounded-md hover:bg-slate-800 transition-colors"
          >
            Return to Search
          </button>
        </div>
      </div>
    );
  }

  return (
    <main className="min-h-screen bg-slate-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-3xl font-extrabold text-slate-900">Analysis Results</h1>
            <p className="text-slate-500">Found {data.length} matching product positions</p>
          </div>
          <button
            onClick={() => router.push('/')}
            className="bg-white border border-slate-300 text-slate-700 px-4 py-2 rounded-md hover:bg-slate-50 transition-colors font-medium"
          >
            New Search
          </button>
        </div>

        <ContractTable data={data} />
      </div>
    </main>
  );
}
