import { useState } from 'react';
import Head from 'next/head';

export default function Home() {
  return (
    <>
      <Head>
        <title>Crime Caster - Toronto</title>
        <meta name="description" content="Toronto crime risk prediction" />
      </Head>
      <main className="min-h-screen bg-gray-100">
        <div className="container mx-auto px-4 py-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Crime Caster - Toronto
          </h1>
          <p className="text-lg text-gray-600 mb-8">
            3D interactive map for crime risk prediction
          </p>
          
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-2xl font-semibold mb-4">Status</h2>
            <div className="space-y-2">
              <p>✅ API Backend: Ready</p>
              <p>⏳ Frontend: In Development</p>
              <p>⏳ 3D Map: Coming Soon</p>
            </div>
          </div>

          <div className="mt-8 bg-yellow-50 border border-yellow-200 rounded-lg p-4">
            <p className="text-sm text-yellow-800">
              <strong>Note:</strong> This is a work in progress. The 3D map visualization will be added next.
            </p>
          </div>
        </div>
      </main>
    </>
  );
}

