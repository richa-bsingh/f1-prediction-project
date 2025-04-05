import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line } from 'recharts';

const F1Dashboard = () => {
  const [grandsPrix, setGrandsPrix] = useState([]);
  const [availableGPs, setAvailableGPs] = useState([]);
  const [activeGP, setActiveGP] = useState(null);
  const [isAddingGP, setIsAddingGP] = useState(false);
  const [newGPName, setNewGPName] = useState('');
  const [driverData, setDriverData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [apiBaseUrl, setApiBaseUrl] = useState('http://localhost:5001/api');

  // F1 official colors - using 2023/2024 brand colors
  const f1Colors = {
    primary: '#E10600', // F1 red
    secondary: '#000000', // F1 black
    tertiary: '#FFFFFF', // White
    background: '#FFFFFF', // White background
    cardBg: '#F8F8F8', // Light gray for cards
    textPrimary: '#15151E', // Dark blue/black
    textSecondary: '#67676F' // Medium gray
  };

  // F1 team colors for consistent styling - 2024 season
  const teamColors = {
    'McLaren': '#FF8700',
    'Red Bull': '#0600EF',
    'Mercedes': '#00D2BE',
    'Ferrari': '#DC0000',
    'RB': '#2B4562', // Formerly AlphaTauri
    'Williams': '#005AFF',
    'Alpine': '#0090FF',
    'Aston Martin': '#006F62',
    'Haas': '#FFFFFF',
    'Sauber': '#900000', // Formerly Alfa Romeo
    'default': '#333333' // Default color if team not specified
  };

  // Driver to team mapping - 2024 season teams
  const driverTeams = {
    'Lando Norris': 'McLaren',
    'Oscar Piastri': 'McLaren',
    'Max Verstappen': 'Red Bull',
    'Sergio Perez': 'Red Bull',
    'George Russell': 'Mercedes',
    'Lewis Hamilton': 'Mercedes',
    'Charles Leclerc': 'Ferrari',
    'Carlos Sainz': 'Ferrari',
    'Yuki Tsunoda': 'RB',
    'Daniel Ricciardo': 'RB',
    'Alexander Albon': 'Williams',
    'Logan Sargeant': 'Williams',
    'Pierre Gasly': 'Alpine',
    'Esteban Ocon': 'Alpine',
    'Fernando Alonso': 'Aston Martin',
    'Lance Stroll': 'Aston Martin',
    'Nico Hulkenberg': 'Haas',
    'Kevin Magnussen': 'Haas',
    'Valtteri Bottas': 'Sauber',
    'Zhou Guanyu': 'Sauber'
  };

  // Function to get color based on driver's team
  const getDriverColor = (driverName) => {
    const team = driverTeams[driverName];
    return team ? teamColors[team] : teamColors.default;
  };

  // Function to get country flag emoji for GP
  const getGPFlag = (gpName) => {
    const flagMap = {
      'Australia': '🇦🇺',
      'Austria': '🇦🇹',
      'Azerbaijan': '🇦🇿',
      'Bahrain': '🇧🇭',
      'Belgium': '🇧🇪',
      'Brazil': '🇧🇷',
      'Canada': '🇨🇦',
      'China': '🇨🇳',
      'Emilia Romagna': '🇮🇹',
      'France': '🇫🇷',
      'Great Britain': '🇬🇧',
      'Hungary': '🇭🇺',
      'Italy': '🇮🇹',
      'Japan': '🇯🇵',
      'Las Vegas': '🇺🇸',
      'Mexico': '🇲🇽',
      'Mexico City': '🇲🇽',
      'Miami': '🇺🇸',
      'Monaco': '🇲🇨',
      'Netherlands': '🇳🇱',
      'Portugal': '🇵🇹',
      'Qatar': '🇶🇦',
      'Saudi Arabia': '🇸🇦',
      'Singapore': '🇸🇬',
      'Spain': '🇪🇸',
      'United States': '🇺🇸',
      'USA': '🇺🇸',
      'Abu Dhabi': '🇦🇪'
    };
    
    return flagMap[gpName] || '🏁';
  };
  
  // Load available GP list from API
  const fetchAvailableGPs = async () => {
    try {
      const response = await fetch(`${apiBaseUrl}/gp-list`);
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      setAvailableGPs(data.gpList || []);
    } catch (error) {
      console.error('Error fetching GP list:', error);
      setError('Failed to fetch available Grand Prix list. Please check if the API server is running.');
      
      // Fallback to a basic list
      setAvailableGPs([
        'Australia', 'Bahrain', 'Saudi Arabia', 'Japan', 'China', 
        'Miami', 'Emilia Romagna', 'Monaco', 'Canada', 'Spain'
      ]);
    }
  };

  // Fetch predictions for a GP from the API
  const fetchPredictions = async (gpName) => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ gpName }),
      });
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Check if there's an error in the response
      if (data.error) {
        throw new Error(data.error);
      }
      
      // Format the data for our dashboard
      return data.map(item => ({
        driver: item.driver,
        qualifyingTime: item.qualifyingTime,
        predictedTime: item.predictedTime,
        gap: item.gap,
        position: item.position
      }));
      
    } catch (error) {
      console.error(`Error fetching predictions for ${gpName}:`, error);
      setError(`Failed to get predictions for ${gpName}. ${error.message}`);
      return [];
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // Load available GPs when component mounts
    fetchAvailableGPs();
    
    // Initial load with Japan GP (as in your original Python script)
    const loadInitialData = async () => {
      try {
        const predictions = await fetchPredictions('Japan');
        
        if (predictions.length > 0) {
          // Create Japan GP object
          const japanGP = {
            name: 'Japan',
            year: 2025,
            date: '2025-04-10',
            predictions
          };
          
          setGrandsPrix([japanGP]);
          setActiveGP(japanGP);
          setDriverData(japanGP.predictions);
        }
      } catch (error) {
        console.error('Error loading initial data:', error);
        setError('Failed to load initial Japan GP data. Please check if the API server is running.');
      } finally {
        setLoading(false);
      }
    };
    
    loadInitialData();
  }, []);

  // Add a new Grand Prix
  const handleAddGP = async () => {
    if (!newGPName.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Call the API to get predictions for the new GP
      const predictions = await fetchPredictions(newGPName);
      
      if (predictions && predictions.length > 0) {
        // Create new GP object
        const newGP = {
          name: newGPName,
          year: 2025,
          date: new Date().toISOString().split('T')[0],
          predictions
        };
        
        setGrandsPrix([...grandsPrix, newGP]);
        setActiveGP(newGP);
        setDriverData(newGP.predictions);
        setNewGPName('');
        setIsAddingGP(false);
      } else {
        setError(`Could not get predictions for ${newGPName}. Please try another Grand Prix.`);
      }
    } catch (error) {
      console.error(`Error adding ${newGPName}:`, error);
      setError(`Failed to add ${newGPName}. ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  // Handle clicking on a GP to show its data
  const handleGPClick = (gp) => {
    setActiveGP(gp);
    setDriverData(gp.predictions);
  };

  // For comparing qualifying vs predicted race time
  const prepareComparisonData = () => {
    return driverData.map(d => ({
      name: d.driver.split(' ')[1], // Just last name for cleaner display
      qualifying: d.qualifyingTime,
      predicted: d.predictedTime,
      fullName: d.driver,
      color: getDriverColor(d.driver)
    }));
  };

  // For driver standings across all GPs
  const prepareDriverStandings = () => {
    // Create a map to store each driver's positions across all GPs
    const driverPositions = {};
    
    grandsPrix.forEach(gp => {
      // Assign positions
      gp.predictions.forEach((prediction) => {
        const driver = prediction.driver;
        const position = prediction.position || 0;
        
        if (!driverPositions[driver]) {
          driverPositions[driver] = [];
        }
        
        driverPositions[driver].push({ 
          gp: gp.name, 
          position 
        });
      });
    });
    
    // Convert to array for chart
    return Object.entries(driverPositions).map(([driver, positions]) => {
      return {
        driver,
        positions,
        color: getDriverColor(driver),
        avgPosition: positions.reduce((sum, p) => sum + p.position, 0) / positions.length
      };
    }).sort((a, b) => a.avgPosition - b.avgPosition);
  };

  if (loading && grandsPrix.length === 0) {
    return (
      <div className="flex justify-center items-center h-screen" style={{ 
        backgroundColor: f1Colors.background,
        fontFamily: "'Titillium Web', 'Formula1', sans-serif"
      }}>
        <div className="text-lg" style={{ color: f1Colors.primary }}>Loading... 🏎️</div>
      </div>
    );
  }

  return (
    <div style={{ 
      backgroundColor: f1Colors.background, 
      color: f1Colors.textPrimary,
      fontFamily: "'Titillium Web', 'Formula1', sans-serif"
    }} className="min-h-screen">
      {/* Custom font imports */}
      <style>
        {`
          @import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@400;600;700;900&display=swap');
          
          @font-face {
            font-family: 'Formula1';
            src: url('https://www.formula1.com/etc/designs/fom-website/fonts/F1Regular/Formula1-Regular.ttf') format('truetype');
            font-weight: normal;
            font-style: normal;
            font-display: swap;
          }
          
          h1, h2, h3, h4, .f1-font {
            font-family: 'Titillium Web', 'Formula1', sans-serif;
            font-weight: 700;
            letter-spacing: 0.05em;
          }
          
          .f1-title {
            font-family: 'Titillium Web', 'Formula1', sans-serif;
            font-weight: 900;
            letter-spacing: 0.05em;
            text-transform: uppercase;
          }
        `}
      </style>
      
      {/* Header with minimalist F1 style */}
      <div style={{ backgroundColor: f1Colors.primary }} className="py-4 px-6">
        <div className="max-w-6xl mx-auto flex items-center justify-between">
          <h1 className="text-2xl font-bold text-white tracking-wider f1-title flex items-center">
            <span className="mr-2">🏎️</span>
            F1 PREDICTIONS 2025
            <span className="ml-2">🏁</span>
          </h1>
          <div className="text-white text-sm">Machine Learning Race Simulator</div>
        </div>
      </div>
      
      <div className="max-w-6xl mx-auto p-4">
        {/* Error message display */}
        {error && (
          <div className="mb-4 p-3 rounded" style={{ backgroundColor: '#FFDDDD', color: '#D8000C' }}>
            <p className="flex items-center">
              <span className="mr-2">⚠️</span> {error}
            </p>
          </div>
        )}
        
        {/* Grand Prix Selection */}
        <div className="mb-8">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold f1-font" style={{ color: f1Colors.textPrimary }}>
              <span className="mr-1">🏁</span> GRAND PRIX
            </h2>
            <button 
              onClick={() => setIsAddingGP(!isAddingGP)}
              style={{ backgroundColor: f1Colors.primary, color: f1Colors.tertiary }}
              className="px-4 py-2 text-sm font-bold rounded f1-font"
              disabled={loading}
            >
              {isAddingGP ? 'CANCEL' : '+ ADD GRAND PRIX'}
            </button>
          </div>
          
          {isAddingGP && (
            <div className="mb-4">
              <div className="flex gap-2 mb-2">
                <select
                  value={newGPName}
                  onChange={(e) => setNewGPName(e.target.value)}
                  className="flex-grow p-2 rounded border"
                  style={{ 
                    borderColor: f1Colors.textSecondary,
                    fontFamily: "'Titillium Web', sans-serif"
                  }}
                  disabled={loading}
                >
                  <option value="">Select a Grand Prix</option>
                  {availableGPs.map(gp => (
                    <option key={gp} value={gp}>{gp}</option>
                  ))}
                </select>
                <button 
                  onClick={handleAddGP}
                  style={{ backgroundColor: f1Colors.primary, color: f1Colors.tertiary }}
                  className="px-4 py-2 text-sm font-bold rounded f1-font"
                  disabled={!newGPName || loading}
                >
                  {loading ? 'LOADING...' : 'ADD'}
                </button>
              </div>
              <p className="text-sm" style={{ color: f1Colors.textSecondary }}>
                Note: Adding a new Grand Prix will run the ML model to predict results based on historical data.
              </p>
            </div>
          )}
          
          <div className="flex flex-wrap gap-2">
            {grandsPrix.map(gp => (
              <button
                key={gp.name}
                onClick={() => handleGPClick(gp)}
                className="py-2 px-4 text-sm font-bold rounded transition f1-font flex items-center"
                style={{ 
                  backgroundColor: activeGP && activeGP.name === gp.name ? f1Colors.primary : f1Colors.cardBg,
                  color: activeGP && activeGP.name === gp.name ? f1Colors.tertiary : f1Colors.textPrimary,
                  border: `1px solid ${f1Colors.textSecondary}`
                }}
              >
                <span className="mr-1">{getGPFlag(gp.name)}</span>
                {gp.name.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        
        {activeGP && (
          <>
            {/* Active GP Info */}
            <div className="mb-6 pb-2 border-b" style={{ borderColor: f1Colors.textSecondary }}>
              <h2 className="text-2xl font-bold f1-title flex items-center" style={{ color: f1Colors.textPrimary }}>
                <span className="mr-2">{getGPFlag(activeGP.name)}</span>
                {activeGP.name.toUpperCase()} GRAND PRIX
                <span className="ml-2">🏎️</span>
              </h2>
              <p style={{ color: f1Colors.textSecondary }}>2025 Predicted Results</p>
            </div>
            
            {/* Loading indicator for data fetching */}
            {loading ? (
              <div className="flex justify-center items-center p-12">
                <div className="text-lg" style={{ color: f1Colors.primary }}>Loading predictions... 🏎️</div>
              </div>
            ) : (
              <>
                {/* Winner Prediction - Prominently displayed */}
                {driverData.length > 0 && (
                  <div className="mb-8 p-6 rounded text-center" style={{ 
                    backgroundColor: f1Colors.cardBg,
                    boxShadow: `0 4px 6px -1px rgba(225, 6, 0, 0.1), 0 2px 4px -1px rgba(225, 6, 0, 0.06)`
                  }}>
                    <h3 className="text-sm font-bold mb-1 f1-font" style={{ color: f1Colors.textSecondary }}>
                      <span className="mr-1">🏆</span> PREDICTED WINNER <span className="ml-1">🏆</span>
                    </h3>
                    <div className="text-3xl font-bold mb-1 f1-title" style={{ color: getDriverColor(driverData[0]?.driver) }}>
                      {driverData[0]?.driver}
                    </div>
                    <p style={{ color: f1Colors.textSecondary }}>
                      Predicted Lap Time: <span className="font-semibold">{driverData[0]?.predictedTime.toFixed(3)}s</span>
                    </p>
                  </div>
                )}
                
                {/* Race Prediction Charts in minimal style */}
                {driverData.length > 0 && (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                    {/* Predicted Race Times */}
                    <div className="p-4 rounded" style={{ backgroundColor: f1Colors.cardBg }}>
                      <h3 className="text-sm font-bold mb-4 uppercase f1-font flex items-center" style={{ color: f1Colors.textSecondary }}>
                        <span className="mr-1">🏎️</span> Predicted Race Pace
                      </h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart
                          data={driverData}
                          layout="vertical"
                          margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                          <XAxis type="number" domain={['dataMin - 0.2', 'dataMax + 0.2']} />
                          <YAxis 
                            dataKey="driver" 
                            type="category" 
                            width={80}
                            tick={{ fontSize: 12, fontFamily: "'Titillium Web', sans-serif" }}
                            tickFormatter={(value) => value.split(' ')[1]} // Just last name
                          />
                          <Tooltip 
                            contentStyle={{ 
                              backgroundColor: f1Colors.background, 
                              borderColor: f1Colors.textSecondary,
                              fontFamily: "'Titillium Web', sans-serif" 
                            }}
                            formatter={(value) => [`${value.toFixed(3)}s`, 'Predicted Time']}
                            labelFormatter={(label) => label} // Full driver name
                          />
                          <Bar 
                            dataKey="predictedTime" 
                            name="Predicted Lap Time" 
                            isAnimationActive={false}
                          >
                            {driverData.map((entry, index) => (
                              <rect key={`cell-${index}`} fill={getDriverColor(entry.driver)} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    
                    {/* Qualifying vs Race Time Comparison */}
                    <div className="p-4 rounded" style={{ backgroundColor: f1Colors.cardBg }}>
                      <h3 className="text-sm font-bold mb-4 uppercase f1-font flex items-center" style={{ color: f1Colors.textSecondary }}>
                        <span className="mr-1">⏱️</span> Qualifying vs Race Pace
                      </h3>
                      <ResponsiveContainer width="100%" height={300}>
  <BarChart
    data={prepareComparisonData()}
    margin={{ top: 5, right: 30, left: 0, bottom: 30 }}
  >
    <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
    <XAxis 
      dataKey="name" 
      angle={0} 
      tick={{ fontSize: 12, fontFamily: "'Titillium Web', sans-serif" }}
    />
    <YAxis 
      domain={[
        dataMin => Math.floor(dataMin - 1), 
        dataMax => Math.ceil(dataMax + 1)
      ]}
      tickFormatter={(value) => value.toFixed(1)}
    />
    <Tooltip 
      contentStyle={{ 
        backgroundColor: f1Colors.background, 
        borderColor: f1Colors.textSecondary,
        fontFamily: "'Titillium Web', sans-serif" 
      }}
      formatter={(value, name) => [`${value.toFixed(3)}s`, name === 'qualifying' ? 'Qualifying' : 'Race Pace']}
      labelFormatter={(label, data) => data[0]?.payload?.fullName}
    />
    <Legend />
    <Bar dataKey="qualifying" name="Qualifying" fill="#1E1E1E" />
    <Bar dataKey="predicted" name="Race Pace" fill={f1Colors.primary} />
  </BarChart>
</ResponsiveContainer>
                    </div>
                  </div>
                )}
                
                {/* Season Progress Visualization - only show if we have multiple GPs */}
                {grandsPrix.length > 1 && (
                  <div className="p-4 mb-8 rounded" style={{ backgroundColor: f1Colors.cardBg }}>
                    <h3 className="text-sm font-bold mb-4 uppercase f1-font flex items-center" style={{ color: f1Colors.textSecondary }}>
                      <span className="mr-1">📊</span> Season Performance
                    </h3>
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart
                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                        <XAxis 
                          dataKey="name" 
                          type="category" 
                          allowDuplicatedCategory={false} 
                          categories={grandsPrix.map(gp => gp.name)}
                          tick={{ fontFamily: "'Titillium Web', sans-serif" }}
                        />
                        <YAxis 
                          domain={[1, 12]} 
                          reversed={true}
                          tick={{ fontFamily: "'Titillium Web', sans-serif" }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: f1Colors.background, 
                            borderColor: f1Colors.textSecondary,
                            fontFamily: "'Titillium Web', sans-serif" 
                          }}
                        />
                        <Legend />
                        
                        {prepareDriverStandings().slice(0, 8).map((driver) => (
                          <Line
                            key={driver.driver}
                            name={driver.driver.split(' ')[1]} // Just last name for legend
                            dataKey="position"
                            data={driver.positions}
                            stroke={driver.color}
                            strokeWidth={2}
                            dot={{ fill: driver.color, r: 4 }}
                            isAnimationActive={false}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}
                
                {/* Additional race data in a clean table */}
                {driverData.length > 0 && (
                  <div className="mb-8 p-4 rounded" style={{ backgroundColor: f1Colors.cardBg }}>
                    <h3 className="text-sm font-bold mb-4 uppercase f1-font flex items-center" style={{ color: f1Colors.textSecondary }}>
                      <span className="mr-1">📋</span> Complete Results
                    </h3>
                    <div className="overflow-x-auto">
                      <table className="w-full" style={{ 
                        borderCollapse: 'collapse',
                        fontFamily: "'Titillium Web', sans-serif" 
                      }}>
                        <thead>
                          <tr style={{ borderBottom: `1px solid ${f1Colors.textSecondary}` }}>
                            <th className="py-2 px-4 text-left">POS</th>
                            <th className="py-2 px-4 text-left">DRIVER</th>
                            <th className="py-2 px-4 text-right">QUALIFYING</th>
                            <th className="py-2 px-4 text-right">PREDICTED PACE</th>
                            <th className="py-2 px-4 text-right">GAP TO P1</th>
                          </tr>
                        </thead>
                        <tbody>
                          {driverData.map((driver, index) => (
                            <tr 
                              key={driver.driver} 
                              style={{ 
                                borderBottom: `1px solid ${f1Colors.textSecondary}20`,
                                backgroundColor: index === 0 ? `${f1Colors.primary}10` : 'transparent'
                              }}
                            >
                              <td className="py-2 px-4 text-left font-bold">{index + 1}</td>
                              <td className="py-2 px-4 text-left">
                                <div className="flex items-center">
                                  <div 
                                    className="w-3 h-3 rounded-full mr-2" 
                                    style={{ backgroundColor: getDriverColor(driver.driver) }}
                                  ></div>
                                  {driver.driver}
                                </div>
                              </td>
                              <td className="py-2 px-4 text-right">{driver.qualifyingTime.toFixed(3)}s</td>
                              <td className="py-2 px-4 text-right font-semibold">{driver.predictedTime.toFixed(3)}s</td>
                              <td className="py-2 px-4 text-right">
                                {index === 0 ? '-' : `+${driver.gap.toFixed(3)}s`}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </>
        )}
        
        {/* Instructions section */}
        <div className="mt-8 p-4 rounded" style={{ backgroundColor: f1Colors.cardBg }}>
          <h3 className="text-sm font-bold mb-2 uppercase f1-font flex items-center" style={{ color: f1Colors.textSecondary }}>
            <span className="mr-1">ℹ️</span> How To Use
          </h3>
          <ul className="list-disc pl-5 space-y-1 text-sm" style={{ 
            color: f1Colors.textPrimary,
            fontFamily: "'Titillium Web', sans-serif" 
          }}>
            <li>Click "ADD GRAND PRIX" to predict results for a new race</li>
            <li>Select a Grand Prix from the dropdown menu</li>
            <li>The dashboard will run the ML model to generate predictions</li>
            <li>Click on any GP button to switch between races</li>
            <li>The season performance chart updates automatically as you add more races</li>
          </ul>
          
          <div className="mt-4 p-3 rounded" style={{ backgroundColor: '#E6F7FF' }}>
            <p className="text-sm flex items-start">
              <span className="mr-2 text-lg">💡</span> 
              <span>This dashboard uses FastF1 API and machine learning to predict race results based on historical data. For the best experience, run the Python backend server before using this dashboard.</span>
            </p>
          </div>
        </div>
      </div>
      
      {/* Footer */}
      <div className="py-4 text-center text-sm" style={{ 
        color: f1Colors.textSecondary, 
        borderTop: `1px solid ${f1Colors.textSecondary}20`,
        fontFamily: "'Titillium Web', sans-serif"
      }}>
        <span className="mr-2">🏎️ </span>
        F1 2025 PREDICTION DASHBOARD • Copyright © 2025 by Richa Singh
        <span className="ml-2"> 🏁 </span>
      </div>
    </div>
  );
};

export default F1Dashboard;