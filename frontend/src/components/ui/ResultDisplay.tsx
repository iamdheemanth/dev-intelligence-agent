import { FileText, Lightbulb, AlertTriangle, CheckSquare, Code } from 'lucide-react';

interface ResultDisplayProps {
  action: string;
  data: any;
}

export const ResultDisplay = ({ action, data }: ResultDisplayProps) => {
  const renderSummarize = (data: any) => {
    if (typeof data === 'string') {
      return <div className="prose max-w-none text-gray-900 whitespace-pre-wrap leading-relaxed">{data}</div>;
    }
    if (data.summary) {
      return <div className="prose max-w-none text-gray-900 whitespace-pre-wrap leading-relaxed">{data.summary}</div>;
    }
    if (data.text) {
      return <div className="prose max-w-none text-gray-900 whitespace-pre-wrap leading-relaxed">{data.text}</div>;
    }
    return <pre className="text-sm whitespace-pre-wrap text-gray-900 bg-gray-100 p-4 rounded border border-gray-300">{JSON.stringify(data, null, 2)}</pre>;
  };

  const renderRecommend = (data: any) => {
    if (data.results && Array.isArray(data.results)) {
      return (
        <div className="space-y-3">
          <p className="text-sm text-gray-700 mb-3 font-medium">Query: {data.query || 'Recommendations'}</p>
          {data.results.map((item: any, idx: number) => (
            <div key={idx} className="p-4 bg-gray-50 rounded-lg border border-gray-300 shadow-sm">
              <h5 className="font-semibold mb-2 text-gray-900">
                {item.library || item.name || item.title || `Recommendation ${idx + 1}`}
              </h5>
              {item.description && (
                <p className="text-sm text-gray-800 mb-2 leading-relaxed">{item.description}</p>
              )}
              {item.reason && (
                <p className="text-sm text-gray-700 mb-2 leading-relaxed">{item.reason}</p>
              )}
              {item.use_case && (
                <p className="text-sm text-gray-700 mb-2 leading-relaxed">
                  <span className="font-medium text-gray-900">Use case: </span>{item.use_case}
                </p>
              )}
              {item.score !== undefined && (
                <p className="text-xs text-gray-600">Relevance: {(item.score * 100).toFixed(1)}%</p>
              )}
              {item.url && (
                <a href={item.url} target="_blank" rel="noopener noreferrer" className="text-sm text-blue-600 hover:underline mt-2 inline-block font-medium">
                  Learn more →
                </a>
              )}
            </div>
          ))}
        </div>
      );
    }
    return <pre className="text-sm whitespace-pre-wrap text-gray-900 bg-gray-100 p-4 rounded border border-gray-300">{JSON.stringify(data, null, 2)}</pre>;
  };

  const renderTriage = (data: any) => {
    // Handle different response structures
    let parsedData = data;
    
    // Check if there's a classification object
    if (data.classification && typeof data.classification === 'object') {
      parsedData = data.classification;
    }

    // Extract all possible field variations
    const label = parsedData.label || parsedData.priority || parsedData.category || data.label;
    const score = parsedData.score !== undefined ? parsedData.score : data.score;
    
    // Try to get suggested actions from multiple possible locations
    let suggestedActions = [];
    if (parsedData.suggested_actions && Array.isArray(parsedData.suggested_actions)) {
      suggestedActions = parsedData.suggested_actions;
    } else if (data.suggested_actions && Array.isArray(data.suggested_actions)) {
      suggestedActions = data.suggested_actions;
    } else if (parsedData.actions && Array.isArray(parsedData.actions)) {
      suggestedActions = parsedData.actions;
    } else if (data.actions && Array.isArray(data.actions)) {
      suggestedActions = data.actions;
    }

    return (
      <div className="space-y-5">
        {/* Classification Label */}
        {label && (
          <div className="p-5 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-700 mb-2">Classification</p>
                <p className="text-2xl font-bold text-blue-700 capitalize">{label.replace(/-/g, ' ')}</p>
              </div>
              {score !== undefined && (
                <div className="text-right">
                  <p className="text-sm text-gray-600 mb-1">Confidence Score</p>
                  <p className="text-3xl font-bold text-gray-900">{(score * 100).toFixed(0)}%</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Suggested Actions */}
        {suggestedActions.length > 0 && (
          <div>
            <h4 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
              <span className="text-xl mr-2">📋</span>
              Suggested Actions
            </h4>
            <ul className="space-y-3">
              {suggestedActions.map((action: string, idx: number) => (
                <li key={idx} className="flex items-start space-x-3 p-4 bg-white rounded-lg border border-gray-300 shadow-sm hover:shadow-md transition-shadow">
                  <span className="flex-shrink-0 w-7 h-7 bg-gray-900 text-white rounded-full flex items-center justify-center text-sm font-bold">
                    {idx + 1}
                  </span>
                  <p className="text-sm text-gray-800 leading-relaxed pt-0.5 flex-1">{action}</p>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Other fields */}
        {(parsedData.reasoning || data.reasoning) && (
          <div className="p-4 bg-gray-50 rounded-lg border border-gray-300">
            <p className="text-sm font-semibold text-gray-900 mb-2 flex items-center">
              <span className="text-lg mr-2">💭</span>
              Reasoning
            </p>
            <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
              {parsedData.reasoning || data.reasoning}
            </p>
          </div>
        )}

        {(parsedData.explanation || data.explanation) && (
          <div className="p-4 bg-gray-50 rounded-lg border border-gray-300">
            <p className="text-sm font-semibold text-gray-900 mb-2 flex items-center">
              <span className="text-lg mr-2">ℹ️</span>
              Explanation
            </p>
            <p className="text-sm text-gray-800 leading-relaxed whitespace-pre-wrap">
              {parsedData.explanation || data.explanation}
            </p>
          </div>
        )}

        {/* Debug: Show raw JSON if nothing was parsed */}
        {!label && suggestedActions.length === 0 && !parsedData.reasoning && !parsedData.explanation && (
          <div>
            <p className="text-sm text-red-600 mb-2">⚠️ Unable to parse response. Showing raw data:</p>
            <pre className="text-sm whitespace-pre-wrap text-gray-900 bg-gray-100 p-4 rounded border border-gray-300 overflow-x-auto">
{JSON.stringify(data, null, 2)}
            </pre>
          </div>
        )}
      </div>
    );
  };

  const renderRefactor = (data: any) => {
    let suggestions = data.suggestions;
    
    // Parse if suggestions is a string
    if (typeof suggestions === 'string') {
      try {
        suggestions = JSON.parse(suggestions);
      } catch (e) {
        return <pre className="text-sm whitespace-pre-wrap text-gray-900 bg-gray-100 p-4 rounded border border-gray-300">{suggestions}</pre>;
      }
    }
    
    if (Array.isArray(suggestions)) {
      return (
        <div className="space-y-4">
          <p className="text-sm text-gray-700 font-medium">
            Found {suggestions.length} refactoring suggestion{suggestions.length !== 1 ? 's' : ''}
          </p>
          {suggestions.map((suggestion: any, idx: number) => {
            // Parse if suggestion is a string
            let parsedSuggestion = suggestion;
            if (typeof suggestion === 'string') {
              try {
                parsedSuggestion = JSON.parse(suggestion);
              } catch (e) {
                parsedSuggestion = { explanation: suggestion };
              }
            }
            
            return (
              <div key={idx} className="p-5 bg-gray-50 rounded-lg border border-gray-300 shadow-sm">
                {/* File Header */}
                <div className="flex items-start space-x-3 mb-4 pb-3 border-b border-gray-200">
                  <Code size={20} className="text-blue-600 mt-1 flex-shrink-0" />
                  <div className="flex-1">
                    <h5 className="font-semibold text-lg text-gray-900">
                      {parsedSuggestion.file || parsedSuggestion.filename || `Suggestion ${idx + 1}`}
                    </h5>
                  </div>
                </div>
                
                {/* Explanation */}
                {parsedSuggestion.explanation && (
                  <div className="mb-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
                    <p className="font-semibold text-sm text-gray-900 mb-2 flex items-center">
                      <span className="text-lg mr-2">💡</span>
                      Explanation
                    </p>
                    <p className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
                      {parsedSuggestion.explanation}
                    </p>
                  </div>
                )}

                {/* Example */}
                {parsedSuggestion.example && (
                  <div>
                    <p className="font-semibold text-sm text-gray-900 mb-2 flex items-center">
                      <span className="text-lg mr-2">📝</span>
                      Example
                    </p>
                    <pre className="text-sm bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto font-mono border border-gray-700">
{parsedSuggestion.example}</pre>
                  </div>
                )}

                {/* Fallback: show raw if no recognized fields */}
                {!parsedSuggestion.explanation && !parsedSuggestion.example && (
                  <pre className="text-sm whitespace-pre-wrap text-gray-800 bg-gray-100 p-3 rounded border border-gray-300">
{JSON.stringify(parsedSuggestion, null, 2)}</pre>
                )}
              </div>
            );
          })}
        </div>
      );
    }
    
    return <pre className="text-sm whitespace-pre-wrap text-gray-900 bg-gray-100 p-4 rounded border border-gray-300">{JSON.stringify(data, null, 2)}</pre>;
  };

  const getIcon = () => {
    switch (action) {
      case 'summarize': return <FileText size={20} className="text-blue-600" />;
      case 'recommend': return <Lightbulb size={20} className="text-yellow-600" />;
      case 'triage': return <AlertTriangle size={20} className="text-orange-600" />;
      case 'review': return <CheckSquare size={20} className="text-green-600" />;
      case 'refactor': return <Code size={20} className="text-purple-600" />;
      default: return null;
    }
  };

  const renderContent = () => {
    switch (action) {
      case 'summarize': return renderSummarize(data);
      case 'recommend': return renderRecommend(data);
      case 'triage': return renderTriage(data);
      case 'refactor': return renderRefactor(data);
      default: return <pre className="text-sm whitespace-pre-wrap text-gray-900 bg-gray-100 p-4 rounded border border-gray-300">{JSON.stringify(data, null, 2)}</pre>;
    }
  };

  return (
    <div className="mt-6">
      <div className="flex items-center space-x-2 mb-4">
        {getIcon()}
        <h4 className="font-semibold text-lg capitalize text-gray-900">{action} Results</h4>
      </div>
      <div className="p-6 bg-white rounded-lg border border-gray-300 shadow-sm max-h-[700px] overflow-auto">
        {renderContent()}
      </div>
    </div>
  );
};
