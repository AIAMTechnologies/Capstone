import React, { useState } from 'react';
import { updateFinalSelection } from '../services/api';
import type { AlternativeInstaller } from '../types';

interface FinalSelectionDropdownImprovedProps {
  leadId: number;
  currentSelection?: number;
  assignedInstallerId?: number;
  assignedInstallerName?: string;
  assignedInstallerCity?: string;
  alternativeInstallers?: AlternativeInstaller[];
  onSelectionChange?: () => void;
}

const FinalSelectionDropdownImproved: React.FC<FinalSelectionDropdownImprovedProps> = ({
  leadId,
  currentSelection,
  assignedInstallerId,
  assignedInstallerName,
  assignedInstallerCity,
  alternativeInstallers = [],
  onSelectionChange
}) => {
  // Default to ML assignment if no selection made yet
  const [selectedId, setSelectedId] = useState<number | undefined>(
    currentSelection || assignedInstallerId
  );
  const [isUpdating, setIsUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSelectionChange = async (event: React.ChangeEvent<HTMLSelectElement>) => {
    const newInstallerId = parseInt(event.target.value);
    setSelectedId(newInstallerId);
    setIsUpdating(true);
    setError(null);

    try {
      await updateFinalSelection(leadId, newInstallerId);
      if (onSelectionChange) {
        onSelectionChange();
      }
    } catch (err) {
      setError('Failed to update selection');
      console.error('Error updating final selection:', err);
      // Revert the selection on error
      setSelectedId(currentSelection || assignedInstallerId);
    } finally {
      setIsUpdating(false);
    }
  };

  // Build options array: ML assignment first, then alternatives
  const options = [];
  
  // Add ML assignment
  if (assignedInstallerId && assignedInstallerName) {
    options.push({
      id: assignedInstallerId,
      label: `${assignedInstallerName} (ML Assignment)${assignedInstallerCity ? ` - ${assignedInstallerCity}` : ''}`,
      isML: true
    });
  }
  
  // Add alternatives (excluding ML if it's in alternatives)
  if (alternativeInstallers && alternativeInstallers.length > 0) {
    alternativeInstallers.forEach(alt => {
      if (alt.id !== assignedInstallerId) {
        options.push({
          id: alt.id,
          label: `${alt.name} (${alt.city}) - ${alt.distance_km.toFixed(1)}km`,
          isML: false
        });
      }
    });
  }

  if (options.length === 0) {
    return (
      <div style={{ fontSize: '12px', color: '#95a5a6' }}>
        No options available
      </div>
    );
  }

  return (
    <div style={{ position: 'relative' }}>
      <select
        value={selectedId || ''}
        onChange={handleSelectionChange}
        disabled={isUpdating}
        style={{
          width: '100%',
          padding: '6px 8px',
          fontSize: '13px',
          border: '1px solid #ddd',
          borderRadius: '4px',
          backgroundColor: selectedId === assignedInstallerId ? '#e3f2fd' : '#fff',
          cursor: isUpdating ? 'not-allowed' : 'pointer',
          opacity: isUpdating ? 0.6 : 1,
          minWidth: '200px'
        }}
      >
        {options.map((option) => (
          <option 
            key={option.id} 
            value={option.id}
            style={{ 
              fontWeight: option.isML ? 'bold' : 'normal'
            }}
          >
            {option.label}
          </option>
        ))}
      </select>
      
      {isUpdating && (
        <div style={{ 
          position: 'absolute', 
          right: '8px', 
          top: '50%', 
          transform: 'translateY(-50%)',
          width: '14px',
          height: '14px',
          border: '2px solid #3498db',
          borderTop: '2px solid transparent',
          borderRadius: '50%',
          animation: 'spin 0.8s linear infinite'
        }} />
      )}
      
      {error && (
        <div style={{ 
          fontSize: '11px', 
          color: '#e74c3c', 
          marginTop: '2px' 
        }}>
          {error}
        </div>
      )}
      
      {selectedId === assignedInstallerId && !error && (
        <div style={{ 
          fontSize: '11px', 
          color: '#3498db', 
          marginTop: '2px' 
        }}>
          ✓ Using ML recommendation
        </div>
      )}
      
      <style>{`
        @keyframes spin {
          0% { transform: translateY(-50%) rotate(0deg); }
          100% { transform: translateY(-50%) rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default FinalSelectionDropdownImproved;