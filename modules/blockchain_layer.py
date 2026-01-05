"""
Module 4: Blockchain Traceability Layer
Provides immutable proof, ownership, and timestamp.
For hackathon: Uses mock/testnet blockchain.
"""

import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import json


class BlockchainTraceability:
    """
    Blockchain layer for storing fingerprints and protection records.
    MVP: Mock blockchain for hackathon demonstration.
    """
    
    def __init__(self, use_testnet: bool = True):
        """
        Initialize blockchain layer.
        
        Args:
            use_testnet: If True, use mock/testnet (for hackathon).
                        If False, would connect to real blockchain.
        """
        self.use_testnet = use_testnet
        self.mock_chain = []  # In-memory mock blockchain
        self.last_block_hash = "0" * 64  # Genesis block hash
    
    def register_fingerprint(self, 
                            fingerprint_data: Dict[str, Any],
                            protection_flag: bool = False,
                            consent_status: bool = True) -> Dict[str, Any]:
        """
        Register fingerprint on blockchain.
        
        Returns:
            {
                'transaction_id': str,
                'block_number': int,
                'timestamp': str (ISO format),
                'fingerprint_id': str,
                'protection_flag': bool,
                'consent_status': bool,
                'blockchain_type': str
            }
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Create transaction data
        transaction_data = {
            'fingerprint_id': fingerprint_data.get('fingerprint_id'),
            'sha256_hash': fingerprint_data.get('sha256'),
            'perceptual_hash': fingerprint_data.get('perceptual_hash'),
            'protection_flag': protection_flag,
            'consent_status': consent_status,
            'timestamp': timestamp,
            'metadata_hash': fingerprint_data.get('metadata_hash')
        }
        
        # Create block
        block = self._create_block(transaction_data)
        
        # Add to chain
        self.mock_chain.append(block)
        self.last_block_hash = block['block_hash']
        
        return {
            'transaction_id': block['transaction_id'],
            'block_number': block['block_number'],
            'timestamp': timestamp,
            'fingerprint_id': fingerprint_data.get('fingerprint_id'),
            'protection_flag': protection_flag,
            'consent_status': consent_status,
            'blockchain_type': 'mock_testnet' if self.use_testnet else 'production',
            'block_hash': block['block_hash'],
            'previous_hash': block['previous_hash']
        }
    
    def _create_block(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new block in the chain"""
        block_number = len(self.mock_chain)
        
        # Create transaction ID
        tx_data_str = json.dumps(transaction_data, sort_keys=True)
        transaction_id = hashlib.sha256(tx_data_str.encode()).hexdigest()
        
        # Create block header
        block_header = {
            'block_number': block_number,
            'previous_hash': self.last_block_hash,
            'timestamp': transaction_data['timestamp'],
            'transaction_id': transaction_id,
            'merkle_root': transaction_id  # Simplified for MVP
        }
        
        # Calculate block hash
        block_data_str = json.dumps(block_header, sort_keys=True)
        block_hash = hashlib.sha256(block_data_str.encode()).hexdigest()
        
        return {
            'block_number': block_number,
            'block_hash': block_hash,
            'previous_hash': self.last_block_hash,
            'transaction_id': transaction_id,
            'transaction_data': transaction_data,
            'timestamp': transaction_data['timestamp']
        }
    
    def verify_transaction(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Verify a transaction exists on the blockchain"""
        for block in self.mock_chain:
            if block['transaction_id'] == transaction_id:
                return {
                    'verified': True,
                    'block_number': block['block_number'],
                    'timestamp': block['timestamp'],
                    'block_hash': block['block_hash']
                }
        return {'verified': False}
    
    def get_chain_info(self) -> Dict[str, Any]:
        """Get blockchain information"""
        return {
            'chain_length': len(self.mock_chain),
            'last_block_hash': self.last_block_hash,
            'blockchain_type': 'mock_testnet' if self.use_testnet else 'production',
            'genesis_block': self.mock_chain[0] if self.mock_chain else None
        }

