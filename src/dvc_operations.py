"""
DVC operations module for IRIS pipeline.
Handles DVC remote operations and data/model fetching.
"""

import os
import subprocess
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DVCOperations:
    """Handles DVC operations for data and model versioning."""
    
    def __init__(self, remote_name: str = "gcsremote"):
        """
        Initialize DVCOperations.
        
        Args:
            remote_name: Name of the DVC remote
        """
        self.remote_name = remote_name
    
    def run_dvc_command(self, command: str) -> tuple:
        """
        Run a DVC command and return the result.
        
        Args:
            command: DVC command to run
            
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                check=False
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            logger.error(f"Error running DVC command '{command}': {e}")
            raise
    
    def pull_data(self, file_path: str) -> bool:
        """
        Pull data from DVC remote.
        
        Args:
            file_path: Path to the file to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            returncode, stdout, stderr = self.run_dvc_command(f"dvc pull {file_path}")
            if returncode == 0:
                logger.info(f"Successfully pulled {file_path}")
                return True
            else:
                logger.error(f"Failed to pull {file_path}: {stderr}")
                return False
        except Exception as e:
            logger.error(f"Error pulling data: {e}")
            return False
    
    def pull_model(self, model_path: str) -> bool:
        """
        Pull model from DVC remote.
        
        Args:
            model_path: Path to the model file to pull
            
        Returns:
            True if successful, False otherwise
        """
        try:
            returncode, stdout, stderr = self.run_dvc_command(f"dvc pull {model_path}")
            if returncode == 0:
                logger.info(f"Successfully pulled {model_path}")
                return True
            else:
                logger.error(f"Failed to pull {model_path}: {stderr}")
                return False
        except Exception as e:
            logger.error(f"Error pulling model: {e}")
            return False
    
    def checkout_version(self, version: str) -> bool:
        """
        Checkout a specific version using git and dvc.
        
        Args:
            version: Git tag or commit hash
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # First checkout the git version
            returncode, stdout, stderr = self.run_dvc_command(f"git checkout {version}")
            if returncode != 0:
                logger.error(f"Failed to checkout git version {version}: {stderr}")
                return False
            
            # Then checkout the DVC files
            returncode, stdout, stderr = self.run_dvc_command("dvc checkout")
            if returncode == 0:
                logger.info(f"Successfully checked out version {version}")
                return True
            else:
                logger.error(f"Failed to checkout DVC files: {stderr}")
                return False
        except Exception as e:
            logger.error(f"Error checking out version: {e}")
            return False
    
    def list_remotes(self) -> list:
        """
        List available DVC remotes.
        
        Returns:
            List of remote names
        """
        try:
            returncode, stdout, stderr = self.run_dvc_command("dvc remote list")
            if returncode == 0:
                remotes = [line.strip() for line in stdout.split('\n') if line.strip()]
                logger.info(f"Available remotes: {remotes}")
                return remotes
            else:
                logger.error(f"Failed to list remotes: {stderr}")
                return []
        except Exception as e:
            logger.error(f"Error listing remotes: {e}")
            return []
    
    def setup_remote(self, remote_url: str) -> bool:
        """
        Setup DVC remote if not already configured.
        
        Args:
            remote_url: URL of the remote storage
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if remote already exists
            remotes = self.list_remotes()
            if self.remote_name in remotes:
                logger.info(f"Remote {self.remote_name} already exists")
                return True
            
            # Add remote
            returncode, stdout, stderr = self.run_dvc_command(f"dvc remote add -d {self.remote_name} {remote_url}")
            if returncode == 0:
                logger.info(f"Successfully added remote {self.remote_name}")
                return True
            else:
                logger.error(f"Failed to add remote: {stderr}")
                return False
        except Exception as e:
            logger.error(f"Error setting up remote: {e}")
            return False
