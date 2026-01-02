-- Multi-Agent Portfolio Advisor - PostgreSQL Initialization
-- This script runs once when the PostgreSQL container is first created.
-- It creates separate databases for Langfuse and the Portfolio Advisor application.

-- Create database for Langfuse observability platform
CREATE DATABASE langfuse;

-- Create database for Portfolio Advisor application state
CREATE DATABASE portfolio_advisor;

-- Grant privileges (using the default postgres user)
GRANT ALL PRIVILEGES ON DATABASE langfuse TO postgres;
GRANT ALL PRIVILEGES ON DATABASE portfolio_advisor TO postgres;

-- Log completion
\echo 'Databases created: langfuse, portfolio_advisor'
