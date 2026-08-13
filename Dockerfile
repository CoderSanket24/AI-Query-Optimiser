# Use the official PostgreSQL 18 image as the base
FROM postgres:18

# Update the package list and install the pg_hint_plan extension for PG 18
RUN apt-get update && \
    apt-get install -y postgresql-18-pg-hint-plan && \
    rm -rf /var/lib/apt/lists/*

# Add a configuration to preload the extension when the database starts
RUN echo "shared_preload_libraries = 'pg_hint_plan'" >> /usr/share/postgresql/postgresql.conf.sample