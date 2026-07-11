-- runsautomatically the first time the container initializes its data volume.
-- If you ever need to re-run this (e.g. after schema changes), the easiest
-- path is `docker compose down -v` to wipe the volume and start fresh —
-- fine during development, never do this once there's real data you care about.
 
CREATE EXTENSION IF NOT EXISTS vector;
 
