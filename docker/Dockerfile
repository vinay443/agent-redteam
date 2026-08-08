# =============================================================================
# Target agent image.
#
# This image runs ONLY the agent under test. It contains no shell tools the
# agent can reach (the agent has no shell tool), no build toolchain, and a
# non-root user that owns nothing outside /app and the bind-mounted /sandbox.
#
# The real containment guarantees are enforced at run time by docker-compose
# (no host mounts except ./sandbox, resource limits, dropped capabilities,
# read-only root fs). See docker/docker-compose.yml and SAFETY.md.
# =============================================================================
FROM python:3.12-slim AS base

# Never write .pyc, never buffer stdout/stderr — the runner parses our stdout.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Non-root user. UID/GID are overridable at build time so that, on Linux hosts,
# the container user can own files it writes into the bind-mounted ./sandbox.
ARG AGENT_UID=10001
ARG AGENT_GID=10001
RUN groupadd --gid "${AGENT_GID}" agent \
    && useradd --uid "${AGENT_UID}" --gid "${AGENT_GID}" --create-home --shell /usr/sbin/nologin agent

WORKDIR /app

# Install only the target agent's dependencies (minimal by design).
COPY requirements-target.txt /app/requirements-target.txt
RUN pip install --no-cache-dir -r /app/requirements-target.txt

# Copy only the packages the container needs. The attacker, judge, runner and
# report packages never enter the image — the target must not be able to import
# its own grader.
COPY common /app/common
COPY target_agent /app/target_agent

# The sandbox is a bind mount at run time; create the mountpoint and hand it to
# the agent user so writes succeed even before the first attack.
RUN mkdir -p /sandbox && chown -R agent:agent /sandbox /app

USER agent

ENV AGENT_SANDBOX_ROOT=/sandbox \
    PYTHONPATH=/app

# Default command idles; the runner drives the agent via `docker exec`, one
# invocation per attack. `tail -f /dev/null` keeps the container alive with no
# work and no listening socket.
CMD ["python", "-c", "import time;\nwhile True: time.sleep(3600)"]
