# Security Policy

Omni Slated Mem Core is a local memory service. Its security properties are
user data isolation (`user_id` namespacing), retrieval integrity, and
determinism. A bug that breaks any of these is a security issue, not a
normal bug.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Use GitHub's private vulnerability reporting on this repository
(Security tab → "Report a vulnerability"). You can also email
corey@slidphilabs.com with the subject line `omni-slated-mem-core security`.

Include the affected file or endpoint, steps or inputs to reproduce, and
what you expected versus what happened.

You can expect an acknowledgement within 3 business days. We will keep you
updated while we investigate and credit you in the changelog unless you
prefer to stay anonymous.

## In scope

- `user_id` isolation bypass: any input where a user retrieves another
  user's memories
- Retrieval integrity: planted or poisoned results, forged supersede links,
  relevance-gate bypass
- Determinism violations that could mask tampering
- Vendored model or wheel substitution going undetected

## Out of scope

- Operator deployments we do not run
- Social engineering, spam, or denial-of-service against hosted demos
