# Pod SSH Instructions

Status: debug/development notes for disposable GPU pods.

The normal production stage image should not start SSH by default. Production
jobs should run the stage command, upload results to R2, and exit. SSH is a
debug convenience for manual smoke tests, interactive diagnosis, `scp`, or
Remote SSH access.

## Mental Model

There are three different access modes:

```text
Web Terminal:
  No SSH key required.
  No exposed SSH port required.
  Requires the container to stay running.

True SSH:
  Requires your public SSH key.
  Requires TCP port 22 exposed in the pod/template.
  Requires openssh-server/sshd running inside the container.

Production job:
  No terminal.
  No SSH daemon.
  No exposed ports.
  Controller starts the container with a stage command and reads results from R2.
```

## Why Web Terminal May Close Immediately

The current COLMAP image was built as a runtime image and its default command is
effectively:

```text
colmap -h
```

That prints help and exits. If the container has no long-running process,
RunPod may start the Web Terminal spinner and then turn it off again because
there is nothing stable to attach to.

For manual debugging, override the container start command so the pod stays
alive:

```bash
bash -lc 'sleep infinity'
```

Or, for the current R2 smoke-test workflow:

```bash
bash -lc 'apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends awscli git ca-certificates && \
  sleep infinity'
```

After the pod is running, open the Web Terminal and run the stage commands
manually.

## Creating The SSH Key On Your Mac

RunPod asks for a public key. Paste the `.pub` file contents, never the private
key.

Check for an existing public key:

```bash
ls ~/.ssh/*.pub
```

Print the public key:

```bash
cat ~/.ssh/id_ed25519.pub
```

The value should be one line starting with:

```text
ssh-ed25519
```

If you do not have an Ed25519 key yet:

```bash
ssh-keygen -t ed25519 -C "blackrock.jmk@gmail.com"
```

Press Enter for the default path unless you intentionally want a separate key:

```text
~/.ssh/id_ed25519
```

Then print and paste:

```bash
cat ~/.ssh/id_ed25519.pub
```

## True SSH Requirements

For true SSH, configure the RunPod template/pod with:

```text
Expose TCP port: 22
Environment variable:
  PUBLIC_KEY=<full ssh-ed25519 public key line>
```

The public port shown by RunPod will not usually be `22`. RunPod maps an
external port to internal container port `22`.

Use this debug start command:

```bash
bash -lc 'apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server awscli git ca-certificates && \
  mkdir -p /var/run/sshd ~/.ssh && \
  chmod 700 ~/.ssh && \
  if [ -n "$PUBLIC_KEY" ]; then echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys; fi && \
  chmod 600 ~/.ssh/authorized_keys || true && \
  service ssh start && \
  sleep infinity'
```

Then connect with the command RunPod shows, shaped like:

```bash
ssh root@<public-ip> -p <mapped-port> -i ~/.ssh/id_ed25519
```

If SSH refuses the key because of permissions:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
```

## Recommended Current COLMAP Smoke-Test Access

For the R2-backed COLMAP smoke test, prefer Web Terminal first:

```text
Template start command: keep container alive with sleep infinity
No exposed ports needed
No SSH daemon needed
Run commands manually in Web Terminal
```

Use true SSH only if Web Terminal is unavailable or if you need `scp`, SFTP, or
Remote SSH from an editor.

## Future Image Policy

Keep the normal processing image clean:

```text
No sshd by default.
No exposed ports by default.
Stage command runs and exits.
```

If SSH setup becomes too slow during development, make a separate debug image
tag later:

```text
buildvision3d-colmap-gpu:debug-ssh
```

Do not mix debug SSH behavior into the production runner tag.
