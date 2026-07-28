# SSH Access -- NHR@FAU HPC Documentation

Source: https://doc.nhr.fau.de/access/ssh-command-line/

## Overview

This documentation explains how to use OpenSSH to connect to NHR@FAU HPC systems across Linux, Mac, and Windows platforms.

## SSH Key Pair Generation

Create an SSH key pair using ED25519 format (RSA 4096-bit and ECDSA 512-bit also accepted):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_nhr_fau
```

**Requirements:**
- A strong passphrase is mandatory
- Private key permissions: `600`
- Public key permissions: `644`
- `.ssh` directory permissions: `700`

The process generates two files:
- Private key: `~/.ssh/id_ed25519_nhr_fau` (keep secret locally)
- Public key: `~/.ssh/id_ed25519_nhr_fau.pub` (upload to HPC Portal)

## Uploading Public Key

Follow HPC Portal instructions to upload your public SSH key through the User tab. Key distribution to all systems takes up to two hours.

## SSH Configuration

Add configuration templates to `~/.ssh/config` (no file extension).

### Required Adjustments

- Replace `<HPC ACCOUNT>` with your account name from HPC Portal
- Update `IdentityFile` path if using a different key filename
- Include the `csnhr.nhr.fau.de` entry if any section contains its `ProxyJump`

### HPC Systems Template

```
Host csnhr.nhr.fau.de csnhr
    HostName csnhr.nhr.fau.de
    User <HPC account>
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
    PreferredAuthentications publickey
    ForwardX11 no
    ForwardX11Trusted no

Host fritz.nhr.fau.de fritz
    HostName fritz.nhr.fau.de
    User <HPC account>
    ProxyJump csnhr.nhr.fau.de
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
    PreferredAuthentications publickey
    ForwardX11 no
    ForwardX11Trusted no

Host alex.nhr.fau.de alex
    HostName alex.nhr.fau.de
    User <HPC account>
    ProxyJump csnhr.nhr.fau.de
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
    PreferredAuthentications publickey
    ForwardX11 no
    ForwardX11Trusted no

Host helma.nhr.fau.de helma
    HostName helma.nhr.fau.de
    User <HPC account>
    ProxyJump csnhr.nhr.fau.de
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
    PreferredAuthentications publickey
    ForwardX11 no
    ForwardX11Trusted no

Host woody.nhr.fau.de woody
    HostName woody.nhr.fau.de
    User <HPC account>
    ProxyJump csnhr.nhr.fau.de
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
    PreferredAuthentications publickey
    ForwardX11 no
    ForwardX11Trusted no

# Fritz cluster nodes
Host f????.nhr.fau.de
    User <HPC account>
    ProxyJump csnhr.nhr.fau.de
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
    PreferredAuthentications publickey

# Alex cluster nodes
Host a????.nhr.fau.de
    User <HPC account>
    ProxyJump csnhr.nhr.fau.de
    IdentityFile ~/.ssh/id_ed25519_nhr_fau
    IdentitiesOnly yes
    PasswordAuthentication no
    PreferredAuthentications publickey
```

## Connection Testing

1. Test connection to the dialog server:
   ```bash
   ssh csnhr.nhr.fau.de
   ```

2. Verify the host key fingerprint against documented values

3. Connect to cluster frontends:
   ```bash
   ssh fritz.nhr.fau.de
   ssh alex.nhr.fau.de
   ssh helma.nhr.fau.de
   ```

## Port Forwarding Setup

Create SSH tunnels to access remote services (e.g., Jupyter Notebooks):

```bash
ssh -L <local port>:localhost:<remote port> <remote server>
```

**Example:** Access Jupyter running on `a123.nhr.fau.de:54321` locally on port `12345`:

```bash
ssh -L 12345:localhost:54321 a123.nhr.fau.de
```

Then navigate to `http://localhost:12345` in your browser.

## Troubleshooting

### Enable Debug Output

```bash
ssh -vv <your options>   # detailed debug output
```

### Verify SSH Key Fingerprints

```bash
# MD5 format
ssh-keygen -E MD5 -l -f ~/.ssh/id_ed25519_nhr_fau

# SHA256 format
ssh-keygen -E SHA256 -l -f ~/.ssh/id_ed25519_nhr_fau
```

### Common Issues

- "sign_and_send_pubkey: signing failed ... agent refused operation" -- typically indicates an incorrect passphrase entry
- Password prompt at `csnhr`: The dialog server does not retain SSH keys for subsequent connections; configure SSH proxy jump or use SSH agent forwarding
- New accounts become usable the following morning after file systems and Slurm databases are updated

## Support

For unresolved connection issues, contact `hpc-support@fau.de` with the command invocation and output including `-vv` flags.
